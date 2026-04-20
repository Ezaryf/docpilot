import asyncio
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag.llm import classify_vllm_startup_error, parse_openai_models_response


LocalModelState = Literal[
    "idle",
    "checking_docker",
    "starting_docker",
    "starting_container",
    "loading_model",
    "ready",
    "failed",
]

GpuMemoryMode = Literal["safe_10gb", "balanced", "max_context"]

CONTAINER_NAME = "docpilot-vllm"
BASE_URL = "http://localhost:8001/v1"
HOST_PORT = "8001"
VLLM_IMAGE = "vllm/vllm-openai:latest"
READY_TIMEOUT_SECONDS = int(os.getenv("DOCPILOT_VLLM_READY_TIMEOUT", "900"))


class LocalModelStatus(TypedDict):
    state: LocalModelState
    model: str
    served_model: str
    base_url: str
    container_name: str
    progress_message: str
    error_code: str
    error_message: str
    logs_tail: str
    hf_token_available: bool


_status: LocalModelStatus = {
    "state": "idle",
    "model": "",
    "served_model": "",
    "base_url": BASE_URL,
    "container_name": CONTAINER_NAME,
    "progress_message": "No local model has been requested yet.",
    "error_code": "",
    "error_message": "",
    "logs_tail": "",
    "hf_token_available": False,
}
_status_lock = asyncio.Lock()
_apply_task: asyncio.Task | None = None


def _config_dir() -> Path:
    return Path(os.getenv("DOCPILOT_HOME", Path.home() / ".docpilot"))


def _hf_token_path() -> Path:
    return _config_dir() / "hf_token"


def save_hf_token(token: str) -> None:
    cleaned = token.strip()
    if not cleaned:
        return

    config_dir = _config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    token_path = _hf_token_path()
    token_path.write_text(cleaned, encoding="utf-8")
    try:
        token_path.chmod(0o600)
    except OSError:
        pass


def load_hf_token() -> str:
    env_token = os.getenv("HF_TOKEN", "").strip()
    if env_token:
        return env_token

    try:
        return _hf_token_path().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def delete_hf_token() -> None:
    try:
        _hf_token_path().unlink()
    except FileNotFoundError:
        pass


def hf_token_available() -> bool:
    return bool(load_hf_token())


def _gpu_memory_defaults(mode: str) -> tuple[str, str]:
    if mode == "max_context":
        return "0.9", "8192"
    if mode == "balanced":
        return "0.85", "4096"
    return "0.82", "4096"


def build_vllm_docker_args(model: str, gpu_memory_mode: str = "safe_10gb") -> list[str]:
    gpu_memory_utilization, max_model_len = _gpu_memory_defaults(gpu_memory_mode)
    return [
        "docker",
        "run",
        "-d",
        "--gpus",
        "all",
        "--name",
        CONTAINER_NAME,
        "--rm",
        "-p",
        f"{HOST_PORT}:8000",
        "--ipc=host",
        "--env",
        "HF_TOKEN",
        "--label",
        "docpilot.managed=true",
        "--label",
        f"docpilot.model={model}",
        VLLM_IMAGE,
        model,
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--max-model-len",
        max_model_len,
    ]


def _redact_tokens(value: str) -> str:
    words: list[str] = []
    for word in value.split():
        if word.startswith(("hf_", "hf-")) and len(word) > 12:
            words.append(f"{word[:4]}...redacted")
        else:
            words.append(word)
    return " ".join(words)


def _run_sync(args: list[str], *, timeout: int = 30, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        command = " ".join(str(part) for part in args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout=(error.stdout or "") if isinstance(error.stdout, str) else "",
            stderr=f"{command} timed out after {timeout} seconds.",
        )
    except OSError as error:
        return subprocess.CompletedProcess(
            args=args,
            returncode=127,
            stdout="",
            stderr=str(error),
        )


async def _run(args: list[str], *, timeout: int = 30, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return await asyncio.to_thread(_run_sync, args, timeout=timeout, env=env)


def _docker_desktop_candidates() -> list[Path]:
    if platform.system().lower() != "windows":
        return []
    return [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe",
        Path(os.environ.get("LocalAppData", "")) / "Docker" / "Docker Desktop.exe",
    ]


def _try_start_docker_desktop() -> bool:
    for candidate in _docker_desktop_candidates():
        if candidate.exists():
            subprocess.Popen(
                [str(candidate)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            return True
    return False


async def _docker_status() -> tuple[bool, bool, str]:
    if shutil.which("docker") is None:
        return False, False, "Docker is not installed or not available on PATH."

    result = await _run(["docker", "ps"], timeout=10)
    details = _redact_tokens(f"{result.stdout}\n{result.stderr}".strip())
    return True, result.returncode == 0, details


async def _wait_for_docker(timeout_seconds: int = 120) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_seconds
    details = ""
    while time.monotonic() < deadline:
        _, running, details = await _docker_status()
        if running:
            return True, details
        if "docker" in details.lower() and "timed out" in details.lower():
            return False, details
        await asyncio.sleep(3)
    return False, details


async def _managed_container_exists() -> bool:
    result = await _run(
        ["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER_NAME}$", "--format", "{{.Names}}"],
        timeout=10,
    )
    return CONTAINER_NAME in result.stdout.splitlines()


async def _stop_managed_container() -> None:
    if not await _managed_container_exists():
        return
    await _run(["docker", "rm", "-f", CONTAINER_NAME], timeout=30)


async def _logs_tail() -> str:
    result = await _run(["docker", "logs", "--tail", "80", CONTAINER_NAME], timeout=10)
    return _redact_tokens(f"{result.stdout}\n{result.stderr}".strip())


def _models_url() -> str:
    return f"{BASE_URL}/models"


def _probe_models_sync() -> tuple[bool, list[str], str]:
    request = Request(_models_url(), method="GET")
    try:
        with urlopen(request, timeout=3) as response:
            body = response.read().decode("utf-8")
            return True, parse_openai_models_response(body), ""
    except HTTPError as error:
        return False, [], f"GET {_models_url()} returned HTTP {error.code}."
    except (URLError, TimeoutError, OSError) as error:
        return False, [], str(error)


async def _probe_models() -> tuple[bool, list[str], str]:
    return await asyncio.to_thread(_probe_models_sync)


async def _set_status(**updates) -> LocalModelStatus:
    async with _status_lock:
        _status.update(updates)
        _status["hf_token_available"] = hf_token_available()
        return dict(_status)


def _status_snapshot() -> LocalModelStatus:
    snapshot = dict(_status)
    snapshot["hf_token_available"] = hf_token_available()
    return snapshot


def _friendly_error_code_and_message(details: str) -> tuple[str, str]:
    lower = details.lower()
    if "docker is not installed" in lower or "not available on path" in lower:
        return "docker_missing", "Docker is not installed or not available on PATH."
    if "docker" in lower and "timed out" in lower:
        return "docker_timeout", "Docker did not respond in time. Start or restart Docker Desktop, then try again."
    if "daemon" in lower or "docker_engine" in lower or "permission denied" in lower:
        return "docker_daemon_not_running", "Docker Desktop is not running or DocPilot cannot access the Docker daemon."

    startup_kind, title, recommendation = classify_vllm_startup_error(details)
    if startup_kind == "gated_huggingface_model":
        return (
            "gated_model_auth_required",
            "The model is gated on Hugging Face. Add a valid read token in Settings and confirm the Hugging Face account accepted model access.",
        )
    if startup_kind == "gpu_memory_insufficient":
        return "gpu_memory_insufficient", recommendation

    if "no space left" in lower:
        return "disk_space_low", "Docker does not have enough disk space to pull or unpack the model image/files."
    return "local_model_start_failed", title if startup_kind != "unknown" else "DocPilot could not start the local model."


async def _fail(model: str, details: str) -> LocalModelStatus:
    logs = await _logs_tail() if shutil.which("docker") else ""
    combined = "\n".join(part for part in [details, logs] if part).strip()
    error_code, error_message = _friendly_error_code_and_message(combined)
    return await _set_status(
        state="failed",
        model=model,
        served_model="",
        progress_message=error_message,
        error_code=error_code,
        error_message=error_message,
        logs_tail=combined,
    )


async def _apply_model(model: str, gpu_memory_mode: str) -> None:
    await _set_status(
        state="checking_docker",
        model=model,
        served_model="",
        base_url=BASE_URL,
        container_name=CONTAINER_NAME,
        progress_message="Checking Docker Desktop...",
        error_code="",
        error_message="",
        logs_tail="",
    )

    docker_available, docker_running, docker_details = await _docker_status()
    if not docker_available:
        await _fail(model, docker_details)
        return

    if not docker_running:
        if "docker" in docker_details.lower() and "timed out" in docker_details.lower():
            await _fail(model, docker_details)
            return
        await _set_status(
            state="starting_docker",
            progress_message="Docker Desktop is not running. Trying to start it automatically...",
            logs_tail=docker_details,
        )
        _try_start_docker_desktop()
        docker_running, docker_details = await _wait_for_docker()
        if not docker_running:
            await _fail(model, docker_details)
            return

    reachable, served_models, _ = await _probe_models()
    if reachable and model in served_models:
        await _set_status(
            state="ready",
            model=model,
            served_model=model,
            progress_message=f"Local model {model} is ready.",
            error_code="",
            error_message="",
            logs_tail="",
        )
        return

    await _set_status(
        state="starting_container",
        progress_message=f"Starting local vLLM container for {model}...",
    )
    await _stop_managed_container()

    env = os.environ.copy()
    token = load_hf_token()
    if token:
        env["HF_TOKEN"] = token

    result = await _run(build_vllm_docker_args(model, gpu_memory_mode), timeout=120, env=env)
    if result.returncode != 0:
        await _fail(model, _redact_tokens(f"{result.stdout}\n{result.stderr}".strip()))
        return

    await _set_status(
        state="loading_model",
        progress_message=f"Downloading or loading {model}. This can take a few minutes the first time...",
        logs_tail=_redact_tokens(result.stdout.strip()),
    )

    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last_details = ""
    while time.monotonic() < deadline:
        reachable, served_models, details = await _probe_models()
        if reachable and model in served_models:
            await _set_status(
                state="ready",
                model=model,
                served_model=model,
                progress_message=f"Local model {model} is ready.",
                error_code="",
                error_message="",
                logs_tail=await _logs_tail(),
            )
            return

        logs = await _logs_tail()
        last_details = "\n".join(part for part in [details, logs] if part).strip()
        lower_details = last_details.lower()
        if any(
            fragment in lower_details
            for fragment in [
                "gatedrepoerror",
                "cannot access gated repo",
                "free memory on device cuda",
                "less than desired gpu memory utilization",
                "engine core initialization failed",
            ]
        ):
            await _fail(model, last_details)
            return

        await _set_status(
            state="loading_model",
            progress_message=f"Still loading {model}. Waiting for /v1/models to report it...",
            logs_tail=logs,
        )
        await asyncio.sleep(3)

    await _fail(model, last_details or f"Timed out waiting for {BASE_URL}/models to serve {model}.")


async def _safe_apply_model(model: str, gpu_memory_mode: str) -> None:
    try:
        await _apply_model(model, gpu_memory_mode)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        await _fail(model, f"Unexpected local model manager error: {error}")


async def queue_local_model_apply(
    *,
    model: str,
    hf_token: str | None = None,
    gpu_memory_mode: str = "safe_10gb",
) -> LocalModelStatus:
    global _apply_task

    cleaned_model = model.strip()
    if not cleaned_model:
        return await _set_status(
            state="failed",
            model="",
            served_model="",
            progress_message="Enter a model name first.",
            error_code="model_required",
            error_message="Enter a model name first.",
            logs_tail="",
        )

    if hf_token and hf_token.strip():
        try:
            save_hf_token(hf_token)
        except OSError as error:
            return await _set_status(
                state="failed",
                model=cleaned_model,
                served_model="",
                progress_message="DocPilot could not save the Hugging Face token locally.",
                error_code="hf_token_save_failed",
                error_message=f"DocPilot could not save the Hugging Face token locally: {error}",
                logs_tail="",
            )

    if _apply_task and not _apply_task.done():
        _apply_task.cancel()

    await _set_status(
        state="checking_docker",
        model=cleaned_model,
        served_model="",
        base_url=BASE_URL,
        container_name=CONTAINER_NAME,
        progress_message=f"Preparing local model {cleaned_model}...",
        error_code="",
        error_message="",
        logs_tail="",
    )
    _apply_task = asyncio.create_task(_safe_apply_model(cleaned_model, gpu_memory_mode))
    return get_local_model_status()


def get_local_model_status() -> LocalModelStatus:
    return _status_snapshot()


async def stop_local_model() -> LocalModelStatus:
    global _apply_task
    if _apply_task and not _apply_task.done():
        _apply_task.cancel()
    await _set_status(state="starting_container", progress_message="Stopping local vLLM container...")
    await _stop_managed_container()
    return await _set_status(
        state="idle",
        model="",
        served_model="",
        progress_message="Local model stopped.",
        error_code="",
        error_message="",
        logs_tail="",
    )


async def remove_local_hf_token() -> LocalModelStatus:
    delete_hf_token()
    return await _set_status(progress_message="Saved Hugging Face token removed.")
