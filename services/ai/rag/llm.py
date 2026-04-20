import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from typing import Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse, urlunparse

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

LlmProvider = Literal["groq", "openai-compatible"]

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class LlmConfig(TypedDict):
    provider: LlmProvider
    api_key: str
    model: str
    base_url: str


class LlmConnectionError(RuntimeError):
    pass


class VllmEnvironmentDiagnostic(TypedDict):
    ok: bool
    status: str
    issue_code: str
    title: str
    message: str
    recommendation: str
    platform: str
    python_version: str
    is_windows: bool
    is_wsl: bool
    vllm_installed: bool
    vllm_import_ok: bool
    native_python_vllm_ok: bool
    native_python_vllm_details: str
    server_reachable: bool
    server_status: str
    server_details: str
    served_models: list[str]
    active_served_model: str
    startup_error_kind: str
    hf_token_available: bool
    hf_token_source: str
    gated_model_likely: bool
    hf_token_recommendation: str
    gpu_memory_mode: str
    gpu_memory_utilization: str
    max_model_len: str
    gpu_memory_snapshot: str
    base_url: str
    model: str
    setup_command: str
    docker_command: str
    docker_hint: str
    recommended_action: str
    wsl_available: bool
    wsl_status: str
    wsl_details: str
    docker_available: bool
    docker_running: bool
    docker_status: str
    docker_details: str
    gpu_available: bool
    gpu_details: str
    details: str


def _server_label(config: LlmConfig) -> str:
    return "vLLM/OpenAI-compatible server" if config["provider"] == "openai-compatible" else "LLM provider"


def _vllm_start_command(config: LlmConfig) -> str:
    parsed = urlparse(config["base_url"])
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    bind_host = "127.0.0.1" if host in {"localhost", "127.0.0.1"} else host
    return f'vllm serve "{config["model"]}" --host {bind_host} --port {port}'


def _vllm_wsl_start_command(config: LlmConfig) -> str:
    parsed = urlparse(config["base_url"])
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f'vllm serve "{config["model"]}" --host 0.0.0.0 --port {port}'


def _vllm_docker_start_command(config: LlmConfig) -> str:
    parsed = urlparse(config["base_url"])
    host_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    gpu_memory_utilization, max_model_len = _vllm_memory_defaults(config["model"])
    return (
        "docker run --gpus all --rm "
        f"-p {host_port}:8000 --ipc=host --env HF_TOKEN "
        f'vllm/vllm-openai:latest "{config["model"]}" '
        f"--gpu-memory-utilization {gpu_memory_utilization} --max-model-len {max_model_len}"
    )


def _vllm_docker_powershell_command(config: LlmConfig) -> str:
    return '$env:HF_TOKEN="<your-hugging-face-read-token>"\n' + _vllm_docker_start_command(config)


def _is_docpilot_default_backend_url(config: LlmConfig) -> bool:
    parsed = urlparse(config["base_url"])
    return parsed.hostname in {"localhost", "127.0.0.1"} and parsed.port == 8000


def _normalize_provider(provider: str | None) -> LlmProvider:
    return "openai-compatible" if provider == "openai-compatible" else "groq"


def normalize_openai_base_url(base_url: str | None) -> str:
    raw = (base_url or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
    if not raw:
        raw = DEFAULT_OPENAI_BASE_URL.rstrip("/")

    parsed = urlparse(raw)
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"

    return urlunparse(parsed._replace(path=path))


def is_managed_local_vllm(llm_provider: str | None, openai_base_url: str | None) -> bool:
    if llm_provider != "openai-compatible":
        return False

    parsed = urlparse(normalize_openai_base_url(openai_base_url))
    return parsed.hostname in {"localhost", "127.0.0.1"} and parsed.port == 8001


def _detect_wsl() -> bool:
    if platform.system().lower() != "linux":
        return False

    release = platform.release().lower()
    version = platform.version().lower()
    if "microsoft" in release or "microsoft" in version or "wsl" in release:
        return True

    try:
        with open("/proc/version", encoding="utf-8") as version_file:
            return "microsoft" in version_file.read().lower()
    except OSError:
        return False


def _clean_probe_text(value: str) -> str:
    return value.replace("\x00", "").strip()


def _run_probe(command: list[str], timeout: int = 5) -> tuple[bool, str]:
    if shutil.which(command[0]) is None:
        return False, f"{command[0]} was not found on PATH."

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"{' '.join(command)} timed out."
    except OSError as error:
        return False, str(error)

    output = _clean_probe_text(f"{result.stdout}\n{result.stderr}")
    return result.returncode == 0, output


def _probe_wsl() -> tuple[bool, str, str]:
    if platform.system().lower() != "windows":
        return _detect_wsl(), "not_windows", "WSL probing is only needed on Windows hosts."

    ok, details = _run_probe(["wsl.exe", "-l", "-v"])
    if ok:
        return True, "ready", details

    lowered = details.lower()
    if "access is denied" in lowered or "e_accessdenied" in lowered:
        return False, "access_denied", details
    if "no installed distributions" in lowered:
        return False, "no_distros", details
    return False, "unavailable", details


def _probe_docker() -> tuple[bool, bool, str, str]:
    if shutil.which("docker") is None:
        return False, False, "missing", "docker was not found on PATH."

    ok, details = _run_probe(["docker", "ps"])
    if ok:
        return True, True, "running", details

    lowered = details.lower()
    if "docker_engine" in lowered or "daemon" in lowered or "cannot find the file specified" in lowered:
        return True, False, "daemon_not_running", details
    if "access is denied" in lowered:
        return True, False, "access_denied", details
    return True, False, "unavailable", details


def _probe_gpu() -> tuple[bool, str]:
    ok, details = _run_probe(["nvidia-smi"], timeout=8)
    return ok, details


def _openai_models_url(config: LlmConfig) -> str:
    parsed = urlparse(config["base_url"])
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunparse(parsed._replace(path=f"{path}/models"))


def parse_openai_models_response(payload: dict | str) -> list[str]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []

    if not isinstance(payload, dict):
        return []

    data = payload.get("data")
    if not isinstance(data, list):
        return []

    models: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            models.append(item["id"])
    return models


def _probe_openai_compatible_server(config: LlmConfig) -> tuple[bool, str, str, list[str], str]:
    models_url = _openai_models_url(config)
    request = Request(models_url, method="GET")

    try:
        with urlopen(request, timeout=3) as response:
            body = response.read().decode("utf-8")
            served_models = parse_openai_models_response(body) if body else []
            active_served_model = served_models[0] if served_models else ""
            return (
                True,
                "reachable",
                f"GET {models_url} returned HTTP {response.status}.",
                served_models,
                active_served_model,
            )
    except HTTPError as error:
        if error.code in {401, 403}:
            return True, "auth_required", f"GET {models_url} returned HTTP {error.code}.", [], ""
        return False, "http_error", f"GET {models_url} returned HTTP {error.code}.", [], ""
    except URLError as error:
        reason = getattr(error, "reason", error)
        return False, "not_running", f"Could not reach {models_url}: {reason}", [], ""
    except TimeoutError:
        return False, "timeout", f"Timed out while calling {models_url}.", [], ""
    except OSError as error:
        return False, "not_running", f"Could not reach {models_url}: {error}", [], ""


def classify_vllm_startup_error(message: str) -> tuple[str, str, str]:
    lower = message.lower()
    gated_fragments = [
        "gatedrepoerror",
        "cannot access gated repo",
        "you must have access to it and be authenticated",
        "you are trying to access a gated repo",
    ]

    if any(fragment in lower for fragment in gated_fragments) or (
        "401 unauthorized" in lower and "huggingface" in lower and "restricted" in lower
    ):
        return (
            "gated_huggingface_model",
            "Docker started, but Hugging Face rejected model access",
            "Your current PowerShell session does not expose HF_TOKEN to Docker, or the token account has not accepted Gemma access. Rotate the pasted token, create a new read token, set HF_TOKEN, accept model access, then restart vLLM Docker.",
        )

    gpu_memory_fragments = [
        "free memory on device cuda",
        "less than desired gpu memory utilization",
        "decrease gpu memory utilization",
    ]
    if any(fragment in lower for fragment in gpu_memory_fragments):
        return (
            "gpu_memory_insufficient",
            "vLLM could not reserve enough GPU memory",
            "vLLM authenticated successfully but could not reserve enough GPU memory. Use Safe 10GB mode or close GPU-heavy apps, then restart Docker.",
        )

    return (
        "unknown",
        "vLLM startup failed",
        "Check the vLLM Docker logs for the exact startup error.",
    )


def _base_vllm_diagnostic(config: LlmConfig) -> VllmEnvironmentDiagnostic:
    system = platform.system()
    is_windows = system.lower() == "windows"
    is_wsl = _detect_wsl()
    wsl_available, wsl_status, wsl_details = _probe_wsl()
    docker_available, docker_running, docker_status, docker_details = _probe_docker()
    gpu_available, gpu_details = _probe_gpu()
    (
        server_reachable,
        server_status,
        server_details,
        served_models,
        active_served_model,
    ) = _probe_openai_compatible_server(config)
    hf_token_available = bool(os.getenv("HF_TOKEN"))
    gated_model_likely = _is_likely_gated_huggingface_model(config["model"])
    gpu_memory_utilization, max_model_len = _vllm_memory_defaults(config["model"])
    return {
        "ok": True,
        "status": "ready",
        "issue_code": "none",
        "title": "Local vLLM server is reachable",
        "message": f"DocPilot can reach the OpenAI-compatible server at {config['base_url']}.",
        "recommendation": "Use Test Connection to confirm the selected model responds.",
        "platform": system or "unknown",
        "python_version": platform.python_version(),
        "is_windows": is_windows,
        "is_wsl": is_wsl,
        "vllm_installed": True,
        "vllm_import_ok": True,
        "native_python_vllm_ok": False,
        "native_python_vllm_details": "Not checked. Docker/WSL serving does not require native Windows Python vLLM.",
        "server_reachable": server_reachable,
        "server_status": server_status,
        "server_details": server_details,
        "served_models": served_models,
        "active_served_model": active_served_model,
        "startup_error_kind": "none",
        "hf_token_available": hf_token_available,
        "hf_token_source": "environment" if hf_token_available else "missing",
        "gated_model_likely": gated_model_likely,
        "hf_token_recommendation": _hf_token_recommendation(config["model"], hf_token_available),
        "gpu_memory_mode": _vllm_memory_mode(config["model"]),
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "gpu_memory_snapshot": _gpu_memory_snapshot(gpu_details),
        "base_url": config["base_url"],
        "model": config["model"],
        "setup_command": _vllm_start_command(config),
        "docker_command": _vllm_docker_start_command(config),
        "docker_hint": f"If you use Docker Desktop, start it first, then run:\n{_vllm_docker_powershell_command(config)}",
        "recommended_action": "",
        "wsl_available": wsl_available,
        "wsl_status": wsl_status,
        "wsl_details": wsl_details,
        "docker_available": docker_available,
        "docker_running": docker_running,
        "docker_status": docker_status,
        "docker_details": docker_details,
        "gpu_available": gpu_available,
        "gpu_details": gpu_details,
        "details": "",
    }


def _recommended_local_runtime(diagnostic: VllmEnvironmentDiagnostic, config: LlmConfig) -> str:
    docker_command = _vllm_docker_powershell_command(config)
    wsl_command = _vllm_wsl_start_command(config)

    if diagnostic["docker_available"] and diagnostic["docker_running"]:
        return f"Use Docker now with: {docker_command}"
    if diagnostic["docker_available"] and diagnostic["docker_status"] == "daemon_not_running":
        return f"Start Docker Desktop, then run: {docker_command}"
    if diagnostic["wsl_available"]:
        return f"Use WSL2 Ubuntu with: {wsl_command}"
    if diagnostic["wsl_status"] == "access_denied":
        return (
            "WSL is installed but Windows denied access to it. Fix WSL permissions or run Docker Desktop, "
            f"then serve the model with: {docker_command}"
        )
    return (
        "Install/enable WSL2 Ubuntu or start Docker Desktop. "
        f"Once Docker is running, use: {docker_command}"
    )


def _is_likely_gated_huggingface_model(model: str) -> bool:
    return model.startswith("google/gemma")


def _hf_token_recommendation(model: str, hf_token_available: bool) -> str:
    if not _is_likely_gated_huggingface_model(model):
        return "This model is not marked as likely gated by DocPilot. HF_TOKEN may not be required."
    if hf_token_available:
        return "HF_TOKEN is set in the DocPilot backend environment. If Hugging Face still returns 401, confirm the token account accepted access to this model."
    return "HF_TOKEN is missing in the current environment. Set a new Hugging Face read token before starting vLLM Docker."


def _vllm_memory_mode(model: str) -> str:
    if model == "google/gemma-2-2b-it":
        return "safe_10gb"
    return "balanced"


def _vllm_memory_defaults(model: str) -> tuple[str, str]:
    if _vllm_memory_mode(model) == "safe_10gb":
        return "0.82", "4096"
    return "0.85", "4096"


def _gpu_memory_snapshot(gpu_details: str) -> str:
    for line in gpu_details.splitlines():
        if "MiB /" in line:
            return " ".join(line.split())
    return ""


def classify_vllm_server_status(config: LlmConfig) -> VllmEnvironmentDiagnostic:
    diagnostic = _base_vllm_diagnostic(config)
    runtime_recommendation = _recommended_local_runtime(diagnostic, config)
    diagnostic["recommended_action"] = runtime_recommendation

    if diagnostic["server_reachable"]:
        diagnostic.update(
            {
                "ok": True,
                "status": "ready",
                "issue_code": "server_reachable",
                "title": "Local vLLM server is reachable",
                "message": f"DocPilot can reach {config['base_url']}.",
                "recommendation": "Use Test Connection to verify the selected model name and response path.",
                "details": "",
            }
        )
        return diagnostic

    diagnostic.update(
        {
            "ok": False,
            "status": "server_not_running",
            "issue_code": "vllm_server_not_running",
            "title": "Local vLLM server is not running",
            "message": f"Local vLLM server is not running at {config['base_url']}.",
            "recommendation": runtime_recommendation,
            "details": diagnostic["server_details"],
            "vllm_installed": False,
            "vllm_import_ok": False,
            "native_python_vllm_ok": False,
            "native_python_vllm_details": "Not checked. Docker/WSL serving does not require native Windows Python vLLM.",
        }
    )
    return diagnostic


def classify_vllm_startup_failure(config: LlmConfig, startup_log: str) -> VllmEnvironmentDiagnostic:
    diagnostic = _base_vllm_diagnostic(config)
    issue_code, title, recommendation = classify_vllm_startup_error(startup_log)
    diagnostic.update(
        {
            "ok": False,
            "status": "startup_failed",
            "issue_code": issue_code,
            "title": title,
            "message": "vLLM Docker started, but the selected Hugging Face model could not be loaded.",
            "recommendation": recommendation,
            "startup_error_kind": issue_code,
            "details": _redact_tokens(startup_log),
            "recommended_action": _vllm_docker_powershell_command(config),
        }
    )
    return diagnostic


def _redact_tokens(value: str) -> str:
    redacted_words: list[str] = []
    for word in value.split():
        if word.startswith(("hf_", "hf-")) and len(word) > 12:
            redacted_words.append(f"{word[:4]}...redacted")
        else:
            redacted_words.append(word)
    return " ".join(redacted_words)


def classify_vllm_environment(
    *,
    config: LlmConfig,
    system: str | None = None,
    python_version: str | None = None,
    is_wsl: bool | None = None,
    vllm_installed: bool = True,
    import_error: str = "",
) -> VllmEnvironmentDiagnostic:
    diagnostic = _base_vllm_diagnostic(config)
    system_name = system if system is not None else diagnostic["platform"]
    version = python_version if python_version is not None else diagnostic["python_version"]
    running_in_wsl = is_wsl if is_wsl is not None else diagnostic["is_wsl"]
    is_windows = system_name.lower() == "windows"
    lower_error = import_error.lower()

    diagnostic.update(
        {
            "platform": system_name or "unknown",
            "python_version": version,
            "is_windows": is_windows,
            "is_wsl": running_in_wsl,
            "vllm_installed": vllm_installed,
            "vllm_import_ok": vllm_installed and not import_error,
            "native_python_vllm_ok": vllm_installed and not import_error,
            "native_python_vllm_details": import_error,
            "details": import_error,
        }
    )

    python_major_minor = tuple(int(part) for part in version.split(".")[:2] if part.isdigit())
    runtime_recommendation = _recommended_local_runtime(diagnostic, config)

    if not vllm_installed:
        diagnostic.update(
            {
                "ok": False,
                "status": "missing",
                "issue_code": "vllm_not_installed",
                "title": "vLLM is not installed in this Python environment",
                "message": "DocPilot could not find the vLLM package here. This is okay if you run vLLM from WSL, Docker, or another environment.",
                "recommendation": (
                    "Install and run vLLM in a supported environment, then point DocPilot to its /v1 endpoint. "
                    f"{runtime_recommendation}"
                ),
            }
        )
        return diagnostic

    if "vllm._c" in lower_error or "no module named 'vllm._c'" in lower_error:
        diagnostic.update(
            {
                "ok": False,
                "status": "broken",
                "issue_code": "vllm_compiled_extension_missing",
                "title": "vLLM is installed, but its compiled extension is missing",
                "message": "This usually happens when vLLM is installed in an unsupported local runtime, such as native Windows or an incompatible Python version.",
                "recommendation": (
                    "Run vLLM from WSL2 Ubuntu or Docker instead of native Windows. "
                    f"{runtime_recommendation}"
                ),
            }
        )
        return diagnostic

    if is_windows:
        diagnostic.update(
            {
                "ok": False,
                "status": "unsupported",
                "issue_code": "native_windows_runtime",
                "title": "Native Windows is not the recommended vLLM runtime",
                "message": "vLLM commonly requires Linux-compatible compiled components. On this machine, use WSL2 Ubuntu or Docker for local serving.",
                "recommendation": runtime_recommendation,
            }
        )
        return diagnostic

    if python_major_minor and python_major_minor >= (3, 13):
        diagnostic.update(
            {
                "ok": False,
                "status": "unsupported",
                "issue_code": "python_version_unsupported",
                "title": "Python 3.13 may not be supported by your vLLM install",
                "message": "Your vLLM runtime is using Python 3.13. Many vLLM builds are published for earlier Python versions.",
                "recommendation": f"Create a WSL2 or Docker environment with Python 3.11 or 3.12, then start the server. {runtime_recommendation}",
            }
        )
        return diagnostic

    if any(fragment in lower_error for fragment in ["cuda", "gpu", "torch", "triton"]):
        diagnostic.update(
            {
                "ok": False,
                "status": "broken",
                "issue_code": "gpu_runtime_error",
                "title": "vLLM found a GPU/runtime dependency problem",
                "message": "vLLM imported far enough to hit a CUDA, PyTorch, or Triton runtime issue.",
                "recommendation": "Check the vLLM terminal logs, GPU driver, CUDA/PyTorch compatibility, or try a smaller model such as google/gemma-2-2b-it.",
            }
        )
        return diagnostic

    if import_error:
        diagnostic.update(
            {
                "ok": False,
                "status": "broken",
                "issue_code": "vllm_import_error",
                "title": "vLLM could not be imported cleanly",
                "message": "DocPilot found vLLM, but importing its runtime failed.",
                "recommendation": (
                    "Use WSL2 or Docker for vLLM if this is a local Windows setup, or inspect the error details below. "
                    f"{runtime_recommendation}"
                ),
            }
        )
        return diagnostic

    return diagnostic


def diagnose_vllm_environment(
    *,
    llm_model: str | None = None,
    openai_base_url: str | None = None,
    check_native_python: bool = False,
) -> VllmEnvironmentDiagnostic:
    config = resolve_llm_config(
        llm_provider="openai-compatible",
        llm_model=llm_model,
        openai_base_url=openai_base_url,
    )

    server_diagnostic = classify_vllm_server_status(config)
    if server_diagnostic["server_reachable"] or not check_native_python:
        return server_diagnostic

    if importlib.util.find_spec("vllm") is None:
        return classify_vllm_environment(config=config, vllm_installed=False)

    try:
        importlib.import_module("vllm._C")
    except Exception as error:
        return classify_vllm_environment(
            config=config,
            vllm_installed=True,
            import_error=str(error),
        )

    return classify_vllm_environment(config=config)


def resolve_llm_config(
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> LlmConfig:
    provider = _normalize_provider(llm_provider or DEFAULT_PROVIDER)
    model = (llm_model or DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL

    if provider == "openai-compatible":
        return {
            "provider": provider,
            "api_key": (openai_api_key or DEFAULT_OPENAI_API_KEY or "not-needed").strip(),
            "model": model,
            "base_url": normalize_openai_base_url(openai_base_url),
        }

    return {
        "provider": "groq",
        "api_key": (groq_api_key or DEFAULT_GROQ_API_KEY).strip(),
        "model": model,
        "base_url": "",
    }


def create_llm(
    *,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
    temperature: float,
    max_tokens: int,
):
    config = resolve_llm_config(
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        llm_provider=llm_provider,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )

    if config["provider"] == "openai-compatible":
        kwargs = {
            "api_key": config["api_key"],
            "base_url": config["base_url"],
            "model": config["model"],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if is_managed_local_vllm(config["provider"], config["base_url"]):
            kwargs.update({"timeout": 30, "max_retries": 0})
        return ChatOpenAI(**kwargs)

    return ChatGroq(
        api_key=config["api_key"],
        model=config["model"],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def format_llm_error(error: Exception, config: LlmConfig) -> str:
    message = str(error).strip() or repr(error) or error.__class__.__name__
    lower = message.lower()

    if config["provider"] != "openai-compatible":
        return message

    startup_kind, startup_title, startup_recommendation = classify_vllm_startup_error(message)
    if startup_kind != "unknown":
        return f"{startup_title}. {startup_recommendation}"

    start_command = _vllm_start_command(config)
    port_note = (
        " Hugging Face's standalone vLLM examples often use port 8000, but DocPilot's AI backend uses port 8000 by default; use another port such as 8001 unless you moved DocPilot."
        if _is_docpilot_default_backend_url(config)
        else ""
    )

    if any(fragment in lower for fragment in ["connection error", "connection refused", "failed to establish", "could not connect"]):
        return (
            f"Could not reach {_server_label(config)} at {config['base_url']}.{port_note} "
            f"Start the configured server with: {start_command}"
        )

    if any(fragment in lower for fragment in ["timed out", "timeout", "readtimeout"]):
        if is_managed_local_vllm(config["provider"], config["base_url"]):
            return (
                f"Local model did not produce tokens fast enough at {config['base_url']} while using "
                f"'{config['model']}'. Try facebook/opt-125m for a smoke test, or keep Gemma 2B with shorter context and close GPU-heavy apps."
            )
        return (
            f"The {_server_label(config)} at {config['base_url']} did not respond in time. "
            f"Check whether the model is still loading, the machine has enough VRAM/RAM, or try a smaller model. Configured model: {config['model']}."
        )

    if any(fragment in lower for fragment in ["401", "unauthorized", "invalid api key", "authentication"]):
        return (
            f"The {_server_label(config)} rejected the API key. "
            "If this is local vLLM without --api-key, leave the API key blank in Settings. If you started vLLM with --api-key, enter the same key."
        )

    if any(fragment in lower for fragment in ["404", "not found", "model_not_found", "does not exist", "not served"]):
        try:
            _, _, _, served_models, active_served_model = _probe_openai_compatible_server(config)
        except Exception:
            served_models = []
            active_served_model = ""

        if active_served_model and config["model"] not in served_models:
            return (
                f"vLLM is reachable, but DocPilot is asking for {config['model']} while the server is serving "
                f"{active_served_model}. Update Settings to use the running model."
            )

        return (
            f"The {_server_label(config)} is reachable, but model '{config['model']}' was not accepted. "
            "Set DocPilot's Model name to the exact served model name, or use Settings to sync from the running server."
        )

    if any(fragment in lower for fragment in ["500", "502", "503", "504", "internal server error", "bad gateway", "service unavailable"]):
        return (
            f"The {_server_label(config)} returned a server error while using model '{config['model']}'. "
            "Check the vLLM terminal logs for model load failures, out-of-memory errors, or unsupported model features."
        )

    return message


async def test_llm_connection(
    *,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> dict[str, str | bool]:
    config = resolve_llm_config(
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        llm_provider=llm_provider,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )
    llm = create_llm(
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        llm_provider=llm_provider,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        temperature=0,
        max_tokens=16,
    )

    try:
        await llm.ainvoke("Reply with exactly: ok")
        return {
            "ok": True,
            "provider": config["provider"],
            "model": config["model"],
            "base_url": config["base_url"],
            "message": "Connection test succeeded.",
        }
    except Exception as error:
        raise LlmConnectionError(format_llm_error(error, config)) from error


def create_groq_llm(
    *,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    temperature: float,
    max_tokens: int,
):
    return create_llm(
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        llm_provider="groq",
        temperature=temperature,
        max_tokens=max_tokens,
    )
