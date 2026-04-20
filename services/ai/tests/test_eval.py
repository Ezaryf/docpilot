import asyncio
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

from rag.citations import extract_citations
from rag.eval import serialize_evaluation_result
import rag.generate as generate_module
import rag.graph as graph_module
import rag.local_llm as local_llm_module
import rag.llm as llm_module
from rag.local_llm import build_vllm_docker_args, delete_hf_token, hf_token_available, load_hf_token, save_hf_token
from rag.llm import (
    classify_vllm_environment,
    classify_vllm_startup_error,
    diagnose_vllm_environment,
    format_llm_error,
    is_managed_local_vllm,
    normalize_openai_base_url,
    parse_openai_models_response,
    resolve_llm_config,
)


class FakeEvaluationResult:
    scores = [
        {
            "faithfulness": 0.75,
            "answer_relevancy": float("nan"),
        },
        {
            "faithfulness": None,
            "answer_relevancy": 0.5,
        },
    ]


def test_serialize_evaluation_result_normalizes_ragas_scores():
    assert serialize_evaluation_result(FakeEvaluationResult()) == {
        "faithfulness": [0.75, 0.0],
        "answer_relevancy": [0.0, 0.5],
    }


def test_resolve_llm_config_supports_groq():
    assert resolve_llm_config(
        groq_api_key="gsk_test",
        llm_model="llama-3.1-8b-instant",
        llm_provider="groq",
    ) == {
        "provider": "groq",
        "api_key": "gsk_test",
        "model": "llama-3.1-8b-instant",
        "base_url": "",
    }


def test_resolve_llm_config_supports_openai_compatible():
    assert resolve_llm_config(
        llm_model="google/gemma-test",
        llm_provider="openai-compatible",
        openai_base_url="http://localhost:8001/v1",
        openai_api_key="local-key",
    ) == {
        "provider": "openai-compatible",
        "api_key": "local-key",
        "model": "google/gemma-test",
        "base_url": "http://localhost:8001/v1",
    }


def test_normalize_openai_base_url_adds_v1():
    assert normalize_openai_base_url("http://localhost:8001") == "http://localhost:8001/v1"
    assert normalize_openai_base_url("http://localhost:8001/") == "http://localhost:8001/v1"
    assert normalize_openai_base_url("http://localhost:8001/v1") == "http://localhost:8001/v1"


def test_is_managed_local_vllm_detects_docpilot_local_endpoint():
    assert is_managed_local_vllm("openai-compatible", "http://localhost:8001/v1") is True
    assert is_managed_local_vllm("openai-compatible", "http://127.0.0.1:8001") is True
    assert is_managed_local_vllm("openai-compatible", "http://localhost:9000/v1") is False
    assert is_managed_local_vllm("groq", "http://localhost:8001/v1") is False


def test_parse_openai_models_response_extracts_served_models():
    assert parse_openai_models_response(
        {
            "object": "list",
            "data": [
                {"id": "google/gemma-2-2b-it", "object": "model"},
                {"id": "local-chat", "object": "model"},
                {"object": "model"},
            ],
        }
    ) == ["google/gemma-2-2b-it", "local-chat"]


def test_format_llm_error_uses_configured_vllm_port():
    config = resolve_llm_config(
        llm_model="local-chat",
        llm_provider="openai-compatible",
        openai_base_url="http://localhost:9000/v1",
    )

    message = format_llm_error(Exception("Connection error."), config)

    assert "http://localhost:9000/v1" in message
    assert 'vllm serve "local-chat" --host 127.0.0.1 --port 9000' in message


def test_format_llm_error_warns_for_docpilot_backend_port():
    config = resolve_llm_config(
        llm_model="local-chat",
        llm_provider="openai-compatible",
        openai_base_url="http://localhost:8000/v1",
    )

    message = format_llm_error(Exception("Connection error."), config)

    assert "DocPilot's AI backend uses port 8000" in message


def test_format_llm_error_handles_auth_and_model_errors(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "_probe_openai_compatible_server",
        lambda config: (True, "reachable", "GET /models returned HTTP 200.", [], ""),
    )
    config = resolve_llm_config(
        llm_model="local-chat",
        llm_provider="openai-compatible",
        openai_base_url="http://localhost:8001/v1",
    )

    auth_message = format_llm_error(Exception("401 Unauthorized"), config)
    model_message = format_llm_error(Exception("model_not_found"), config)

    assert "rejected the API key" in auth_message
    assert "model 'local-chat' was not accepted" in model_message


def test_format_llm_error_reports_running_vllm_model_mismatch(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "_probe_openai_compatible_server",
        lambda config: (
            True,
            "reachable",
            "GET /models returned HTTP 200.",
            ["google/gemma-2-2b-it"],
            "google/gemma-2-2b-it",
        ),
    )
    config = resolve_llm_config(
        llm_model="google/gemma-4-31B-it",
        llm_provider="openai-compatible",
        openai_base_url="http://localhost:8001/v1",
    )

    message = format_llm_error(Exception("model_not_found"), config)

    assert "asking for google/gemma-4-31B-it" in message
    assert "server is serving google/gemma-2-2b-it" in message


def test_format_llm_error_handles_empty_local_timeout():
    config = resolve_llm_config(
        llm_model="google/gemma-2-2b-it",
        llm_provider="openai-compatible",
        openai_base_url="http://localhost:8001/v1",
    )

    message = format_llm_error(TimeoutError(), config)

    assert "Local model did not produce tokens fast enough" in message
    assert "facebook/opt-125m" in message


def test_classify_vllm_environment_detects_missing_compiled_extension():
    config = resolve_llm_config(
        llm_model="google/gemma-2-2b-it",
        llm_provider="openai-compatible",
        openai_base_url="http://localhost:8001/v1",
    )

    result = classify_vllm_environment(
        config=config,
        system="Windows",
        python_version="3.13.0",
        is_wsl=False,
        vllm_installed=True,
        import_error="No module named 'vllm._C'",
    )

    assert result["ok"] is False
    assert result["issue_code"] == "vllm_compiled_extension_missing"
    assert "WSL2" in result["recommendation"]
    assert 'google/gemma-2-2b-it' in result["recommendation"]


def test_classify_vllm_environment_detects_native_windows():
    config = resolve_llm_config(
        llm_model="google/gemma-2-2b-it",
        llm_provider="openai-compatible",
        openai_base_url="http://localhost:8001/v1",
    )

    result = classify_vllm_environment(
        config=config,
        system="Windows",
        python_version="3.12.0",
        is_wsl=False,
    )

    assert result["ok"] is False
    assert result["issue_code"] == "native_windows_runtime"
    assert "Docker" in result["recommendation"] or "WSL2" in result["recommendation"]


def test_classify_vllm_environment_detects_missing_package():
    config = resolve_llm_config(
        llm_model="google/gemma-2-2b-it",
        llm_provider="openai-compatible",
        openai_base_url="http://localhost:8001/v1",
    )

    result = classify_vllm_environment(
        config=config,
        system="Linux",
        python_version="3.11.0",
        is_wsl=True,
        vllm_installed=False,
    )

    assert result["ok"] is False
    assert result["issue_code"] == "vllm_not_installed"


def test_classify_vllm_environment_detects_generic_import_error():
    config = resolve_llm_config(
        llm_model="google/gemma-2-2b-it",
        llm_provider="openai-compatible",
        openai_base_url="http://localhost:8001/v1",
    )

    result = classify_vllm_environment(
        config=config,
        system="Linux",
        python_version="3.11.0",
        is_wsl=False,
        vllm_installed=True,
        import_error="unexpected import failure",
    )

    assert result["ok"] is False
    assert result["issue_code"] == "vllm_import_error"


def test_diagnose_vllm_environment_prioritizes_server_status(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "_probe_openai_compatible_server",
        lambda config: (False, "not_running", "Could not reach http://localhost:8001/v1/models", [], ""),
    )
    monkeypatch.setattr(
        llm_module,
        "_probe_docker",
        lambda: (True, False, "daemon_not_running", "Docker daemon is not running."),
    )
    monkeypatch.setattr(
        llm_module,
        "_probe_wsl",
        lambda: (False, "access_denied", "Access is denied."),
    )
    monkeypatch.setattr(llm_module, "_probe_gpu", lambda: (True, "NVIDIA GPU detected."))

    result = diagnose_vllm_environment(
        llm_model="google/gemma-2-2b-it",
        openai_base_url="http://localhost:8001/v1",
    )

    assert result["ok"] is False
    assert result["issue_code"] == "vllm_server_not_running"
    assert result["server_reachable"] is False
    assert result["docker_running"] is False
    assert "Start Docker Desktop" in result["recommendation"]
    assert "vllm._C" not in result["details"]


def test_diagnose_vllm_environment_reports_native_vllm_only_when_requested(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "_probe_openai_compatible_server",
        lambda config: (False, "not_running", "Could not reach http://localhost:8001/v1/models", [], ""),
    )
    monkeypatch.setattr(llm_module, "_probe_docker", lambda: (True, True, "running", "Docker running."))
    monkeypatch.setattr(llm_module, "_probe_wsl", lambda: (True, "ready", "Ubuntu running."))
    monkeypatch.setattr(llm_module, "_probe_gpu", lambda: (True, "NVIDIA GPU detected."))
    monkeypatch.setattr(llm_module.importlib.util, "find_spec", lambda name: object())

    def fail_native_import(name):
        raise ModuleNotFoundError("No module named 'vllm._C'")

    monkeypatch.setattr(llm_module.importlib, "import_module", fail_native_import)

    result = diagnose_vllm_environment(
        llm_model="google/gemma-2-2b-it",
        openai_base_url="http://localhost:8001/v1",
        check_native_python=True,
    )

    assert result["ok"] is False
    assert result["issue_code"] == "vllm_compiled_extension_missing"
    assert "vllm._C" in result["details"]


def test_vllm_docker_command_uses_hf_token_and_positional_model(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "_probe_openai_compatible_server",
        lambda config: (False, "not_running", "Could not reach http://localhost:8001/v1/models", [], ""),
    )
    monkeypatch.setattr(llm_module, "_probe_docker", lambda: (True, True, "running", "Docker running."))
    monkeypatch.setattr(llm_module, "_probe_wsl", lambda: (True, "ready", "Ubuntu running."))
    monkeypatch.setattr(llm_module, "_probe_gpu", lambda: (True, "NVIDIA GPU detected."))

    result = diagnose_vllm_environment(
        llm_model="google/gemma-2-2b-it",
        openai_base_url="http://localhost:8001/v1",
    )

    assert '$env:HF_TOKEN="<your-hugging-face-read-token>"' in result["docker_hint"]
    assert "--env HF_TOKEN " in result["docker_command"]
    assert '--model "google/gemma-2-2b-it"' not in result["docker_command"]
    assert 'vllm/vllm-openai:latest "google/gemma-2-2b-it"' in result["docker_command"]
    assert "--gpu-memory-utilization 0.82 --max-model-len 4096" in result["docker_command"]


def test_classify_vllm_startup_error_detects_gated_huggingface_model():
    kind, title, recommendation = classify_vllm_startup_error(
        "GatedRepoError: 401 Unauthorized. Cannot access gated repo. You must have access to it and be authenticated."
    )

    assert kind == "gated_huggingface_model"
    assert "rejected model access" in title
    assert "HF_TOKEN" in recommendation


def test_classify_vllm_startup_error_detects_gpu_memory_insufficient():
    kind, title, recommendation = classify_vllm_startup_error(
        "ValueError: Free memory on device cuda:0 (8.86/10.0 GiB) on startup is less than desired GPU memory utilization (0.9, 9.0 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes."
    )

    assert kind == "gpu_memory_insufficient"
    assert "could not reserve enough GPU memory" in title
    assert "Safe 10GB mode" in recommendation


def test_diagnose_vllm_environment_reports_missing_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        llm_module,
        "_probe_openai_compatible_server",
        lambda config: (False, "not_running", "Could not reach http://localhost:8001/v1/models", [], ""),
    )
    monkeypatch.setattr(llm_module, "_probe_docker", lambda: (True, True, "running", "Docker running."))
    monkeypatch.setattr(llm_module, "_probe_wsl", lambda: (True, "ready", "Ubuntu running."))
    monkeypatch.setattr(llm_module, "_probe_gpu", lambda: (True, "NVIDIA GPU detected."))

    result = diagnose_vllm_environment(
        llm_model="google/gemma-2-2b-it",
        openai_base_url="http://localhost:8001/v1",
    )

    assert result["hf_token_available"] is False
    assert result["hf_token_source"] == "missing"
    assert result["gated_model_likely"] is True
    assert "HF_TOKEN is missing" in result["hf_token_recommendation"]


def test_diagnose_vllm_environment_marks_non_gated_smoke_test(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        llm_module,
        "_probe_openai_compatible_server",
        lambda config: (False, "not_running", "Could not reach http://localhost:8001/v1/models", [], ""),
    )
    monkeypatch.setattr(llm_module, "_probe_docker", lambda: (True, True, "running", "Docker running."))
    monkeypatch.setattr(llm_module, "_probe_wsl", lambda: (True, "ready", "Ubuntu running."))
    monkeypatch.setattr(llm_module, "_probe_gpu", lambda: (True, "NVIDIA GPU detected."))

    result = diagnose_vllm_environment(
        llm_model="facebook/opt-125m",
        openai_base_url="http://localhost:8001/v1",
    )

    assert result["gated_model_likely"] is False
    assert "may not be required" in result["hf_token_recommendation"]


def test_local_model_docker_args_are_detached_safe_and_token_free():
    args = build_vllm_docker_args("google/gemma-2-2b-it", "safe_10gb")

    assert args[:3] == ["docker", "run", "-d"]
    assert "--name" in args
    assert args[args.index("--name") + 1] == "docpilot-vllm"
    assert "--rm" in args
    assert "-p" in args
    assert args[args.index("-p") + 1] == "8001:8000"
    assert "--env" in args
    assert args[args.index("--env") + 1] == "HF_TOKEN"
    assert "google/gemma-2-2b-it" in args
    assert "--gpu-memory-utilization" in args
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.82"
    assert "--max-model-len" in args
    assert args[args.index("--max-model-len") + 1] == "4096"
    assert not any(arg.startswith(("hf_", "hf-")) for arg in args)


def test_local_model_token_storage_is_local_and_removable(monkeypatch):
    test_home = Path.cwd() / ".tmp-local-llm-token-test"
    shutil.rmtree(test_home, ignore_errors=True)
    monkeypatch.setenv("DOCPILOT_HOME", str(test_home))
    monkeypatch.delenv("HF_TOKEN", raising=False)

    try:
        save_hf_token("hf_test_read_token")

        assert hf_token_available() is True
        assert load_hf_token() == "hf_test_read_token"

        delete_hf_token()

        assert hf_token_available() is False
        assert load_hf_token() == ""
    finally:
        shutil.rmtree(test_home, ignore_errors=True)


def test_local_model_error_classifier_maps_common_failures():
    gated_code, gated_message = local_llm_module._friendly_error_code_and_message(
        "GatedRepoError: 401 Unauthorized. Cannot access gated repo. You must have access to it and be authenticated."
    )
    gpu_code, gpu_message = local_llm_module._friendly_error_code_and_message(
        "ValueError: Free memory on device cuda:0 is less than desired GPU memory utilization. Decrease GPU memory utilization."
    )
    docker_code, docker_message = local_llm_module._friendly_error_code_and_message(
        "permission denied while trying to connect to the Docker daemon"
    )

    assert gated_code == "gated_model_auth_required"
    assert "read token" in gated_message
    assert gpu_code == "gpu_memory_insufficient"
    assert "Safe 10GB mode" in gpu_message
    assert docker_code == "docker_daemon_not_running"
    assert "Docker Desktop" in docker_message


def test_local_model_subprocess_timeout_becomes_completed_process(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["docker", "ps"], timeout=10)

    monkeypatch.setattr(local_llm_module.subprocess, "run", fake_run)

    result = local_llm_module._run_sync(["docker", "ps"], timeout=10)
    code, message = local_llm_module._friendly_error_code_and_message(result.stderr)

    assert result.returncode == 124
    assert "timed out" in result.stderr
    assert code == "docker_timeout"
    assert "Docker did not respond" in message


def test_local_model_safe_apply_marks_failed_on_unexpected_error(monkeypatch):
    async def fake_apply_model(model, gpu_memory_mode):
        raise RuntimeError("boom")

    async def fake_logs_tail():
        return ""

    monkeypatch.setattr(local_llm_module, "_apply_model", fake_apply_model)
    monkeypatch.setattr(local_llm_module, "_logs_tail", fake_logs_tail)

    asyncio.run(local_llm_module._safe_apply_model("google/gemma-2-2b-it", "safe_10gb"))
    status = local_llm_module.get_local_model_status()

    assert status["state"] == "failed"
    assert status["model"] == "google/gemma-2-2b-it"
    assert status["error_message"]


def test_local_rag_fast_mode_skips_llm_grading_and_rewrite(monkeypatch):
    search_top_k = []
    rerank_top_k = []

    async def fake_search(query, *, top_k, document_names=None):
        search_top_k.append(top_k)
        return [
            {"text": "Ezaryf works on AI systems.", "document_name": "resume.pdf"},
            {"text": "He manages data pipelines.", "document_name": "resume.pdf"},
            {"text": "Extra chunk.", "document_name": "resume.pdf"},
        ]

    async def fake_rerank(query, documents, *, top_k):
        rerank_top_k.append(top_k)
        return documents[:top_k]

    async def forbidden_grade(*args, **kwargs):
        raise AssertionError("local fast mode should not call LLM grading")

    async def forbidden_rewrite(*args, **kwargs):
        raise AssertionError("local fast mode should not rewrite")

    async def fake_generate(*args, **kwargs):
        yield "fast answer"

    monkeypatch.setattr(graph_module, "HYBRID_USE", False)
    monkeypatch.setattr(graph_module, "search_documents", fake_search)
    monkeypatch.setattr(graph_module, "rerank_documents", fake_rerank)
    monkeypatch.setattr(graph_module, "grade_documents", forbidden_grade)
    monkeypatch.setattr(graph_module, "rewrite_query", forbidden_rewrite)
    monkeypatch.setattr(graph_module, "generate_answer", fake_generate)

    async def collect_events():
        events = []
        async for event in graph_module.run_rag_pipeline(
            query="Summarize my documents",
            session_id="test",
            has_documents=True,
            llm_provider="openai-compatible",
            openai_base_url="http://localhost:8001/v1",
        ):
            events.append(event)
        return events

    events = asyncio.run(collect_events())
    trace_events = [event for event in events if event["type"] == "trace"]
    status_messages = [event["message"] for event in events if event["type"] == "status"]

    assert search_top_k == [3]
    assert rerank_top_k == [2]
    assert any(event == {"type": "token", "content": "fast answer"} for event in events)
    assert any("Searching documents" in message for message in status_messages)
    assert "Skipped for fast local mode" in str(trace_events)


def test_local_context_builder_truncates_documents():
    docs = [
        {"text": "A" * 2000, "document_name": "one.pdf"},
        {"text": "B" * 2000, "document_name": "two.pdf"},
        {"text": "C" * 2000, "document_name": "three.pdf"},
    ]

    context = generate_module.build_local_context(docs)

    assert "one.pdf" in context
    assert "two.pdf" in context
    assert "three.pdf" not in context
    assert len(context) <= generate_module.LOCAL_CONTEXT_CHAR_LIMIT + 20


def test_local_generation_uses_small_token_budget(monkeypatch):
    calls = []

    class FakeLlm:
        async def astream(self, prompt):
            yield SimpleNamespace(content="ok")

    def fake_create_llm(**kwargs):
        calls.append(kwargs)
        return FakeLlm()

    monkeypatch.setattr(generate_module, "create_llm", fake_create_llm)

    async def collect():
        output = []
        async for token in generate_module.generate_answer(
            "Summarize",
            [{"text": "Short context", "document_name": "doc.pdf"}],
            llm_provider="openai-compatible",
            openai_base_url="http://localhost:8001/v1",
        ):
            output.append(token)
        return output

    assert asyncio.run(collect()) == ["ok"]
    assert calls[0]["max_tokens"] == generate_module.LOCAL_MAX_TOKENS
    assert calls[0]["temperature"] == 0.1


def test_identity_fallback_extracts_resume_name():
    docs = [
        {
            "text": "Feb 2022 Diploma in Computer Science GPA 3.71",
            "document_name": "resume.pdf",
        },
        {
            "text": "EZARYF BIN HAMDAN JUNIOR FULL-STACK SOFTWARE ENGINEER Kuala Lumpur Email: ezaryfhamdan@gmail.com",
            "document_name": "resume.pdf",
        },
    ]

    answer = generate_module.build_extractive_fallback(docs, query="what is the name of this person")

    assert answer == "Based on the retrieved document, this person is Ezaryf Bin Hamdan. [Source 2]"


def test_fallback_hides_timeout_details_and_summarizes_cleanly():
    docs = [
        {
            "text": (
                "EZARYF BIN HAMDAN JUNIOR FULL-STACK SOFTWARE ENGINEER Kuala Lumpur, 54200 "
                "CAREER OBJECTIVE Dynamic Computer Science graduate specializing in Full-stack "
                "Programming for software development, web development, AI, and database management."
            ),
            "document_name": "resume.pdf",
        },
        {
            "text": (
                "Feb 2022 Diploma in Computer Science, UiTM Kampus Arau - GPA: 3.71 / 4.00 "
                "ACHIEVEMENT Aug 2022 - Vice Chancellor Award and First Class, Faculty of Computer "
                "Science and Mathematics Aug 2024 Present Packaged App Development Associate"
            ),
            "document_name": "resume.pdf",
        }
    ]

    answer = generate_module.build_extractive_fallback(docs, query="summarize this")

    assert "ReadTimeout" not in answer
    assert "local model did not produce" not in answer.lower()
    assert "facebook/opt-125m" not in answer
    assert "Based on the retrieved document" in answer
    assert "[Source 1]" in answer
    assert "Junior Full Stack Software Engineer" in answer
    assert "Technical focus includes Full-stack Programming" in answer
    assert "Education: Diploma in Computer Science" in answer
    assert "Vice Chancellor Award" in answer
    assert "Tel.:" not in answer


def test_citation_page_uses_one_based_chunk_index():
    citations = extract_citations(
        "This person is Ezaryf Bin Hamdan. [Source 1]",
        [{"text": "EZARYF BIN HAMDAN", "document_name": "resume.pdf", "chunk_index": 0}],
    )

    assert citations[0]["page"] == 1


def test_local_generation_timeout_streams_extractive_fallback(monkeypatch):
    async def fake_generate(*args, **kwargs):
        if False:
            yield ""
        raise TimeoutError()

    monkeypatch.setattr(graph_module, "HYBRID_USE", False)
    monkeypatch.setattr(
        graph_module,
        "search_documents",
        lambda *args, **kwargs: asyncio.sleep(
            0,
            result=[
                {"text": "Ezaryf has experience in AI and machine learning.", "document_name": "resume.pdf"},
                {"text": "He works on software development and data management.", "document_name": "resume.pdf"},
            ],
        ),
    )
    monkeypatch.setattr(
        graph_module,
        "rerank_documents",
        lambda query, documents, top_k: asyncio.sleep(0, result=documents[:top_k]),
    )
    monkeypatch.setattr(graph_module, "generate_answer", fake_generate)

    async def collect_events():
        events = []
        async for event in graph_module.run_rag_pipeline(
            query="Summarize my documents",
            session_id="test",
            has_documents=True,
            llm_provider="openai-compatible",
            openai_base_url="http://localhost:8001/v1",
        ):
            events.append(event)
        return events

    events = asyncio.run(collect_events())

    assert any(
        event["type"] == "status" and "using retrieved document snippets" in event["message"]
        for event in events
    )
    fallback_tokens = [event["content"] for event in events if event["type"] == "token"]
    assert fallback_tokens
    assert "Based on the retrieved document" in fallback_tokens[0]
    assert "ReadTimeout" not in fallback_tokens[0]
    assert "[Source 1]" in fallback_tokens[0]


def test_local_model_apply_replaces_container_and_marks_ready(monkeypatch):
    test_home = Path.cwd() / ".tmp-local-llm-apply-test"
    shutil.rmtree(test_home, ignore_errors=True)
    monkeypatch.setenv("DOCPILOT_HOME", str(test_home))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    stop_calls = []
    run_calls = []
    probes = iter(
        [
            (False, [], "not running"),
            (True, ["google/gemma-2-2b-it"], ""),
        ]
    )

    async def fake_docker_status():
        return True, True, "Docker running."

    async def fake_probe_models():
        return next(probes)

    async def fake_stop():
        stop_calls.append(True)

    async def fake_run(args, *, timeout=30, env=None):
        run_calls.append((args, env))
        return SimpleNamespace(returncode=0, stdout="container-id", stderr="")

    async def fake_logs_tail():
        return "Uvicorn running"

    monkeypatch.setattr(local_llm_module, "_docker_status", fake_docker_status)
    monkeypatch.setattr(local_llm_module, "_probe_models", fake_probe_models)
    monkeypatch.setattr(local_llm_module, "_stop_managed_container", fake_stop)
    monkeypatch.setattr(local_llm_module, "_run", fake_run)
    monkeypatch.setattr(local_llm_module, "_logs_tail", fake_logs_tail)

    try:
        asyncio.run(local_llm_module._apply_model("google/gemma-2-2b-it", "safe_10gb"))
        status = local_llm_module.get_local_model_status()

        assert stop_calls == [True]
        assert run_calls
        assert status["state"] == "ready"
        assert status["served_model"] == "google/gemma-2-2b-it"
    finally:
        shutil.rmtree(test_home, ignore_errors=True)
