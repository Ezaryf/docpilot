from rag.eval import serialize_evaluation_result
from rag.llm import resolve_llm_config


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
