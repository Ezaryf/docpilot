import os
from typing import Literal, TypedDict

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


def _normalize_provider(provider: str | None) -> LlmProvider:
    return "openai-compatible" if provider == "openai-compatible" else "groq"


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
            "base_url": (openai_base_url or DEFAULT_OPENAI_BASE_URL).strip(),
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
        return ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return ChatGroq(
        api_key=config["api_key"],
        model=config["model"],
        temperature=temperature,
        max_tokens=max_tokens,
    )


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
