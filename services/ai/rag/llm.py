import os

from langchain_groq import ChatGroq

DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


def resolve_llm_config(groq_api_key: str | None = None, llm_model: str | None = None) -> dict[str, str]:
    return {
        "api_key": (groq_api_key or DEFAULT_GROQ_API_KEY).strip(),
        "model": (llm_model or DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL,
    }


def create_groq_llm(
    *,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    temperature: float,
    max_tokens: int,
):
    config = resolve_llm_config(groq_api_key, llm_model)
    return ChatGroq(
        api_key=config["api_key"],
        model=config["model"],
        temperature=temperature,
        max_tokens=max_tokens,
    )
