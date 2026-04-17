"""
Query rewriting: reformulate queries for better retrieval.
"""

import os
import logging
from rag.llm import create_groq_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def rewrite_query(
    original_query: str,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
) -> str:
    """Rewrite a query to improve retrieval results."""
    logger.info(f"[rewrite_query] Rewriting query: {original_query[:60]}")
    llm = create_groq_llm(
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        temperature=0.3,
        max_tokens=200,
    )

    prompt = f"""You are a search query optimizer. The original query did not return
sufficiently relevant results. Rewrite it to be more specific and likely to match
relevant document chunks.

Rules:
- Keep the core intent
- Add relevant synonyms or related terms
- Make it more specific
- Output ONLY the rewritten query, nothing else

Original query: {original_query}

Rewritten query:"""

    try:
        response = await llm.ainvoke(prompt)
        rewritten = response.content.strip()
        result = rewritten if rewritten else original_query
        logger.info(f"[rewrite_query] Result: {result[:60]}")
        return result
    except Exception as e:
        logger.error(f"[rewrite_query] Failed: {e}")
        raise
