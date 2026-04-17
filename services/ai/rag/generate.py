"""
Grounded answer generation with citations.
"""

import os
import logging
from typing import List, Dict, AsyncGenerator
from rag.llm import create_groq_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _build_context(documents: List[Dict]) -> str:
    """Build context string from retrieved documents."""
    parts = []
    for i, doc in enumerate(documents):
        parts.append(
            f"[Source {i + 1}: {doc['document_name']}]\n{doc['text']}"
        )
    return "\n\n---\n\n".join(parts)


async def generate_answer(
    query: str,
    documents: List[Dict],
    groq_api_key: str | None = None,
    llm_model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate a grounded answer with streaming."""
    logger.info(f"[generate] Starting answer generation, documents: {len(documents)}")
    if not documents:
        logger.warning("[generate] No documents provided, generating fallback response")

    llm = create_groq_llm(
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        temperature=0.4,
        max_tokens=2048,
    )

    if not documents:
        prompt = f"""You are DocPilot, an AI document assistant. The user asked a question
but no relevant documents were found in the knowledge base.

Respond helpfully, noting that no documents were found that match the query.
Suggest the user upload relevant documents.

Question: {query}

Response:"""
    else:
        context = _build_context(documents)
        prompt = f"""You are DocPilot, an AI document assistant. Answer the user's question
based ONLY on the provided document sources. Follow these rules:

1. Use information ONLY from the provided sources
2. Reference sources using [Source N] notation
3. If the sources don't contain enough information, say so
4. Be comprehensive but concise
5. Use markdown formatting for readability

Sources:
{context}

Question: {query}

Answer:"""

    # Stream tokens
    try:
        async for chunk in llm.astream(prompt):
            if chunk.content:
                yield chunk.content
        logger.info("[generate] Answer generation complete")
    except Exception as e:
        logger.error(f"[generate] LLM streaming failed: {e}")
        raise


async def generate_direct_answer(
    query: str,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate a direct answer without retrieval (for non-retrieval queries)."""
    llm = create_groq_llm(
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        temperature=0.4,
        max_tokens=2048,
    )

    prompt = f"""You are DocPilot, an AI document assistant. The user asked a general question
that doesn't require document retrieval.

Answer the question helpfully and concisely. If the question would benefit from
uploaded documents, mention that.

Question: {query}

Answer:"""

    async for chunk in llm.astream(prompt):
        if chunk.content:
            yield chunk.content
