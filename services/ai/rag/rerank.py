"""
Reranking: reorder retrieved documents using a cross-encoder.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_cross_encoder: CrossEncoder | None = None
_executor = ThreadPoolExecutor(max_workers=1)


def _get_reranker() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info(f"[rerank] Loading cross-encoder: {RERANK_MODEL}")
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


async def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Rerank documents by relevance to the query using a cross-encoder."""
    if not documents:
        logger.warning("[rerank] No documents to rerank")
        return []

    if len(documents) <= top_k:
        logger.info(f"[rerank] Fewer docs ({len(documents)}) than top_k ({top_k}), skipping rerank")
        return documents

    logger.info(f"[rerank] Reranking {len(documents)} documents, returning top {top_k}")

    try:
        reranker = _get_reranker()

        # Limit text length to avoid OOM
        pairs = [
            (query, doc.get("text", "")[:2000])
            for doc in documents
        ]

        # Run sync predict in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            _executor,
            lambda: reranker.predict(pairs),
        )

        # Add scores to documents and sort by score (descending)
        reranked = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            reranked.append(doc_copy)

        # Sort by rerank score descending
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        result = reranked[:top_k]
        logger.info(f"[rerank] Reranked, top score: {result[0].get('rerank_score', 0):.4f}")

        return result

    except Exception as e:
        logger.error(f"[rerank] Reranking failed: {e}")
        # Fallback: return original top_k documents
        return documents[:top_k]