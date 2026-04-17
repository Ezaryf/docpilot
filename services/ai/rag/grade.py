"""
Relevance grading: decide if retrieved documents are relevant to the query.
"""

import os
import re
import logging
from rag.llm import create_groq_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "up",
    "with",
    "my",
    "your",
    "our",
    "uploaded",
    "document",
    "documents",
    "file",
    "files",
}
def _normalize_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOP_WORDS
    }


def _has_lexical_overlap(query: str, document_text: str) -> bool:
    query_tokens = _normalize_tokens(query)
    if not query_tokens:
        return False

    doc_tokens = _normalize_tokens(document_text[:2000])
    overlap = query_tokens & doc_tokens
    return len(overlap) >= 1


async def grade_relevance(
    query: str,
    document_text: str,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
) -> bool:
    """Grade whether a document chunk is relevant to the query."""
    logger.debug(f"[grade_relevance] Grading document for query: {query[:60]}")

    # Cheap lexical fallback keeps broad "summarize my uploaded docs" style queries
    # from being rejected when the LLM grader is overly strict.
    if _has_lexical_overlap(query, document_text):
        logger.debug("[grade_relevance] Accepted via lexical overlap")
        return True

    llm = create_groq_llm(
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        temperature=0,
        max_tokens=100,
    )

    prompt = f"""You are a relevance grader. Given a user question and a document chunk,
determine if the document contains information relevant to answering the question.

Respond with ONLY "yes" or "no".

Question: {query}

Document: {document_text[:1000]}

Is this document relevant to the question? (yes/no):"""

    try:
        response = await llm.ainvoke(prompt)
        answer = response.content.strip().lower()
        result = answer.startswith("yes")
        logger.debug(f"[grade_relevance] Result: {result} (answer: {answer})")
        return result
    except Exception as e:
        logger.error(f"[grade_relevance] LLM failed: {e}")
        raise


async def grade_documents(
    query: str,
    documents: list,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
) -> tuple[list, list]:
    """Grade a list of documents and split into relevant/irrelevant."""
    logger.info(f"[grade_documents] Grading {len(documents)} documents")
    relevant = []
    irrelevant = []

    for i, doc in enumerate(documents):
        logger.debug(f"[grade_documents] Grading document {i+1}/{len(documents)}")
        try:
            is_relevant = await grade_relevance(
                query,
                doc["text"],
                groq_api_key=groq_api_key,
                llm_model=llm_model,
            )
        except Exception as e:
            logger.error(f"[grade_documents] Failed to grade doc {i}: {e}")
            is_relevant = False
        if is_relevant:
            relevant.append(doc)
        else:
            irrelevant.append(doc)

    if not relevant and documents:
        logger.warning("[grade_documents] No documents graded relevant; falling back to top retrieved chunks")
        relevant = documents[: min(2, len(documents))]
        irrelevant = documents[len(relevant) :]

    logger.info(f"[grade_documents] Result: {len(relevant)} relevant, {len(irrelevant)} irrelevant")
    return relevant, irrelevant
