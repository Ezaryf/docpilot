"""
Citation formatting: extract and format source citations from generated answers.
"""

import re
import uuid
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_citations(
    answer: str, documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Extract [Source N] references from the answer and map to documents."""
    logger.info(f"[citations] Extracting citations from answer ({len(answer)} chars) with {len(documents)} documents")
    # Find all [Source N] patterns in the answer
    pattern = r"\[Source\s+(\d+)\]"
    referenced = set(int(m) for m in re.findall(pattern, answer))
    logger.info(f"[citations] Found explicit references: {referenced}")

    citations = []
    for idx in sorted(referenced):
        # Source N is 1-indexed
        doc_idx = idx - 1
        if 0 <= doc_idx < len(documents):
            doc = documents[doc_idx]
            citations.append(
                {
                    "id": str(uuid.uuid4()),
                    "documentName": doc.get("document_name", "unknown"),
                    "chunkText": doc.get("text", "")[:300],
                    "page": doc.get("chunk_index", 0),
                    "score": doc.get("score", 0.0),
                }
            )

    # If no explicit references found but we have documents, include top ones
    if not citations and documents:
        logger.info("[citations] No explicit refs, adding top 3 documents as fallback")
        for doc in documents[:3]:
            citations.append(
                {
                    "id": str(uuid.uuid4()),
                    "documentName": doc.get("document_name", "unknown"),
                    "chunkText": doc.get("text", "")[:300],
                    "page": doc.get("chunk_index", 0),
                    "score": doc.get("score", 0.0),
                }
            )

    logger.info(f"[citations] Returning {len(citations)} citations")
    return citations
