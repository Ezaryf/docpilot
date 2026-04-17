"""
Document ingestion: parse → clean → chunk → embed → upsert into Qdrant.
Uses Unstructured for layout-aware parsing and chunking.
"""

import os
import re
import uuid
import io
import logging
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("COLLECTION_NAME", "docpilot_docs")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Lazy globals
_qdrant: QdrantClient | None = None
_embedder: SentenceTransformer | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    return _qdrant


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ──────────────────── Document Parsing (Unstructured) ────────────────────
def _parse_with_unstructured(filename: str, content: bytes, ext: str) -> List[str]:
    """Parse document using Unstructured to extract elements."""
    logger.info(f"[ingest] Parsing {filename} with Unstructured")

    file_obj = io.BytesIO(content)
    file_obj.name = filename

    try:
        elements = partition(file=file_obj, filename=filename)
        logger.info(f"[ingest] Unstructured found {len(elements)} elements")
        return elements
    except Exception as e:
        logger.warning(f"[ingest] Unstructured failed: {e}, falling back to basic parser")
        raise


# ──────────────────── Cleaning / Normalization ────────────────────
def _clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""

    cleaned = text

    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
    cleaned = cleaned.strip()

    return cleaned


def _clean_elements(elements: List) -> List[str]:
    """Clean unstructured elements and extract text."""
    cleaned_texts = []

    for elem in elements:
        text = str(elem).strip()
        if not text:
            continue

        text = _clean_text(text)

        if len(text) < 10:
            continue

        cleaned_texts.append(text)

    return cleaned_texts


# ──────────────────── Chunking (Unstructured + Fallback) ────────────────────
def _chunk_with_unstructured(texts: List[str]) -> List[str]:
    """Chunk text elements using Unstructured's chunker."""
    try:
        from unstructured.documents.elements import Text
        elements = [Text(text=t) for t in texts if t]
        chunks = chunk_elements(elements, max_characters=CHUNK_SIZE)
        return [str(c) for c in chunks]
    except Exception as e:
        logger.warning(f"[ingest] Unstructured chunking failed: {e}")
        raise


def _chunk_fallback(texts: List[str]) -> List[str]:
    """Fallback: simple word-based chunking."""
    all_chunks = []

    for text in texts:
        words = text.split()
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + CHUNK_SIZE])
            if chunk.strip():
                all_chunks.append(chunk.strip())
            i += CHUNK_SIZE - CHUNK_OVERLAP

    return all_chunks


def chunk_text(texts: List[str]) -> List[str]:
    """Main chunking function: try Unstructured, fallback to simple."""
    if not texts:
        return []

    try:
        chunks = _chunk_with_unstructured(texts)
        if chunks:
            logger.info(f"[ingest] Unstructured chunking: {len(chunks)} chunks")
            return chunks
    except Exception as e:
        logger.warning(f"[ingest] Chunking failed, using fallback: {e}")

    chunks = _chunk_fallback(texts)
    logger.info(f"[ingest] Fallback chunking: {len(chunks)} chunks")
    return chunks


# ──────────────────── Backward Compatibility ────────────────────
def parse_document(filename: str, content: bytes, ext: str) -> str:
    """Parse single document to plain text (legacy interface)."""
    elements = _parse_with_unstructured(filename, content, ext)
    cleaned = _clean_elements(elements)
    return "\n\n".join(cleaned)


# ──────────────────── Ingest Pipeline ────────────────────
async def ingest_document(filename: str, content: bytes, ext: str) -> Dict[str, Any]:
    """Full ingestion: parse, clean, chunk, embed, upsert."""
    logger.info(f"[ingest] Starting ingestion for: {filename} (ext: {ext})")

    # Step 1: Parse with Unstructured
    logger.info("[ingest] Step 1: Parse document (Unstructured)")
    try:
        elements = _parse_with_unstructured(filename, content, ext)
    except Exception as e:
        logger.error(f"[ingest] Parse failed: {e}")
        raise

    if not elements:
        logger.error("[ingest] No elements extracted")
        raise ValueError("Document could not be parsed")

    logger.info(f"[ingest] Extracted {len(elements)} elements")

    # Step 2: Clean elements
    logger.info("[ingest] Step 2: Clean elements")
    try:
        cleaned_texts = _clean_elements(elements)
    except Exception as e:
        logger.error(f"[ingest] Cleaning failed: {e}")
        raise

    if not cleaned_texts:
        logger.error("[ingest] No text after cleaning")
        raise ValueError("Document is empty after cleaning")

    logger.info(f"[ingest] Cleaned {len(cleaned_texts)} text blocks")

    # Step 3: Chunk text
    logger.info("[ingest] Step 3: Chunk text")
    try:
        chunks = chunk_text(cleaned_texts)
    except Exception as e:
        logger.error(f"[ingest] Chunking failed: {e}")
        raise

    if not chunks:
        logger.error("[ingest] No chunks produced")
        raise ValueError("No chunks produced from document")

    logger.info(f"[ingest] Produced {len(chunks)} chunks")

    # Step 4: Generate embeddings
    logger.info("[ingest] Step 4: Generate embeddings")
    try:
        embedder = _get_embedder()
        embeddings = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    except Exception as e:
        logger.error(f"[ingest] Embedding failed: {e}")
        raise

    logger.info(f"[ingest] Generated {len(embeddings)} embeddings, dim: {embeddings.shape[1]}")

    # Step 5: Upsert to Qdrant
    logger.info("[ingest] Step 5: Upsert to Qdrant")
    try:
        client = _get_qdrant()
    except Exception as e:
        logger.error(f"[ingest] Qdrant connection failed: {e}")
        raise

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "document_name": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
        )

    batch_size = 100
    for i in range(0, len(points), batch_size):
        try:
            client.upsert(collection_name=COLLECTION, points=points[i : i + batch_size])
            logger.info(f"[ingest] Upserted batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
        except Exception as e:
            logger.error(f"[ingest] Batch upsert failed (batch {i//batch_size + 1}): {e}")
            raise

    total_chars = sum(len(c) for c in chunks)
    logger.info(f"[ingest] Complete: {filename} ({len(chunks)} chunks, {total_chars} chars)")

    return {
        "document_name": filename,
        "chunks": len(chunks),
        "characters": total_chars,
        "status": "ready",
    }