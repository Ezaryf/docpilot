"""
Document ingestion: parse → chunk → embed → upsert into Qdrant.
"""

import os
import uuid
import logging
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

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


# ──────────────────── Document Parsing ────────────────────


def _parse_pdf(content: bytes) -> str:
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(content))
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def _parse_docx(content: bytes) -> str:
    from docx import Document
    import io

    doc = Document(io.BytesIO(content))
    return "\n".join(para.text for para in doc.paragraphs)


def _parse_text(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


def parse_document(filename: str, content: bytes, ext: str) -> str:
    """Parse document content to plain text."""
    parsers = {
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
        ".txt": _parse_text,
        ".md": _parse_text,
    }
    parser = parsers.get(ext)
    if not parser:
        raise ValueError(f"Unsupported format: {ext}")
    return parser(content)


# ──────────────────── Chunking ────────────────────


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


# ──────────────────── Ingest Pipeline ────────────────────


async def ingest_document(filename: str, content: bytes, ext: str) -> Dict[str, Any]:
    """Full ingestion: parse, chunk, embed, upsert."""
    logger.info(f"[ingest] Starting ingestion for: {filename} (ext: {ext})")

    # Parse
    logger.info("[ingest] Step 1: Parse document")
    try:
        text = parse_document(filename, content, ext)
    except Exception as e:
        logger.error(f"[ingest] Parse failed: {e}")
        raise
    if not text.strip():
        logger.error("[ingest] Document is empty after parsing")
        raise ValueError("Document is empty or could not be parsed")
    logger.info(f"[ingest] Parsed text length: {len(text)} chars")

    # Chunk
    logger.info("[ingest] Step 2: Chunk text")
    try:
        chunks = chunk_text(text)
    except Exception as e:
        logger.error(f"[ingest] Chunking failed: {e}")
        raise
    if not chunks:
        logger.error("[ingest] No chunks produced")
        raise ValueError("No chunks produced from document")
    logger.info(f"[ingest] Produced {len(chunks)} chunks")

    # Embed
    logger.info("[ingest] Step 3: Generate embeddings")
    try:
        embedder = _get_embedder()
        embeddings = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    except Exception as e:
        logger.error(f"[ingest] Embedding failed: {e}")
        raise
    logger.info(f"[ingest] Generated {len(embeddings)} embeddings, dim: {embeddings.shape[1]}")

    # Upsert to Qdrant
    logger.info("[ingest] Step 4: Upsert to Qdrant")
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

    # Batch upsert (100 at a time)
    batch_size = 100
    for i in range(0, len(points), batch_size):
        try:
            client.upsert(collection_name=COLLECTION, points=points[i : i + batch_size])
            logger.info(f"[ingest] Upserted batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
        except Exception as e:
            logger.error(f"[ingest] Batch upsert failed (batch {i//batch_size + 1}): {e}")
            raise

    logger.info(f"[ingest] Complete: {filename} ({len(chunks)} chunks, {len(text)} chars)")
    return {
        "document_name": filename,
        "chunks": len(chunks),
        "characters": len(text),
        "status": "ready",
    }
