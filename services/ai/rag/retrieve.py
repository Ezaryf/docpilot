import os
import logging
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    SearchParams,
    FieldCondition,
    MatchValue,
    TextIndexParams,
    TokenizerType,
    KeywordIndexParams,
)
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("COLLECTION_NAME", "docpilot_docs")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

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

async def ensure_collection():
    """Create the Qdrant collection if it doesn't exist."""
    client = _get_qdrant()
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    if COLLECTION not in names:
        embedder = _get_embedder()
        dim = embedder.get_sentence_embedding_dimension()
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        client.create_field_index(
            collection_name=COLLECTION,
            field_name="document_name",
            field_schema=KeywordIndexParams(),
        )
        print(f"Created Qdrant collection: {COLLECTION} (dim={dim})")
    else:
        print(f"Qdrant collection exists: {COLLECTION}")
        try:
            client.create_field_index(
                collection_name=COLLECTION,
                field_name="document_name",
                field_schema=KeywordIndexParams(),
            )
            print(f"Created keyword index on document_name")
        except Exception as e:
            logger.info(f"Index may already exist: {e}")

def _build_document_filter(document_names: list[str]) -> Filter | None:
    clean_names = [name for name in document_names if name.strip()]
    if not clean_names:
        return None

    return Filter(
        should=[
            FieldCondition(
                key="document_name",
                match=MatchValue(value=name),
            )
            for name in clean_names
        ]
    )

async def search_documents(
    query: str,
    top_k: int = 5,
    document_names: list[str] | None = None,
) -> List[Dict[str, Any]]:
    """Dense vector search against the collection."""
    logger.info(f"[retrieve] Encoding query: {query[:60]}")
    embedder = _get_embedder()
    try:
        query_vector = embedder.encode(query, normalize_embeddings=True).tolist()
    except Exception as e:
        logger.error(f"[retrieve] Embedding failed: {e}")
        raise
    logger.info(f"[retrieve] Query vector encoded, dimension: {len(query_vector)}")

    logger.info(f"[retrieve] Searching Qdrant collection '{COLLECTION}' with top_k={top_k}")
    client = _get_qdrant()
    document_filter = _build_document_filter(document_names or [])
    try:
        response = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128, exact=False),
            query_filter=document_filter,
        )
        results = response.points
    except Exception as e:
        error_message = str(e)
        if document_filter and "Index required but not found" in error_message:
            logger.warning(
                "[retrieve] document_name filter index missing in Qdrant; falling back to client-side filtering"
            )
            response = client.query_points(
                collection_name=COLLECTION,
                query=query_vector,
                limit=max(top_k * 10, 20),
                with_payload=True,
                search_params=SearchParams(hnsw_ef=128, exact=False),
            )
            requested_names = {name.strip().lower() for name in (document_names or []) if name.strip()}
            results = [
                point
                for point in response.points
                if ((point.payload or {}).get("document_name", "").strip().lower() in requested_names)
            ][:top_k]
        else:
            logger.error(f"[retrieve] Qdrant query failed: {e}")
            raise
    logger.info(f"[retrieve] Qdrant returned {len(results)} points")

    docs = []
    for hit in results:
        payload = hit.payload or {}
        docs.append(
            {
                "id": str(hit.id),
                "text": payload.get("text", ""),
                "document_name": payload.get("document_name", "unknown"),
                "chunk_index": payload.get("chunk_index", 0),
                "score": hit.score,
            }
        )
    logger.info(f"[retrieve] Parsed {len(docs)} documents")
    return docs

async def list_indexed_documents() -> List[Dict[str, Any]]:
    """List unique documents in the collection."""
    client = _get_qdrant()
    try:
        info = client.get_collection(COLLECTION)
        # Scroll a sample to get document names
        records, _ = client.scroll(
            collection_name=COLLECTION, limit=500, with_payload=True
        )
        doc_names = {}
        for r in records:
            name = r.payload.get("document_name", "unknown") if r.payload else "unknown"
            if name not in doc_names:
                doc_names[name] = {
                    "name": name,
                    "chunks": 0,
                    "total_chunks": r.payload.get("total_chunks", 0) if r.payload else 0,
                }
            doc_names[name]["chunks"] += 1

        return list(doc_names.values())
    except Exception:
        return []

async def get_collection_info() -> Dict[str, Any]:
    """Get collection stats."""
    client = _get_qdrant()
    try:
        info = client.get_collection(COLLECTION)
        return {
            "name": COLLECTION,
            "points_count": info.points_count or 0,
            "vectors_count": info.vectors_count or 0,
        }
    except Exception:
        return {"name": COLLECTION, "points_count": 0, "vectors_count": 0}

async def delete_document(document_name: str) -> bool:
    """Delete all chunks belonging to a document from Qdrant."""
    logger.info(f"[delete] Removing document: {document_name}")
    client = _get_qdrant()
    try:
        # Scroll through all points and find matching ones
        points_to_delete = []
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=COLLECTION,
                limit=100,
                with_payload=True,
                offset=offset,
            )
            for point in results:
                payload = point.payload or {}
                if payload.get("document_name") == document_name:
                    points_to_delete.append(point.id)
            if offset is None:
                break

        if points_to_delete:
            client.delete(
                collection_name=COLLECTION,
                points_selector=points_to_delete,
            )
            logger.info(f"[delete] Deleted {len(points_to_delete)} chunks for: {document_name}")
        else:
            logger.info(f"[delete] No chunks found for: {document_name}")
        return True
    except Exception as e:
        logger.error(f"[delete] Failed to delete {document_name}: {e}")
        raise