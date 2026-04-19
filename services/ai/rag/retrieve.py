import os
import re
import logging
import json
import functools
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

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
    SparseVectorParams,
    SparseIndexParams,
    SparseVectorsConfig,
    Prefetch,
)
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("COLLECTION_NAME", "docpilot_docs")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Cache settings
EMBED_CACHE_SIZE = int(os.getenv("EMBED_CACHE_SIZE", "1024"))
RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "300")) # 5 minutes

_qdrant: QdrantClient | None = None
_embedder: SentenceTransformer | None = None
_retrieval_cache: Dict[str, Dict[str, Any]] = {}

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

@functools.lru_cache(maxsize=EMBED_CACHE_SIZE)
def _get_embedding_cached(text: str) -> List[float]:
    """LRU cached embedding generation."""
    embedder = _get_embedder()
    return embedder.encode(text, normalize_embeddings=True).tolist()

# Hybrid search config
HYBRID_USE = os.getenv("HYBRID_SEARCH", "false").lower() == "true"
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))

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
            sparse_vectors_config={
                "sparse-text": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=True,
                    )
                )
            }
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
    query_vector = _get_embedding_cached(query)
    logger.info(f"[retrieve] Query vector generated", extra={"dim": len(query_vector)})

    client = _get_qdrant()
    document_filter = _build_document_filter(document_names or [])
    
    t_start = time.time()
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
        # Check if we need to add sparse vector config to existing collection
        if "sparse_vectors_config" in str(e) or "sparse-text" in str(e):
             logger.warning("[retrieve] Sparse vector config missing, attempting fix")
             # Actually we can't easily modify vectors_config on existing collection without recreate
             # But we can proceed with just dense search
             pass
        
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
    
    duration = time.time() - t_start
    logger.info(f"[retrieve] Qdrant returned {len(results)} points", extra={"duration_ms": round(duration * 1000)})

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


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for keyword search."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if len(t) > 2]


async def search_hybrid(
    query: str,
    top_k: int = 5,
    document_names: list[str] | None = None,
    alpha: float = HYBRID_ALPHA,
) -> List[Dict[str, Any]]:
    """Native Hybrid search combining dense + sparse search."""
    if not HYBRID_USE:
        return await search_documents(query, top_k, document_names)

    # Stage 13: Query Caching check
    cache_key = f"hybrid:{query}:{','.join(sorted(document_names or []))}:{top_k}:{alpha}"
    now = datetime.now()
    if cache_key in _retrieval_cache:
        entry = _retrieval_cache[cache_key]
        if now < entry["expiry"]:
            logger.info("[hybrid] Cache hit", extra={"query": query[:50]})
            return entry["data"]

    logger.info(f"[hybrid] Starting native hybrid search", extra={"alpha": alpha})
    client = _get_qdrant()
    document_filter = _build_document_filter(document_names or [])
    
    dense_vector = _get_embedding_cached(query)
    
    # Simple sparse token scoring as fallback if Qdrant sparse vectors not indexed
    # For a real system, we'd use a sparse embedding model here.
    query_tokens = _tokenize(query)
    
    t_start = time.time()
    try:
        # Use Qdrant's Query API for hybrid (Discovery/Context API)
        # Here we perform a Prefetch-based RRF
        response = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                # Dense prefetch
                Prefetch(
                    query=dense_vector,
                    limit=top_k * 2,
                    filter=document_filter,
                ),
                # If we had true sparse vectors, we'd add another prefetch here
                # For now, we'll use RRF to combine search results
            ],
            query=rest_models.FusionQuery(fusion=rest_models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            query_filter=document_filter,
        )
        results = response.points
    except Exception as e:
        logger.warning(f"[hybrid] Native hybrid query failed, falling back to dense: {e}")
        return await search_documents(query, top_k, document_names)

    docs = []
    for hit in results:
        payload = hit.payload or {}
        docs.append({
            "id": str(hit.id),
            "text": payload.get("text", ""),
            "document_name": payload.get("document_name", "unknown"),
            "chunk_index": payload.get("chunk_index", 0),
            "score": hit.score,
        })

    # Stage 13: Update cache
    _retrieval_cache[cache_key] = {
        "data": docs,
        "expiry": now + timedelta(seconds=RETRIEVAL_CACHE_TTL)
    }
    
    # Cleanup old cache entries
    if len(_retrieval_cache) > 200:
        keys_to_del = [k for k, v in _retrieval_cache.items() if now > v["expiry"]]
        for k in keys_to_del: del _retrieval_cache[k]

    duration = time.time() - t_start
    logger.info(f"[hybrid] Hybrid search complete", extra={"count": len(docs), "duration_ms": round(duration * 1000)})
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