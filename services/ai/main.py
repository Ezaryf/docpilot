"""
DocPilot AI Service — FastAPI + LangGraph Agentic RAG
"""

import os
import json
import asyncio
import uuid
import logging
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Stage 14: Structured Monitoring
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

from rag.ingest import ingest_document
from rag.graph import run_rag_pipeline

app = FastAPI(
    title="DocPilot AI Service",
    version="1.0.0",
    description="Agentic RAG backend with LangGraph, Qdrant, and Groq",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    session_id: str = ""
    document_names: list[str] = []
    has_documents: bool = False
    groq_api_key: str | None = None
    llm_model: str | None = None
    llm_provider: str | None = None
    openai_base_url: str | None = None
    openai_api_key: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


# ──────────────────── Health ────────────────────


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="1.0.0")


# ──────────────────── Chat (SSE Streaming) ────────────────────


async def _stream_rag(
    query: str,
    session_id: str,
    document_names: list[str],
    has_documents: bool,
    groq_api_key: str | None,
    llm_model: str | None,
    llm_provider: str | None,
    openai_base_url: str | None,
    openai_api_key: str | None,
) -> AsyncGenerator[str, None]:
    """Run the RAG pipeline and yield SSE events."""
    logger.info(
        f"[chat] Session: {session_id}, Query: {query[:80]}, Documents: {document_names or 'all'}, HasDocuments: {has_documents}, Provider: {llm_provider or 'groq'}, Model: {llm_model or 'env-default'}"
    )
    try:
        async for event in run_rag_pipeline(
            query=query,
            session_id=session_id,
            document_names=document_names,
            has_documents=has_documents,
            groq_api_key=groq_api_key,
            llm_model=llm_model,
            llm_provider=llm_provider,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"
        logger.info(f"[chat] Stream complete for session: {session_id}")
    except Exception as e:
        logger.error(f"[chat] Stream error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    logger.info(f"[chat] Received request: {req.query[:80]}")
    if not req.query.strip():
        logger.warning("[chat] Empty query rejected")
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    return StreamingResponse(
        _stream_rag(
            req.query,
            req.session_id,
            req.document_names,
            req.has_documents,
            req.groq_api_key,
            req.llm_model,
            req.llm_provider,
            req.openai_base_url,
            req.openai_api_key,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────── Upload ────────────────────


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    logger.info(f"[upload] Received file: {file.filename}")
    if not file.filename:
        logger.warning("[upload] No filename provided")
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_ext = {".pdf", ".txt", ".md", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        logger.warning(f"[upload] Unsupported file type: {ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_ext)}",
        )

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        logger.warning(f"[upload] File too large: {len(content)} bytes")
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    logger.info(f"[upload] Processing: {file.filename}, size: {len(content)} bytes")
    try:
        result = await ingest_document(file.filename, content, ext)
        logger.info(f"[upload] Success: {file.filename}")
        return result
    except Exception as e:
        logger.error(f"[upload] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────── Documents List ────────────────────


@app.get("/api/documents")
async def list_documents():
    logger.info("[documents] Listing indexed documents")
    from rag.retrieve import list_indexed_documents
    try:
        docs = await list_indexed_documents()
        logger.info(f"[documents] Found {len(docs)} documents")
        return {"documents": docs}
    except Exception as e:
        logger.error(f"[documents] Failed: {e}")
        raise


@app.delete("/api/documents/{document_name}")
async def delete_document(document_name: str):
    """Delete a document and all its chunks from the index."""
    logger.info(f"[documents] Delete request for: {document_name}")
    from rag.retrieve import delete_document as do_delete
    try:
        await do_delete(document_name)
        logger.info(f"[documents] Deleted: {document_name}")
        return {"status": "deleted", "document_name": document_name}
    except Exception as e:
        logger.error(f"[documents] Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────── Evaluation (Ragas) ────────────────────


@app.post("/api/eval")
async def run_evaluation(request: Optional[dict] = None):
    """
    Run Ragas evaluation on RAG pipeline.
    If no body is provided, runs a batch evaluation on indexed documents.
    """
    from rag.eval import evaluate_rag, run_batch_evaluation
    from rag.retrieve import get_collection_info

    request = request or {}
    llm_provider = request.get("llm_provider")
    llm_model = request.get("llm_model")
    groq_api_key = request.get("groq_api_key")
    openai_base_url = request.get("openai_base_url")
    openai_api_key = request.get("openai_api_key")

    # 1. Check if it's a batch request (no body)
    if not request.get("question") and not request.get("answer"):
        logger.info(f"[eval] Triggering batch evaluation with provider={llm_provider or 'groq'}, model={llm_model or 'env-default'}")
        try:
            return await run_batch_evaluation(
                num_samples=3,
                groq_api_key=groq_api_key,
                llm_model=llm_model,
                llm_provider=llm_provider,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
            )
        except Exception as e:
            logger.error(f"[eval] Batch evaluation failed: {e}")
            return {"error": str(e)}

    # 2. Case: Single query evaluation
    question = request.get("question", "")
    answer = request.get("answer", "")
    contexts = request.get("contexts", [])
    ground_truth = request.get("ground_truth")

    if not question or not answer:
        return {"error": "question and answer required for single evaluation"}

    info = await get_collection_info()
    if info["points_count"] == 0:
        return {"error": "No documents indexed"}

    try:
        results = await evaluate_rag(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            groq_api_key=groq_api_key,
            llm_model=llm_model,
            llm_provider=llm_provider,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
        )
        return {
            "results": [{
                "query": question,
                "score": sum(v[0] for v in results.values() if isinstance(v, list) and v) / len(results) if results else 0,
                "latency_ms": 0,
                "relevant": True,
                "rewritten": False,
                "citations": 0
            }],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"[eval] Failed: {e}")
        return {"error": str(e)}


@app.get("/api/stats")
async def get_stats():
    """Stage 14: Metrics and stats endpoint."""
    from rag.retrieve import get_collection_info, _retrieval_cache
    try:
        info = await get_collection_info()
        return {
            "status": "ok",
            "qdrant": info,
            "cache": {
                "size": len(_retrieval_cache),
                "items": list(_retrieval_cache.keys())[:10] # Show some keys
            },
            "environment": {
                "hybrid_search": os.getenv("HYBRID_SEARCH", "false"),
                "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
                "llm_model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            }
        }
    except Exception as e:
        logger.error(f"[stats] Failed: {e}")
        return {"status": "error", "message": str(e)}


# ──────────────────── Startup ────────────────────


@app.on_event("startup")
async def startup():
    logger.info("[startup] Initializing DocPilot AI Service")
    from rag.retrieve import ensure_collection
    try:
        await ensure_collection()
        logger.info("[startup] Ready")
        print("DocPilot AI Service ready")
    except Exception as e:
        logger.error(f"[startup] Failed: {e}")
        raise
