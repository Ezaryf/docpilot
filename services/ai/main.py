"""
DocPilot AI Service — FastAPI + LangGraph Agentic RAG
"""

import os
import json
import asyncio
import uuid
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
) -> AsyncGenerator[str, None]:
    """Run the RAG pipeline and yield SSE events."""
    logger.info(
        f"[chat] Session: {session_id}, Query: {query[:80]}, Documents: {document_names or 'all'}, HasDocuments: {has_documents}, Model: {llm_model or 'env-default'}"
    )
    try:
        async for event in run_rag_pipeline(
            query,
            session_id,
            document_names,
            has_documents,
            groq_api_key,
            llm_model,
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


# ──────────────────── Evaluation ────────────────────


@app.post("/api/eval")
async def run_evaluation():
    """Run evaluation queries and return metrics."""
    from rag.retrieve import get_collection_info

    info = await get_collection_info()
    if info["points_count"] == 0:
        return {
            "results": [],
            "metrics": {
                "hit_at_5": 0,
                "avg_latency_ms": 0,
                "citation_coverage": 0,
                "groundedness": 0,
            },
        }

    # Run sample queries through the pipeline and measure
    eval_queries = [
        "What is the main topic of the uploaded documents?",
        "Summarize the key findings",
        "What methodology was used?",
        "What are the conclusions?",
        "List the important metrics mentioned",
    ]

    results = []
    import time

    for query in eval_queries:
        start = time.time()
        full_response = ""
        citations_list = []
        trace_list = []
        rewritten = False

        async for event in run_rag_pipeline(query, "eval"):
            if event.get("type") == "token":
                full_response += event["content"]
            elif event.get("type") == "citations":
                citations_list = event["citations"]
            elif event.get("type") == "trace":
                trace_list = event["trace"]
                rewritten = any(s["step"] == "query_rewrite" for s in trace_list)

        elapsed = (time.time() - start) * 1000
        results.append({
            "query": query,
            "relevant": bool(citations_list),
            "rewritten": rewritten,
            "latency_ms": elapsed,
            "citations": len(citations_list),
            "score": min(1.0, len(citations_list) / 3) if citations_list else 0.2,
        })

    # Aggregate metrics
    n = len(results)
    m = {
        "hit_at_5": sum(1 for r in results if r["relevant"]) / n if n else 0,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / n if n else 0,
        "citation_coverage": sum(1 for r in results if r["citations"] > 0) / n if n else 0,
        "groundedness": sum(r["score"] for r in results) / n if n else 0,
    }

    return {"results": results, "metrics": m}


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
