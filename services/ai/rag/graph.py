"""
LangGraph Agentic RAG Pipeline

Flow:
  START → route_query → [needs_retrieval?]
                        ├─ YES → retrieve → rerank → grade → [relevant?]
                        │                                     ├─ YES → generate → cite → END
                        │                                     └─ NO → rewrite → retrieve (loop, max 2)
                        └─ NO → direct_answer → END
"""

import time
import logging
from typing import TypedDict, AsyncGenerator, Any

from rag.retrieve import search_documents, search_hybrid, HYBRID_USE
from rag.grade import grade_documents
from rag.rewrite import rewrite_query
from rag.rerank import rerank_documents
from rag.generate import build_extractive_fallback, generate_answer, generate_direct_answer
from rag.citations import extract_citations
from rag.llm import create_llm, is_managed_local_vllm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────── State ────────────────────
class RAGState(TypedDict):
    query: str
    original_query: str
    session_id: str
    documents: list
    relevant_docs: list
    answer: str
    citations: list
    trace: list
    rewrite_count: int
    needs_retrieval: bool


# ──────────────────── Router ────────────────────
async def _route_query(
    query: str,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> bool:
    """Decide if the query needs document retrieval."""
    logger.info(f"[route_query] Processing query: {query[:100]}")
    try:
        llm = create_llm(
            groq_api_key=groq_api_key,
            llm_model=llm_model,
            llm_provider=llm_provider,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            temperature=0,
            max_tokens=50,
        )

        prompt = f"""You are a query router. Determine if this query requires searching
through uploaded documents for an answer, or if it's a general/greeting question.

Respond with ONLY "retrieve" or "direct".

Examples:
- "What does the document say about X?" → retrieve
- "Hello" → direct
- "Summarize the findings" → retrieve
- "What time is it?" → direct
- "Compare the approaches mentioned" → retrieve

Query: {query}

Decision:"""

        response = await llm.ainvoke(prompt)
        decision = response.content.strip().lower()
        result = "retrieve" in decision
        logger.info(f"[route_query] Decision made", extra={"decision": decision, "result": result})
        return result
    except Exception as e:
        logger.error(f"[route_query] Error: {e}")
        raise


def _looks_like_direct_chat(query: str) -> bool:
    normalized = query.strip().lower()
    direct_phrases = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "bye",
        "goodbye",
        "what time is it",
        "how are you",
    }
    return normalized in direct_phrases


# ──────────────────── Pipeline ────────────────────
async def run_rag_pipeline(
    query: str,
    session_id: str,
    document_names: list[str] | None = None,
    has_documents: bool = False,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Execute the full agentic RAG pipeline with streaming.
    Yields SSE-compatible events:
      {"type": "token", "content": "..."}
      {"type": "status", "message": "..."}
      {"type": "citations", "citations": [...]}
      {"type": "trace", "trace": [...]}
    """
    logger.info(f"[pipeline] Starting RAG pipeline for query: {query[:100]}")
    document_names = document_names or []
    trace = []
    t0 = time.time()
    local_fast_mode = is_managed_local_vllm(llm_provider, openai_base_url)
    retrieve_top_k = 3 if local_fast_mode else 10
    rerank_top_k = 2 if local_fast_mode else 5
    if local_fast_mode:
        logger.info("[pipeline] Local fast mode enabled")

    # ── Step 1: Route ──
    t_route = time.time()
    logger.info("[pipeline] Step 1: Route query")
    try:
        if (document_names or has_documents) and not _looks_like_direct_chat(query):
            needs_retrieval = True
            logger.info("[pipeline] Route shortcut: documents available, forcing retrieval")
        else:
            needs_retrieval = await _route_query(
                query,
                groq_api_key=groq_api_key,
                llm_model=llm_model,
                llm_provider=llm_provider,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
            )
    except Exception as e:
        logger.error(f"[pipeline] Route failed: {e}")
        raise
    duration_route = round((time.time() - t_route) * 1000)
    trace.append({
        "step": "route",
        "detail": f"{'retrieval' if needs_retrieval else 'direct'}",
        "duration_ms": duration_route,
    })
    logger.info(f"[pipeline] Route complete", extra={"needs_retrieval": needs_retrieval, "duration_ms": duration_route})

    if not needs_retrieval:
        # Direct answer path
        trace.append({"step": "direct_answer", "detail": "No retrieval needed"})
        yield {"type": "trace", "trace": trace}
        yield {"type": "status", "message": "Generating answer..."}

        async for token in generate_direct_answer(
            query,
            groq_api_key=groq_api_key,
            llm_model=llm_model,
            llm_provider=llm_provider,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
        ):
            yield {"type": "token", "content": token}
        return

    # ── Step 2: Retrieve ──
    current_query = query
    rewrite_count = 0
    max_rewrites = 0 if local_fast_mode else 2
    relevant_docs = []

    logger.info(f"[pipeline] Step 2: Retrieve (max rewrites: {max_rewrites})")
    while rewrite_count <= max_rewrites:
        t_retrieve = time.time()
        logger.info(f"[pipeline] Retrieve attempt {rewrite_count + 1} for query: {current_query[:60]}")
        yield {"type": "status", "message": "Searching documents..."}
        try:
            if HYBRID_USE:
                documents = await search_hybrid(
                    current_query,
                    top_k=retrieve_top_k,
                    document_names=document_names,
                )
            else:
                documents = await search_documents(
                    current_query,
                    top_k=retrieve_top_k,
                    document_names=document_names,
                )
        except Exception as e:
            logger.error(f"[pipeline] Retrieve failed: {e}")
            raise
        trace.append({
            "step": "retrieve",
            "detail": f"Found {len(documents)} chunks for: '{current_query[:60]}...'" + (" (hybrid)" if HYBRID_USE else ""),
            "duration_ms": round((time.time() - t_retrieve) * 1000),
        })
        logger.info(f"[pipeline] Retrieve found {len(documents)} documents")

        if not documents:
            trace.append({"step": "rerank", "detail": "No documents found"})
            trace.append({"step": "grade", "detail": "No documents found"})
            break

        # ── Step 3: Rerank ──
        t_rerank = time.time()
        logger.info(f"[pipeline] Step 3: Rerank {len(documents)} documents")
        yield {"type": "status", "message": "Reranking the best chunks..."}
        try:
            documents = await rerank_documents(
                current_query,
                documents,
                top_k=rerank_top_k,
            )
        except Exception as e:
            logger.error(f"[pipeline] Rerank failed: {e}")
            pass
        trace.append({
            "step": "rerank",
            "detail": f"Reranked to top {rerank_top_k} from initial {retrieve_top_k}",
            "duration_ms": round((time.time() - t_rerank) * 1000),
        })
        logger.info(f"[pipeline] Rerank complete, {len(documents)} documents remain")

        if local_fast_mode:
            relevant_docs = documents[:rerank_top_k]
            trace.append({
                "step": "grade",
                "detail": "Skipped for fast local mode; using top reranked chunks directly",
                "duration_ms": 0,
            })
            logger.info(f"[pipeline] Local fast mode: using {len(relevant_docs)} documents without LLM grading")
            break

        # ── Step 4: Grade ──
        t_grade = time.time()
        logger.info(f"[pipeline] Step 4: Grade {len(documents)} documents")
        yield {"type": "status", "message": "Checking document relevance..."}
        try:
            relevant_docs, irrelevant = await grade_documents(
                current_query,
                documents,
                groq_api_key=groq_api_key,
                llm_model=llm_model,
                llm_provider=llm_provider,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
            )
        except Exception as e:
            logger.error(f"[pipeline] Grade failed: {e}")
            raise
        trace.append({
            "step": "grade",
            "detail": f"{len(relevant_docs)} relevant, {len(irrelevant)} irrelevant",
            "duration_ms": round((time.time() - t_grade) * 1000),
        })
        logger.info(f"[pipeline] Grade result: {len(relevant_docs)} relevant, {len(irrelevant)} irrelevant")

        if relevant_docs:
            break  # We have good results

        # ── Step 5: Rewrite (if needed) ──
        if rewrite_count < max_rewrites:
            t_rewrite = time.time()
            logger.info(f"[pipeline] Step 5: Rewrite query (attempt {rewrite_count + 1})")
            try:
                current_query = await rewrite_query(
                    current_query,
                    groq_api_key=groq_api_key,
                    llm_model=llm_model,
                    llm_provider=llm_provider,
                    openai_base_url=openai_base_url,
                    openai_api_key=openai_api_key,
                )
            except Exception as e:
                logger.error(f"[pipeline] Rewrite failed: {e}")
                raise
            rewrite_count += 1
            trace.append({
                "step": "query_rewrite",
                "detail": f"Rewritten to: '{current_query[:80]}'",
                "duration_ms": round((time.time() - t_rewrite) * 1000),
            })
            logger.info(f"[pipeline] Rewrite complete: {current_query[:80]}")
        else:
            break

    # Emit trace and final retrieval context before generation.
    yield {"type": "status", "message": "Preparing answer..."}
    yield {"type": "trace", "trace": trace}
    yield {"type": "documents", "documents": relevant_docs}

    # ── Step 6: Generate ──
    logger.info(f"[pipeline] Step 6: Generate answer with {len(relevant_docs)} documents")
    yield {
        "type": "status",
        "message": "Generating concise local answer..." if local_fast_mode else "Generating answer...",
    }
    full_answer = ""
    try:
        async for token in generate_answer(
            current_query,
            relevant_docs,
            groq_api_key=groq_api_key,
            llm_model=llm_model,
            llm_provider=llm_provider,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
        ):
            full_answer += token
            yield {"type": "token", "content": token}
    except Exception as e:
        logger.error(f"[pipeline] Generate failed: {repr(e)}")
        if not local_fast_mode:
            raise
        yield {"type": "status", "message": "Local model was slow; using retrieved document snippets..."}
        full_answer = build_extractive_fallback(relevant_docs, query=current_query)
        yield {"type": "token", "content": full_answer}
    logger.info(f"[pipeline] Generate complete, length: {len(full_answer)} chars")

    # ── Step 7: Citations ──
    logger.info("[pipeline] Step 7: Extract citations")
    try:
        citations = extract_citations(full_answer, relevant_docs)
    except Exception as e:
        logger.error(f"[pipeline] Citations extraction failed: {e}")
        citations = []
    if citations:
        yield {"type": "citations", "citations": citations}
    logger.info(f"[pipeline] Citations: {len(citations)} found")

    # Final trace update
    trace.append({
        "step": "complete",
        "detail": f"Total: {round((time.time() - t0) * 1000)}ms, {len(citations)} citations",
        "duration_ms": round((time.time() - t0) * 1000),
    })
    yield {"type": "trace", "trace": trace}
    logger.info(f"[pipeline] Pipeline complete")
