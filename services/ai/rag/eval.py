import os
import logging
import json
import asyncio
import time
import math
from typing import List, Dict, Any, Optional

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

_metrics_initialized = False
_metrics_config_key = None
_faithfulness = None
_answer_relevancy = None
_context_recall = None
_context_precision = None


def _clean_metric_value(value: Any) -> float:
    """Return a JSON-safe numeric metric value."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0

    if math.isnan(numeric) or math.isinf(numeric):
        return 0.0
    return numeric


def serialize_evaluation_result(eval_result: Any) -> Dict[str, List[float]]:
    """
    Normalize Ragas EvaluationResult objects across versions.

    Ragas 0.4.x exposes scores as a list of per-row dicts instead of to_dict().
    The dashboard expects a dict of metric names to score arrays.
    """
    scores = getattr(eval_result, "scores", None)
    if isinstance(scores, list):
        normalized: Dict[str, List[float]] = {}
        for row in scores:
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                normalized.setdefault(key, []).append(_clean_metric_value(value))
        if normalized:
            return normalized

    scores_dict = getattr(eval_result, "_scores_dict", None)
    if isinstance(scores_dict, dict):
        return {
            key: [_clean_metric_value(value) for value in values]
            for key, values in scores_dict.items()
            if isinstance(values, list)
        }

    to_dict = getattr(eval_result, "to_dict", None)
    if callable(to_dict):
        raw = to_dict()
        if isinstance(raw, dict):
            return {
                key: [_clean_metric_value(value) for value in values]
                for key, values in raw.items()
                if isinstance(values, list)
            }

    return {}


def _init_metrics(
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> None:
    """Initialize Ragas metrics with LLM configuration."""
    global _metrics_initialized, _metrics_config_key, _faithfulness, _answer_relevancy, _context_recall, _context_precision

    config_key = (
        llm_provider or "groq",
        llm_model or DEFAULT_LLM_MODEL,
        groq_api_key or "",
        openai_base_url or "",
        openai_api_key or "",
    )

    if _metrics_initialized and _metrics_config_key == config_key:
        return

    from rag.llm import create_llm
    from rag.retrieve import EMBED_MODEL
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    api_key = groq_api_key or DEFAULT_GROQ_API_KEY
    model = llm_model or DEFAULT_LLM_MODEL

    logger.info(f"[eval] Initializing Ragas metrics with provider={llm_provider or 'groq'}, model={model}")

    # Wrap Langchain models for Ragas 0.4.3+
    chat_llm = create_llm(
        groq_api_key=api_key,
        llm_model=model,
        llm_provider=llm_provider,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        temperature=0.0,
        max_tokens=2048,
    )
    hf_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    ragas_llm = LangchainLLMWrapper(chat_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    _faithfulness = Faithfulness(
        llm=ragas_llm,
    )
    _answer_relevancy = AnswerRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    _context_recall = ContextRecall(
        llm=ragas_llm,
    )
    _context_precision = ContextPrecision(
        llm=ragas_llm,
    )

    _metrics_initialized = True
    _metrics_config_key = config_key
    logger.info("[eval] Ragas metrics initialized")


async def evaluate_rag(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str | None = None,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> Dict[str, Any]:
    """
    Evaluate RAG pipeline outputs using Ragas metrics.

    Args:
        question: User question
        answer: Generated answer
        contexts: Retrieved context passages
        ground_truth: Optional ground truth answer
        groq_api_key: Groq API key
        llm_model: LLM model name

    Returns:
        Dictionary with metric scores
    """
    logger.info(f"[eval] Evaluating question: {question[:60]}...")

    _init_metrics(groq_api_key, llm_model, llm_provider, openai_base_url, openai_api_key)

    eval_data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    if ground_truth:
        # Ragas 0.4.3 uses 'reference' instead of 'ground_truth'
        eval_data["reference"] = [ground_truth]

    dataset = Dataset.from_dict(eval_data)

    metrics = [
        _faithfulness,
        _answer_relevancy,
    ]

    if ground_truth:
        metrics.append(_context_recall)
        metrics.append(_context_precision)

    logger.info("[eval] Running evaluation...")
    try:
        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )
        results = serialize_evaluation_result(eval_result)
        logger.info(f"[eval] Evaluation complete: {results}")
        return results
    except Exception as e:
        logger.error(f"[eval] Evaluation failed: {e}")
        raise


async def evaluate_from_response(
    question: str,
    llm_answer: str,
    retrieved_docs: List[Dict[str, Any]],
    ground_truth: str | None = None,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> Dict[str, Any]:
    """
    Evaluate from existing RAG response.

    Args:
        question: User question
        llm_answer: Generated answer from LLM
        retrieved_docs: List of retrieved documents with 'text' key
        ground_truth: Optional ground truth answer
        groq_api_key: Groq API key
        llm_model: LLM model name

    Returns:
        Dictionary with metric scores
    """
    contexts = [doc.get("text", "") for doc in retrieved_docs if doc.get("text")]

    return await evaluate_rag(
        question=question,
        answer=llm_answer,
        contexts=contexts,
        ground_truth=ground_truth,
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        llm_provider=llm_provider,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )

async def generate_test_questions(
    texts: List[str],
    num_questions: int = 5,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> List[str]:
    """Generate potential user questions from context chunks for evaluation."""
    from rag.llm import create_llm
    
    llm = create_llm(
        groq_api_key=groq_api_key or DEFAULT_GROQ_API_KEY, 
        llm_model=llm_model or DEFAULT_LLM_MODEL,
        llm_provider=llm_provider,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        temperature=0.2,
        max_tokens=2048,
    )
    
    combined_context = "\n---\n".join(texts[:3]) # Limit context for prompt
    prompt = f"""You are an objective AI evaluator.
Given the following document excerpts, generate {num_questions} diverse and specific questions that a user might ask who wants to learn from these documents.
Return ONLY a JSON list of strings.

Excerpts:
{combined_context}

Output format: ["question 1", "question 2", ...]"""
    
    try:
        response = await llm.ainvoke(prompt)
        content = response.content.strip()
        # Extract JSON list if LLM adds markdown
        if "[" in content and "]" in content:
            content = content[content.find("["):content.rfind("]")+1]
        questions = json.loads(content)
        return list(set(questions))[:num_questions]
    except Exception as e:
        logger.error(f"[eval] Question generation failed: {e}")
        return ["What is discussed in these documents?"]

async def run_batch_evaluation(
    num_samples: int = 3,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> Dict[str, Any]:
    """Run a full evaluation suite against indexed documents."""
    from rag.retrieve import _get_qdrant, COLLECTION
    client = _get_qdrant()
    
    logger.info(f"[eval] Starting batch evaluation (samples={num_samples})")
    
    # 1. Get sample data
    try:
        records, _ = client.scroll(collection_name=COLLECTION, limit=50, with_payload=True)
        if not records:
            return {"error": "No documents found to evaluate"}
        
        texts = [r.payload.get("text", "") for r in records if r.payload][:10]
    except Exception as e:
        logger.error(f"[eval] Failed to scroll Qdrant: {e}")
        return {"error": "Failed to retrieve samples"}

    # 2. Generate questions
    questions = await generate_test_questions(
        texts,
        num_questions=num_samples,
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        llm_provider=llm_provider,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )
    
    results = []
    total_hit_at_5 = 0
    total_latency = 0
    total_citations = 0
    total_groundedness = 0
    
    # 3. Process each question through the pipeline
    from rag.graph import run_rag_pipeline
    
    for q in questions:
        t_start = time.time()
        try:
            # We use the actual pipeline to get the trace and results
            # run_rag_pipeline returns an async generator yielding dicts
            output = {"answer": "", "documents": [], "trace": [], "citations": []}
            async for chunk in run_rag_pipeline(
                q,
                session_id="eval-batch",
                has_documents=True,
                groq_api_key=groq_api_key,
                llm_model=llm_model,
                llm_provider=llm_provider,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
            ):
                if chunk["type"] == "token":
                    output["answer"] += chunk["content"]
                elif chunk["type"] == "documents":
                    output["documents"] = chunk["documents"]
                elif chunk["type"] == "trace":
                    output["trace"] = chunk["trace"]
                elif chunk["type"] == "citations":
                    output["citations"] = chunk["citations"]
            
            latency = (time.time() - t_start) * 1000
            
            # Evaluate this specific run
            eval_scores = await evaluate_from_response(
                question=q,
                llm_answer=output["answer"],
                retrieved_docs=output["documents"],
                groq_api_key=groq_api_key,
                llm_model=llm_model,
                llm_provider=llm_provider,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
            )
            
            # Score mappings
            faithfulness = eval_scores.get("faithfulness", [0])[0]
            relevancy = eval_scores.get("answer_relevancy", [0])[0]
            
            # Aggregates
            results.append({
                "query": q,
                "relevant": relevancy > 0.7,
                "rewritten": any(t["step"] == "query_rewrite" for t in output["trace"]),
                "latency_ms": latency,
                "citations": len(output.get("citations", [])),
                "score": (faithfulness + relevancy) / 2
            })
            
            total_hit_at_5 += 1 if len(output["documents"]) > 0 else 0
            total_latency += latency
            total_citations += 1 if output.get("citations") else 0
            total_groundedness += faithfulness
            
        except Exception as e:
            logger.error(f"[eval] Failed evaluating question '{q}': {e}")

    # 4. Final Aggregated Metrics
    count = max(1, len(results))
    metrics = {
        "hit_at_5": total_hit_at_5 / count,
        "avg_latency_ms": total_latency / count,
        "citation_coverage": total_citations / count,
        "groundedness": total_groundedness / count
    }
    
    return {
        "results": results,
        "metrics": metrics
    }
