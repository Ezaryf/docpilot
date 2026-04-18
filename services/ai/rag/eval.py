"""
RAG Evaluation using Ragas metrics.
Provides: faithfulness, answer_relevancy, context_recall, context_precision
"""

import os
import logging
from typing import List, Dict, Any, Optional

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from ragas import RunConfig
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

_metrics_initialized = False
_faithfulness = None
_answer_relevancy = None
_context_recall = None
_context_precision = None


def _init_metrics(
    groq_api_key: str | None = None,
    llm_model: str | None = None,
) -> None:
    """Initialize Ragas metrics with LLM configuration."""
    global _metrics_initialized, _faithfulness, _answer_relevancy, _context_recall, _context_precision

    if _metrics_initialized:
        return

    api_key = groq_api_key or DEFAULT_GROQ_API_KEY
    model = llm_model or DEFAULT_LLM_MODEL

    logger.info(f"[eval] Initializing Ragas metrics with model: {model}")

    _faithfulness = Faithfulness(
        llm=api_key,
        model=model,
    )
    _answer_relevancy = AnswerRelevancy(
        llm=api_key,
        model=model,
    )
    _context_recall = ContextRecall(
        llm=api_key,
        model=model,
    )
    _context_precision = ContextPrecision(
        llm=api_key,
        model=model,
    )

    _metrics_initialized = True
    logger.info("[eval] Ragas metrics initialized")


async def evaluate_rag(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str | None = None,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
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

    _init_metrics(groq_api_key, llm_model)

    eval_data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    if ground_truth:
        eval_data["ground_truth"] = [ground_truth]

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
        results = eval_result.to_dict()
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
    )