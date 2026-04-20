"""
Grounded answer generation with citations.
"""

import logging
import re
from typing import List, Dict, AsyncGenerator
from rag.llm import create_llm, is_managed_local_vllm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOCAL_MAX_DOCS = 2
LOCAL_CHUNK_CHAR_LIMIT = 1100
LOCAL_CONTEXT_CHAR_LIMIT = 2400
LOCAL_MAX_TOKENS = 256


class LocalGenerationError(RuntimeError):
    """Raised when local vLLM accepts a request but does not produce usable text."""


def _clean_text(value: str) -> str:
    return " ".join((value or "").split())


def _title_case_name(value: str) -> str:
    return " ".join(word.lower().capitalize() for word in value.strip().split())


def _is_identity_question(query: str) -> bool:
    normalized = query.strip().lower()
    return any(
        phrase in normalized
        for phrase in [
            "what is the name",
            "who is this person",
            "who is the person",
            "person name",
            "candidate name",
            "applicant name",
        ]
    )


def _extract_candidate_name(text: str) -> str:
    raw = text.replace("\n", " ")
    patterns = [
        r"\bName\s*:\s*([A-Z][A-Za-z'’.-]+(?:\s+(?:bin|binti|ibn|[A-Z][A-Za-z'’.-]+)){1,5})",
        r"\b([A-Z][A-Z'’.-]+(?:\s+(?:BIN|BINTI|IBN|[A-Z][A-Z'’.-]+)){1,5}?)(?=\s+(?:JUNIOR|FULL[- ]STACK|SOFTWARE|ENGINEER|DEVELOPER|PROGRAMMER)\b)",
        r"\bLinkedIn\s*:\s*([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" -|,")
            if "@" not in candidate and len(candidate.split()) >= 2:
                return _title_case_name(candidate)

    email_match = re.search(r"\b([a-z]+)([a-z]+)@[\w.-]+\.\w+\b", raw.lower())
    if email_match:
        return _title_case_name(f"{email_match.group(1)} {email_match.group(2)}")

    return ""


def answer_identity_question(query: str, documents: List[Dict]) -> str:
    if not _is_identity_question(query):
        return ""

    for i, doc in enumerate(documents[:LOCAL_MAX_DOCS]):
        name = _extract_candidate_name(str(doc.get("text", "")))
        if name:
            return f"Based on the retrieved document, this person is {name}. [Source {i + 1}]"

    return ""


def _summary_sentence(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    pieces = re.split(r"(?<=[.!?])\s+|(?:\s+-\s+)", cleaned)
    for piece in pieces:
        piece = piece.strip(" •-")
        if 40 <= len(piece) <= 220:
            return piece
    return cleaned[:180].rstrip()


def _find_source_for_pattern(documents: List[Dict], pattern: str) -> tuple[re.Match[str], int] | tuple[None, None]:
    for i, doc in enumerate(documents[:LOCAL_MAX_DOCS]):
        text = _clean_text(str(doc.get("text", "")))
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match, i + 1
    return None, None


def _extract_role_summary(documents: List[Dict]) -> str:
    for i, doc in enumerate(documents[:LOCAL_MAX_DOCS]):
        text = _clean_text(str(doc.get("text", "")))
        name = _extract_candidate_name(text)
        title_match = re.search(
            r"\b(?:JUNIOR\s+)?(?:FULL[- ]STACK\s+)?(?:SOFTWARE\s+)?(?:ENGINEER|DEVELOPER|PROGRAMMER)\b",
            text,
            flags=re.IGNORECASE,
        )
        if name and title_match:
            title = _title_case_name(title_match.group(0).replace("-", " "))
            return f"- {name} is a {title}. [Source {i + 1}]"
    return ""


def _extract_resume_summary_bullets(documents: List[Dict]) -> List[str]:
    bullets = []

    role = _extract_role_summary(documents)
    if role:
        bullets.append(role)

    objective_match, objective_source = _find_source_for_pattern(
        documents,
        r"speciali[sz]ing in ([^.]+)",
    )
    if objective_match and objective_source:
        focus = objective_match.group(1).strip(" .")
        bullets.append(f"- Technical focus includes {focus}. [Source {objective_source}]")

    education_match, education_source = _find_source_for_pattern(
        documents,
        r"(Diploma in Computer Science[^-.]*(?:GPA\s*:\s*[0-9.]+\s*/\s*[0-9.]+)?)",
    )
    if education_match and education_source:
        education = _clean_text(education_match.group(1)).strip(" ,.-")
        bullets.append(f"- Education: {education}. [Source {education_source}]")

    achievement_match, achievement_source = _find_source_for_pattern(
        documents,
        r"(Vice Chancellor Award[^-.]*(?:First Class[^-.]*)?)",
    )
    if achievement_match and achievement_source:
        achievement = _clean_text(achievement_match.group(1)).strip(" ,.-")
        bullets.append(f"- Achievement: {achievement}. [Source {achievement_source}]")

    experience_match, experience_source = _find_source_for_pattern(
        documents,
        r"((?:Packaged App Development|Application Development|Software|Data|AI)[^.]{10,160})",
    )
    if experience_match and experience_source:
        experience = _clean_text(experience_match.group(1)).strip(" ,.-")
        if not any(experience.lower() in bullet.lower() for bullet in bullets):
            bullets.append(f"- Experience includes {experience}. [Source {experience_source}]")

    return bullets[:5]


def _build_context(documents: List[Dict]) -> str:
    """Build context string from retrieved documents."""
    parts = []
    for i, doc in enumerate(documents):
        parts.append(
            f"[Source {i + 1}: {doc['document_name']}]\n{doc['text']}"
        )
    return "\n\n---\n\n".join(parts)


def build_local_context(documents: List[Dict]) -> str:
    """Build a small context for local 2B-class models."""
    parts = []
    total_chars = 0
    for i, doc in enumerate(documents[:LOCAL_MAX_DOCS]):
        text = _clean_text(str(doc.get("text", "")))[:LOCAL_CHUNK_CHAR_LIMIT]
        part = f"[Source {i + 1}: {doc.get('document_name', 'unknown')}]\n{text}"
        if total_chars + len(part) > LOCAL_CONTEXT_CHAR_LIMIT:
            remaining = max(0, LOCAL_CONTEXT_CHAR_LIMIT - total_chars)
            if remaining < 120:
                break
            part = part[:remaining]
        parts.append(part)
        total_chars += len(part)
    return "\n\n---\n\n".join(parts)


def build_extractive_fallback(documents: List[Dict], query: str = "") -> str:
    """Return a cited fallback answer when local generation is too slow."""
    if not documents:
        return "I could not find enough matching document text to answer that from the uploaded files."

    identity_answer = answer_identity_question(query, documents)
    if identity_answer:
        return identity_answer

    intro = "Based on the retrieved document, here are the key points I found:\n\n"
    bullets = _extract_resume_summary_bullets(documents)
    if bullets:
        return intro + "\n".join(bullets)

    bullets = []
    for i, doc in enumerate(documents[:LOCAL_MAX_DOCS]):
        summary = _summary_sentence(str(doc.get("text", "")))
        if summary:
            bullets.append(f"- {summary} [Source {i + 1}]")

    if not bullets:
        return "I found related document chunks, but they did not contain enough readable text to summarize confidently."
    return intro + "\n".join(bullets[:5])


async def generate_answer(
    query: str,
    documents: List[Dict],
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate a grounded answer with streaming."""
    logger.info(f"[generate] Starting answer generation, documents: {len(documents)}")
    if not documents:
        logger.warning("[generate] No documents provided, generating fallback response")

    local_fast_mode = is_managed_local_vllm(llm_provider, openai_base_url)
    llm = create_llm(
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        llm_provider=llm_provider,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        temperature=0.1 if local_fast_mode else 0.4,
        max_tokens=LOCAL_MAX_TOKENS if local_fast_mode else 2048,
    )

    if not documents:
        prompt = f"""You are DocPilot, an AI document assistant. The user asked a question
but no relevant documents were found in the knowledge base.

Respond helpfully, noting that no documents were found that match the query.
Suggest the user upload relevant documents.

Question: {query}

Response:"""
    else:
        context = build_local_context(documents) if local_fast_mode else _build_context(documents)
        if local_fast_mode:
            prompt = f"""Answer using ONLY the sources below.
Write 3-5 short bullets. Cite each bullet with [Source N]. If the sources are insufficient, say so briefly.

Sources:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""You are DocPilot, an AI document assistant. Answer the user's question
based ONLY on the provided document sources. Follow these rules:

1. Use information ONLY from the provided sources
2. Reference sources using [Source N] notation
3. If the sources don't contain enough information, say so
4. Be comprehensive but concise
5. Use markdown formatting for readability
{"6. Keep the answer under about 500 words for local mode" if local_fast_mode else ""}

Sources:
{context}

Question: {query}

Answer:"""

    # Stream tokens
    try:
        emitted = False
        async for chunk in llm.astream(prompt):
            if chunk.content:
                emitted = True
                yield chunk.content
        if local_fast_mode and not emitted:
            raise LocalGenerationError("Local model did not produce tokens fast enough.")
        logger.info("[generate] Answer generation complete")
    except Exception as e:
        logger.error(f"[generate] LLM streaming failed: {repr(e)}")
        raise


async def generate_direct_answer(
    query: str,
    groq_api_key: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate a direct answer without retrieval (for non-retrieval queries)."""
    local_fast_mode = is_managed_local_vllm(llm_provider, openai_base_url)
    llm = create_llm(
        groq_api_key=groq_api_key,
        llm_model=llm_model,
        llm_provider=llm_provider,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        temperature=0.1 if local_fast_mode else 0.4,
        max_tokens=LOCAL_MAX_TOKENS if local_fast_mode else 2048,
    )

    prompt = f"""You are DocPilot, an AI document assistant. The user asked a general question
that doesn't require document retrieval.

Answer the question helpfully and concisely. If the question would benefit from
uploaded documents, mention that.

Question: {query}

Answer:"""

    async for chunk in llm.astream(prompt):
        if chunk.content:
            yield chunk.content
