# DocPilot — Agentic RAG Document Assistant

> Upload documents and get intelligent, citation-backed answers powered by an Agentic RAG pipeline with hybrid retrieval, relevance grading, and query rewriting.

## Architecture

```
[ Next.js 16 + Bun ]  →  [ FastAPI + LangGraph ]  →  [ Qdrant Cloud ]
     Frontend                 AI Service               Vector DB
```

## Quick Start

### 1. Frontend (Next.js)

```bash
cd apps/web
bun install
bun run dev
# → http://localhost:3000
```

### 2. AI Service (FastAPI)

```bash
cd services/ai
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# → http://localhost:8000
```

### 3. Environment

Create `services/ai/.env`:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=docpilot_docs
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile
```

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| Frontend | Next.js 16, TypeScript, Tailwind CSS, Framer Motion |
| Runtime | Bun |
| Backend | FastAPI, LangGraph, LangChain |
| LLM | Groq (LLaMA 3.3 70B) |
| Vector DB | Qdrant Cloud |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |

## RAG Pipeline

1. **Route** — Decide: retrieval needed or direct answer
2. **Retrieve** — Dense vector search via Qdrant
3. **Grade** — LLM judges document relevance
4. **Rewrite** — Auto-reformulate weak queries (up to 2x)
5. **Generate** — Grounded answer with source references
6. **Cite** — Extract and format citation chips

## Features

- 📄 PDF, TXT, MD, DOCX upload with drag-and-drop
- 💬 Streaming chat with token-by-token rendering
- 📎 Expandable citation chips with source scores
- 🔍 Reasoning trace (route, retrieve, grade, rewrite steps)
- 📊 Evaluation dashboard with quality metrics
- ⚙️ Settings page for API key management
- 🌙 Dark mode with Linear/Vercel aesthetic
