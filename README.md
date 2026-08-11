# Manufacturing AI Copilot

Manufacturing AI Copilot is a manufacturing-domain AI assistant built with FastAPI, LangGraph, Qdrant, PostgreSQL, and Dify.

The current version supports knowledge-base Q&A, Qdrant-based RAG retrieval, PostgreSQL-backed work order lookup, fallback handling, and Docker Compose based local deployment.

## Current Capabilities

- FastAPI backend with `/health`, `/search`, and `/chat`
- Qdrant-based RAG retrieval over manufacturing Markdown documents
- LlamaIndex document parsing, chunking, embedding, and Retriever integration
- Qwen answer generation through DashScope
- LangGraph Agent orchestration
- RAG / Tool / Fallback branches
- PostgreSQL-backed work order status lookup
- Answer review and human review stub
- Dify Chatflow as visual demo entrance
- Dockerfile and Docker Compose for local backend, PostgreSQL, and Qdrant
- Retrieval evaluation script with Qdrant as default backend
- Unit and API tests with pytest

## Tech Stack

- Python 3.12
- uv
- FastAPI
- LlamaIndex
- LangGraph
- DashScope / Qwen
- Qdrant
- PostgreSQL
- Dify
- Docker / Docker Compose
- pytest

## Project Structure

```text
src/manufacturing_ai_copilot/  Application source code
scripts/                       CLI utilities
data/raw/                      Raw Markdown input documents
data/eval/                     Retrieval evaluation questions
storage/                       Legacy local LlamaIndex index for comparison
tests/                         Unit and API tests
docs/                          Local project docs and progress records
notes/                         Local Obsidian study notes
```

## Environment Variables

Copy `.env.example` to `.env`, then fill local values.

Required variables:

```env
DASHSCOPE_API_KEY=replace-with-your-dashscope-api-key
LLM_MODEL=qwen3.7-plus
EMBEDDING_MODEL=qwen3.7-text-embedding
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DATABASE_URL=sqlite:///data/manufacturing.db
POSTGRES_DB=manufacturing
POSTGRES_USER=manufacturing
POSTGRES_PASSWORD=replace-with-local-password
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION_NAME=manufacturing_knowledge
```

Do not commit `.env`.

## Build RAG Index

The current main RAG index is stored in Qdrant.

Start Qdrant first:

```powershell
docker compose up -d qdrant
```

When running with Docker Compose, Qdrant is accessed through the internal service URL configured in `docker-compose.yml`.

Then build the index from documents under `data/raw/`:

```powershell
uv run --env-file .env python -m scripts.build_index
```

Rebuild the Qdrant collection when raw documents, chunking settings, embedding model, vector size, or distance metric changes.

`storage/` is kept only as a legacy local index path for comparison and fallback during the learning stage.

## Run Locally with uv

```powershell
uv run --env-file .env uvicorn manufacturing_ai_copilot.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

## Run with Docker Compose

Recommended local deployment mode:

```powershell
docker compose up --build
```

Open:

```text
http://127.0.0.1:8001/docs
http://127.0.0.1:8001/health
```

Compose starts:

- FastAPI backend
- PostgreSQL for structured work order data
- Qdrant for vector retrieval

Before using RAG `/chat`, make sure the Qdrant collection has been built with `scripts.build_index`.

Initialize database seed data before testing Tool questions:

```powershell
docker compose run --rm manufacturing-api python -m scripts.init_database
```

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service health check |
| POST | `/search` | Retrieve matching chunks for debugging |
| POST | `/chat` | Run Agent workflow and return final answer |

## Example Chat Request

Knowledge-base question:

```json
{
  "question": "贴片机报警 E203 怎么处理？",
  "top_k": 3,
  "department": "生产部"
}
```

Tool question:

```json
{
  "question": "查询 WO-20260727-001 工单状态",
  "top_k": 3
}
```

Fallback question:

```json
{
  "question": "今天天气怎么样？",
  "top_k": 3
}
```

## Dify Integration

Dify is used as a visual demo entrance. Core RAG, Tool, Fallback, and LLM logic stays in FastAPI.

When Dify runs in Docker, configure its HTTP Request node with:

```text
POST http://host.docker.internal:8001/chat
```

Request body:

```json
{
  "question": "{{user_input}}",
  "top_k": 3
}
```

Dify should call `/chat`, not `/search`.

## Tests

```powershell
uv run pytest
```

## Retrieval Evaluation

Qdrant backend:

```powershell
uv run --env-file .env python -m scripts.evaluate_retrieval
```

Legacy local storage backend:

```powershell
uv run --env-file .env python -m scripts.evaluate_retrieval --backend storage
```

## Current Boundaries

- Question classification still uses keyword rules.
- Tool branch currently supports only work order status lookup.
- Tool + RAG combined workflow is not implemented yet.
- Rerank, hybrid retrieval, incremental indexing, and production health checks are not implemented yet.
- `storage/` legacy retrieval path is still kept for comparison and rollback.

## Roadmap

- Add LLM-based intent classification
- Add Tool + RAG combined workflow
- Improve retrieval quality with rerank or hybrid retrieval
- Add index health check and incremental indexing
- Improve production deployment documentation
