# Manufacturing AI Copilot

Manufacturing AI Copilot is a manufacturing-domain AI assistant built with FastAPI, RAG, LangGraph, and Dify.

The current version supports knowledge-base Q&A, tool-style work order status lookup, fallback handling, and Docker Compose based local deployment.

## Current Capabilities

- FastAPI backend with `/health`, `/search`, and `/chat`
- LlamaIndex based local RAG over manufacturing Markdown documents
- Persisted local index under `storage/`
- Qwen answer generation through DashScope
- LangGraph Agent orchestration
- RAG / Tool / Fallback branches
- Answer review and human review stub
- Dify Chatflow as visual demo entrance
- Dockerfile and Docker Compose for local backend deployment
- Unit and API tests with pytest

## Tech Stack

- Python 3.12
- uv
- FastAPI
- LlamaIndex
- LangGraph
- DashScope / Qwen
- Dify
- Docker / Docker Compose
- pytest

## Project Structure

```text
src/manufacturing_ai_copilot/  Application source code
scripts/                       CLI utilities
data/raw/                      Raw Markdown input documents
data/eval/                     Retrieval evaluation questions
storage/                       Local generated LlamaIndex index
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
```

Do not commit `.env`.

## Build RAG Index

`storage/` is generated from documents under `data/raw/`.

```powershell
uv run --env-file .env python -m scripts.build_index
```

Rebuild the index when raw documents, chunking settings, or embedding model changes.

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

Recommended V5 local deployment mode:

```powershell
docker compose up --build
```

Open:

```text
http://127.0.0.1:8001/docs
http://127.0.0.1:8001/health
```

The Compose service mounts local `storage/` into the container, so build the RAG index before using `/chat`.

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service health check |
| POST | `/search` | Retrieve raw matching chunks for debugging |
| POST | `/chat` | Run Agent workflow and return final answer |

## Example Chat Request

```json
{
  "question": "贴片机报警 E203 怎么处理？",
  "top_k": 3,
  "department": "生产部"
}
```

Minimal request:

```json
{
  "question": "贴片机报警 E203 怎么处理？",
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

```powershell
uv run --env-file .env python -m scripts.evaluate_retrieval
```

## Roadmap

- Add database-backed business data
- Replace work order stub with real query tool
- Upgrade local LlamaIndex storage to Qdrant or pgvector
- Add production-oriented health checks
- Improve deployment documentation
