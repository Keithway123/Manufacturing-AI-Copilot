# Manufacturing AI Copilot

Manufacturing AI Copilot is a manufacturing-domain AI knowledge assistant built with FastAPI and RAG.

## Current Scope

- Local Markdown document ingestion
- LlamaIndex local persisted index
- FastAPI `/search` endpoint for retrieval debugging
- FastAPI `/chat` endpoint for RAG answer generation
- Source metadata returned with answers
- Retrieval evaluation script
- Unit and API tests

## Tech Stack

- Python 3.12
- uv
- FastAPI
- LlamaIndex
- DashScope `text-embedding-v3`
- Qwen
- pytest

## Project Structure

```text
src/manufacturing_ai_copilot/  Application source code
scripts/                       CLI utilities
data/raw/                      Raw Markdown input documents
data/eval/                     Retrieval evaluation questions
storage/                       Local generated LlamaIndex index
tests/                         Unit and API tests
docs/                          Project docs and learning records
notes/                         Personal study notes
```

## Setup

```powershell
uv sync
```

Create `.env`:

```env
DASHSCOPE_API_KEY=your_api_key
```

## Build Index

```powershell
uv run --env-file .env python -m scripts.build_index
```

## Run API

```powershell
uv run --env-file .env uvicorn manufacturing_ai_copilot.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service health check |
| POST | `/search` | Retrieve raw matching chunks |
| POST | `/chat` | Generate RAG answer with sources |

## Example Chat Request

```json
{
  "question": "贴片机报警 E203 怎么处理？",
  "top_k": 3,
  "department": "生产部"
}
```

## Test

```powershell
uv run pytest
```

## Retrieval Evaluation

```powershell
uv run --env-file .env python -m scripts.evaluate_retrieval
```

## Roadmap

- Replace local storage with vector database
- Add LangGraph multi-agent workflow
- Add Dify chat/workflow entry
- Add Docker Compose deployment
