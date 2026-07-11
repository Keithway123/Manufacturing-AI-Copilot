import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from manufacturing_ai_copilot.core.config import (
    DEFAULT_RETRIEVAL_TOP_K,
    MAX_RETRIEVAL_TOP_K,
    SERVICE_NAME,
    STORAGE_DIR,
    VERSION,
)
from manufacturing_ai_copilot.rag.query_engine import retrieve_matches
from manufacturing_ai_copilot.agent.graph import run_agent

logger = logging.getLogger(__name__)

app = FastAPI(title=SERVICE_NAME, version=VERSION)


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    department: str | None = Field(default=None, min_length=1)

    top_k: int = Field(
        default=DEFAULT_RETRIEVAL_TOP_K,
        ge=1,
        le=MAX_RETRIEVAL_TOP_K,
    )


class SearchMatch(BaseModel):
    score: float | None
    document: str | None
    title: str | None
    # 被检索命中的 chunk 正文
    content: str
    # 原始 metadata，方便后续做权限、部门、文档类型过滤
    metadata: dict


class SearchResponse(BaseModel):
    question: str
    top_k: int
    matches: list[SearchMatch]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    department: str | None = Field(default=None, min_length=1)
    top_k: int = Field(
        default=DEFAULT_RETRIEVAL_TOP_K,
        ge=1,
        le=MAX_RETRIEVAL_TOP_K,
    )


class ChatSource(BaseModel):
    doc_id: str
    title: str | None
    document: str | None
    department: str | None
    version: str | None
    score: float | None


class ChatRetrievalInfo(BaseModel):
    top_k: int
    min_score: float
    retrieved_count: int
    used_count: int


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChatSource]
    retrieval: ChatRetrievalInfo


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": VERSION,
    }


@app.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    try:
        # API 层只负责接收请求和返回响应，RAG 细节放在 query_engine.py
        matches = retrieve_matches(
            storage_dir=STORAGE_DIR,
            question=request.question,
            similarity_top_k=request.top_k,
            department=request.department,
        )
    except FileNotFoundError as exc:
        # storage 不存在或索引没构建时，说明服务暂时不可用
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("RAG search failed")
        raise HTTPException(
            status_code=500,
            detail="RAG search failed. Check server logs.",
        ) from exc

    return SearchResponse(
        question=request.question,
        top_k=request.top_k,
        matches=matches,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = run_agent(
            storage_dir=STORAGE_DIR,
            question=request.question,
            top_k=request.top_k,
            department=request.department,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("RAG chat failed")
        raise HTTPException(
            status_code=500,
            detail="RAG chat failed. Check server logs.",
        ) from exc

    return ChatResponse(**result)
