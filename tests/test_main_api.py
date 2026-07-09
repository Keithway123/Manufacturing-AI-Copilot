from fastapi.testclient import TestClient
from manufacturing_ai_copilot.core.config import (
    MIN_RETRIEVAL_SCORE,
    SERVICE_NAME,
    VERSION,
)
import manufacturing_ai_copilot.main as main

client = TestClient(main.app)


def test_health_check_returns_service_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": VERSION,
    }


def test_chat_returns_response_from_rag_layer(monkeypatch):
    def fake_chat_with_retrieval(storage_dir, question: str, similarity_top_k: int):
        return {
            "question": question,
            "answer": "fake answer",
            "sources": [
                {
                    "title": "SMT设备报警处理SOP",
                    "document": "SMT设备报警处理SOP.md",
                    "score": 0.9,
                }
            ],
            "retrieval": {
                "top_k": similarity_top_k,
                "min_score": MIN_RETRIEVAL_SCORE,
                "retrieved_count": 2,
                "used_count": 1,
            },
        }

    monkeypatch.setattr(main, "chat_with_retrieval", fake_chat_with_retrieval)

    response = client.post(
        "/chat",
        json={
            "question": "贴片机报警 E203 怎么处理？",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "贴片机报警 E203 怎么处理？",
        "answer": "fake answer",
        "sources": [
            {
                "title": "SMT设备报警处理SOP",
                "document": "SMT设备报警处理SOP.md",
                "score": 0.9,
            }
        ],
        "retrieval": {
            "top_k": 3,
            "min_score": MIN_RETRIEVAL_SCORE,
            "retrieved_count": 2,
            "used_count": 1,
        },
    }


def test_chat_returns_503_when_rag_index_is_missing(monkeypatch):
    def fake_chat_with_retrieval(storage_dir, question: str, similarity_top_k: int):
        raise FileNotFoundError("RAG index not found")

    monkeypatch.setattr(main, "chat_with_retrieval", fake_chat_with_retrieval)

    response = client.post(
        "/chat",
        json={
            "question": "贴片机报警 E203 怎么处理？",
            "top_k": 3,
        },
    )

    assert response.status_code == 503
    assert "RAG index not found" in response.json()["detail"]


def test_chat_returns_500_when_rag_layer_fails(monkeypatch):
    def fake_chat_with_retrieval(storage_dir, question: str, similarity_top_k: int):
        raise RuntimeError("unexpected error")

    monkeypatch.setattr(main, "chat_with_retrieval", fake_chat_with_retrieval)

    response = client.post(
        "/chat",
        json={
            "question": "贴片机报警 E203 怎么处理？",
            "top_k": 3,
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "RAG chat failed. Check server logs.",
    }


def test_chat_rejects_empty_question():
    response = client.post(
        "/chat",
        json={
            "question": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422
