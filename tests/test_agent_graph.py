# 只测 Agent Graph 是否把 State 正确传给 RAG 层，并把结果写回 State
from pathlib import Path
from manufacturing_ai_copilot.agent import graph
from manufacturing_ai_copilot.core.config import NO_ANSWER_MESSAGE
from manufacturing_ai_copilot.agent import classifier


def test_run_agent_calls_rag_node_and_returns_final_state(monkeypatch):
    def fake_chat_with_retrieval(
        storage_dir: Path,
        question: str,
        similarity_top_k: int,
        department: str | None,
        domain_type: str,
    ) -> dict:
        assert storage_dir == Path("fake-storage")
        assert question == "贴片机报警 E203 怎么处理？"
        assert similarity_top_k == 3
        assert department == "生产部"
        assert domain_type == classifier.EQUIPMENT_SOP

        return {
            "question": question,
            "answer": "fake answer",
            "sources": [
                {
                    "doc_id": "smt_alarm_sop",
                    "title": "SMT设备报警处理SOP",
                    "document": "SMT设备报警处理SOP.md",
                    "department": "生产部",
                    "version": "v1.0",
                    "score": 0.9,
                }
            ],
            "retrieval": {
                "top_k": 3,
                "min_score": 0.6,
                "retrieved_count": 3,
                "used_count": 1,
            },
        }

    monkeypatch.setattr(
        graph,
        "chat_with_retrieval",
        fake_chat_with_retrieval,
    )

    result = graph.run_agent(
        question="贴片机报警 E203 怎么处理？",
        storage_dir=Path("fake-storage"),
        top_k=3,
        department="生产部",
    )

    assert result["answer"] == "fake answer"
    assert result["sources"][0]["doc_id"] == "smt_alarm_sop"
    assert result["retrieval"]["used_count"] == 1
    assert result["question_type"] == classifier.KNOWLEDGE_QA
    assert result["route"] == classifier.ROUTE_RAG_ANSWER
    assert result["answer_review"]["answer_quality"] == graph.ANSWER_QUALITY_GROUNDED
    assert result["answer_review"]["needs_review"] is False
    assert result["answer_review"]["has_sources"] is True
    assert result["answer_review"]["used_count"] == 1


def test_run_agent_marks_answer_as_weak_when_sources_are_missing(monkeypatch):
    def fake_chat_with_retrieval(
        storage_dir: Path,
        question: str,
        similarity_top_k: int,
        department: str | None,
        domain_type: str,
    ) -> dict:
        return {
            "question": question,
            "answer": "fake weak answer",
            "sources": [],
            "retrieval": {
                "top_k": similarity_top_k,
                "min_score": 0.6,
                "retrieved_count": 1,
                "used_count": 0,
            },
        }

    monkeypatch.setattr(
        graph,
        "chat_with_retrieval",
        fake_chat_with_retrieval,
    )

    result = graph.run_agent(
        question="贴片机报警 E203 怎么处理？",
        storage_dir=Path("fake-storage"),
        top_k=3,
        department="生产部",
    )

    assert result["question_type"] == classifier.KNOWLEDGE_QA
    assert result["route"] == classifier.ROUTE_RAG_ANSWER
    assert result["answer_review"]["answer_quality"] == graph.ANSWER_QUALITY_WEAK
    assert result["answer_review"]["needs_review"] is True
    assert result["answer_review"]["has_sources"] is False
    assert result["answer_review"]["used_count"] == 0
    assert result["answer_review"]["human_review_required"] is True
    assert (
        result["answer_review"]["human_review_status"]
        == graph.HUMAN_REVIEW_STATUS_REQUIRED
    )
    assert result["answer_review"]["human_review_decision"] is None


def test_run_agent_routes_unknown_question_to_fallback(monkeypatch):
    def fake_chat_with_retrieval(**kwargs):
        raise AssertionError("unknown question should not call RAG")

    monkeypatch.setattr(graph, "chat_with_retrieval", fake_chat_with_retrieval)

    result = graph.run_agent(
        question="今天天气如何",
        storage_dir=Path("storage"),
        top_k=3,
    )

    assert result["question_type"] == classifier.UNKNOWN
    assert result["route"] == classifier.ROUTE_FALLBACK
    assert result["answer"] == NO_ANSWER_MESSAGE
    assert result["sources"] == []
    assert result["retrieval"]["retrieved_count"] == 0
    assert result["retrieval"]["used_count"] == 0
    assert result["answer_review"] == {}
