from pathlib import Path


from manufacturing_ai_copilot.core.config import (
    MIN_RETRIEVAL_SCORE,
    NO_ANSWER_MESSAGE,
)
from manufacturing_ai_copilot.rag import query_engine


def test_chat_with_retrieval_returns_no_answer_without_calling_qwen(monkeypatch):
    # 验证/chat编排逻辑，不读取真实storage.
    low_score_matches = [
        {
            "score": MIN_RETRIEVAL_SCORE - 0.1,
            "title": "低相关文档",
            "document": "low.md",
            "content": "低相关内容",
            "metadata": {"doc_id": "low_doc"},
        }
    ]

    def fake_retrieve_matches(storage_dir: Path, question: str, similarity_top_k: int):
        return low_score_matches

    def fail_if_qwen_is_called(question: str, matches: list[dict]):
        raise AssertionError("Qwen should not be called for no-answer branch.")

    monkeypatch.setattr(query_engine, "retrieve_matches", fake_retrieve_matches)
    monkeypatch.setattr(
        query_engine,
        "generate_answer_with_qwen",
        fail_if_qwen_is_called,
    )

    result = query_engine.chat_with_retrieval(
        storage_dir=Path("fake-storage"),
        question="食堂午餐几点开始？",
        similarity_top_k=3,
    )

    assert result["answer"] == NO_ANSWER_MESSAGE
    assert result["sources"] == []
    assert result["retrieval"] == {
        "top_k": 3,
        "min_score": MIN_RETRIEVAL_SCORE,
        "retrieved_count": 1,
        "used_count": 0,
    }


def test_chat_with_retrieval_calls_qwen_and_returns_sources(monkeypatch):
    high_score_matches = [
        {
            "score": 0.9,
            "title": "SMT设备报警处理SOP",
            "document": "SMT设备报警处理SOP.md",
            "content": "E203 处理步骤",
            "metadata": {"doc_id": "smt_alarm_sop"},
        },
        {
            "score": 0.8,
            "title": "SMT设备报警处理SOP",
            "document": "SMT设备报警处理SOP.md",
            "content": "E203 注意事项",
            "metadata": {"doc_id": "smt_alarm_sop"},
        },
    ]

    def fake_retrieve_matches(storage_dir: Path, question: str, similarity_top_k: int):
        return high_score_matches

    def fake_generate_answer_with_qwen(question: str, matches: list[dict]):
        assert matches == high_score_matches
        return "fake answer"

    monkeypatch.setattr(query_engine, "retrieve_matches", fake_retrieve_matches)
    monkeypatch.setattr(
        query_engine,
        "generate_answer_with_qwen",
        fake_generate_answer_with_qwen,
    )

    result = query_engine.chat_with_retrieval(
        storage_dir=Path("fake-storage"),
        question="贴片机报警E203怎么处理？",
        similarity_top_k=3,
    )

    assert result["answer"] == "fake answer"
    assert result["sources"] == [
        {
            "title": "SMT设备报警处理SOP",
            "document": "SMT设备报警处理SOP.md",
            "score": 0.9,
        }
    ]

    assert result["retrieval"] == {
        "top_k": 3,
        "min_score": MIN_RETRIEVAL_SCORE,
        "retrieved_count": 2,
        "used_count": 2,
    }
