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

    def fake_retrieve_matches(
        storage_dir: Path,
        question: str,
        similarity_top_k: int,
        department: str | None,
    ):
        assert department is None
        return low_score_matches

    def fail_if_qwen_is_called(question: str, matches: list[dict], domain_type: str):
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
            "metadata": {
                "doc_id": "smt_alarm_sop",
                "department": "生产部",
                "version": "v1.0",
            },
        },
        {
            "score": 0.8,
            "title": "SMT设备报警处理SOP",
            "document": "SMT设备报警处理SOP.md",
            "content": "E203 注意事项",
            "metadata": {
                "doc_id": "smt_alarm_sop",
                "department": "生产部",
                "version": "v1.0",
            },
        },
    ]

    def fake_retrieve_matches(
        storage_dir: Path,
        question: str,
        similarity_top_k: int,
        department: str | None,
    ):
        assert department == "生产部"
        return high_score_matches

    def fake_generate_answer_with_qwen(question: str, matches: list[dict], domain_type):
        assert matches == high_score_matches
        assert domain_type == "general_knowledge"
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
        department="生产部",
    )

    assert result["answer"] == "fake answer"
    assert result["sources"] == [
        {
            "title": "SMT设备报警处理SOP",
            "document": "SMT设备报警处理SOP.md",
            "doc_id": "smt_alarm_sop",
            "department": "生产部",
            "version": "v1.0",
            "score": 0.9,
        }
    ]

    assert result["retrieval"] == {
        "top_k": 3,
        "min_score": MIN_RETRIEVAL_SCORE,
        "retrieved_count": 2,
        "used_count": 2,
    }


def test_chat_with_qdrant_retrieval_calls_qwen_and_returns_sources(monkeypatch):
    def fake_retrieve_qdrant_raw_matches(
        question: str,
        similarity_top_k: int,
        department: str | None,
    ):
        # Qdrant chat 需要保留过滤前结果，才能统计 retrieved_count / used_count。
        assert similarity_top_k == 3
        assert department is None
        return [
            {
                "score": 0.9,
                "document": "SMT设备报警处理SOP.md",
                "title": "SMT设备报警处理SOP",
                "content": "E203 处理步骤",
                "metadata": {
                    "doc_id": "smt_alarm_sop",
                    "department": "生产部",
                    "version": "v1.0",
                },
            },
            {
                "score": 0.3,
                "document": "质量异常8D报告模板.md",
                "title": "质量异常8D报告模板",
                "content": "低相关内容",
                "metadata": {
                    "doc_id": "quality_8d_template",
                    "department": "质量部",
                    "version": "v1.0",
                },
            },
        ]

    def fake_generate_answer_with_qwen(
        question: str,
        matches: list[dict],
        domain_type: str,
    ):
        assert len(matches) == 1
        assert matches[0]["score"] == 0.9
        assert domain_type == "equipment_sop"
        return "fake qdrant answer"

    monkeypatch.setattr(
        query_engine,
        "retrieve_qdrant_raw_matches",
        fake_retrieve_qdrant_raw_matches,
    )
    monkeypatch.setattr(
        query_engine,
        "generate_answer_with_qwen",
        fake_generate_answer_with_qwen,
    )

    result = query_engine.chat_with_qdrant_retrieval(
        question="贴片机报警 E203 怎么处理？",
        similarity_top_k=3,
        department=None,
        domain_type="equipment_sop",
    )

    assert result["answer"] == "fake qdrant answer"
    assert result["sources"] == [
        {
            "doc_id": "smt_alarm_sop",
            "title": "SMT设备报警处理SOP",
            "document": "SMT设备报警处理SOP.md",
            "department": "生产部",
            "version": "v1.0",
            "score": 0.9,
        }
    ]
    assert result["retrieval"] == {
        "top_k": 3,
        "min_score": MIN_RETRIEVAL_SCORE,
        "retrieved_count": 2,
        "used_count": 1,
    }


def test_chat_with_qdrant_retrieval_returns_no_answer_without_qwen(monkeypatch):
    def fake_retrieve_qdrant_raw_matches(
        question: str,
        similarity_top_k: int,
        department: str | None,
    ):
        return [
            {
                "score": 0.3,
                "document": "质量异常8D报告模板.md",
                "title": "质量异常8D报告模板",
                "content": "低相关内容",
                "metadata": {
                    "doc_id": "quality_8d_template",
                    "department": "质量部",
                    "version": "v1.0",
                },
            }
        ]

    def fail_if_qwen_is_called(
        question: str,
        matches: list[dict],
        domain_type: str,
    ):
        raise AssertionError("Qwen should not be called for low-score qdrant matches")

    monkeypatch.setattr(
        query_engine,
        "retrieve_qdrant_raw_matches",
        fake_retrieve_qdrant_raw_matches,
    )
    monkeypatch.setattr(
        query_engine,
        "generate_answer_with_qwen",
        fail_if_qwen_is_called,
    )

    result = query_engine.chat_with_qdrant_retrieval(
        question="今天天气如何？",
        similarity_top_k=3,
        department=None,
    )

    assert result["answer"] == NO_ANSWER_MESSAGE
    assert result["sources"] == []
    assert result["retrieval"] == {
        "top_k": 3,
        "min_score": MIN_RETRIEVAL_SCORE,
        "retrieved_count": 1,
        "used_count": 0,
    }
