from manufacturing_ai_copilot.rag.query_engine import (
    build_metadata_filters,
    build_sources,
    filter_matches_by_score,
)
from pathlib import Path
from manufacturing_ai_copilot.rag import query_engine


def test_build_metadata_filters_returns_none_without_department():
    assert build_metadata_filters(None) is None


def test_build_metadata_filters_filters_by_department():
    filters = build_metadata_filters("生产部")

    assert filters is not None
    assert len(filters.filters) == 1
    assert filters.filters[0].key == "department"
    assert filters.filters[0].value == "生产部"


def test_filter_matches_by_score_keeps_scores_at_or_above_threshold():
    # 测试过滤掉低分的match
    matches = [
        {"score": 0.9},
        {"score": 0.6},
        {"score": 0.59},
        {"score": None},
        {},
    ]

    filtered = filter_matches_by_score(matches, min_score=0.6)

    assert filtered == [
        {"score": 0.9},
        {"score": 0.6},
    ]


def test_build_sources_deduplicatest_by_doc_id_and_keeps_highest_score():
    # build_source 根据doc_id 去重
    matches = [
        {
            "title": "SMT设备报警处理SOP",
            "document": "SMT设备报警处理SOP.md",
            "score": 0.7,
            "metadata": {
                "doc_id": "smt_alarm_sop",
                "department": "生产部",
                "version": "v1.0",
            },
        },
        {
            "title": "SMT设备报警处理SOP",
            "document": "SMT设备报警处理SOP.md",
            "score": 0.9,
            "metadata": {
                "doc_id": "smt_alarm_sop",
                "department": "生产部",
                "version": "v1.0",
            },
        },
        {
            "title": "MES工单状态说明",
            "document": "MES工单状态说明.md",
            "score": 0.8,
            "metadata": {
                "doc_id": "mes_work_order_status",
                "department": "生产部",
                "version": "v1.1",
            },
        },
    ]

    sources = build_sources(matches)

    assert sources == [
        {
            "title": "SMT设备报警处理SOP",
            "document": "SMT设备报警处理SOP.md",
            "doc_id": "smt_alarm_sop",
            "department": "生产部",
            "version": "v1.0",
            "score": 0.9,
        },
        {
            "title": "MES工单状态说明",
            "document": "MES工单状态说明.md",
            "doc_id": "mes_work_order_status",
            "department": "生产部",
            "version": "v1.1",
            "score": 0.8,
        },
    ]


def test_chat_with_retrieval_passes_domain_type_to_llm(monkeypatch):
    def fake_retrieve_matches(
        storage_dir,
        question,
        similarity_top_k,
        department,
    ):
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
            }
        ]

    def fake_generate_answer_with_qwen(
        question,
        matches,
        domain_type,
    ):
        assert domain_type == "equipment_sop"
        return "fake answer"

    monkeypatch.setattr(query_engine, "retrieve_matches", fake_retrieve_matches)
    monkeypatch.setattr(
        query_engine,
        "generate_answer_with_qwen",
        fake_generate_answer_with_qwen,
    )

    result = query_engine.chat_with_retrieval(
        storage_dir=Path("storage"),
        question="贴片机报警 E203 怎么处理？",
        similarity_top_k=3,
        department=None,
        domain_type="equipment_sop",
    )

    assert result["answer"] == "fake answer"
    assert result["sources"][0]["doc_id"] == "smt_alarm_sop"
