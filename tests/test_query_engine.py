from manufacturing_ai_copilot.rag.query_engine import (
    build_sources,
    filter_matches_by_score,
)


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
            "metadata": {"doc_id": "smt_alarm_sop"},
        },
        {
            "title": "SMT设备报警处理SOP",
            "document": "SMT设备报警处理SOP.md",
            "score": 0.9,
            "metadata": {"doc_id": "smt_alarm_sop"},
        },
        {
            "title": "MES工单状态说明",
            "document": "MES工单状态说明.md",
            "score": 0.8,
            "metadata": {"doc_id": "mes_work_order_status"},
        },
    ]

    sources = build_sources(matches)

    assert sources == [
        {
            "title": "SMT设备报警处理SOP",
            "document": "SMT设备报警处理SOP.md",
            "score": 0.9,
        },
        {
            "title": "MES工单状态说明",
            "document": "MES工单状态说明.md",
            "score": 0.8,
        },
    ]
