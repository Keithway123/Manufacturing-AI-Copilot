from manufacturing_ai_copilot.agent import classifier


def test_classify_question_sets_domain_type():
    cases = [
        ("贴片机报警 E203 怎么处理", classifier.EQUIPMENT_SOP),
        ("MES工单暂停后怎么恢复？", classifier.PRODUCTION_ORDER),
        ("质量8D的根因分析怎么写？", classifier.QUALITY_ISSUE),
        ("VPN密码忘记了怎么办？", classifier.IT_SUPPORT),
    ]

    for question, expected_domain_type in cases:
        result = classifier.classify_question(question)

        assert result["question_type"] == classifier.KNOWLEDGE_QA
        assert result["route"] == "rag_answer"
        assert result["domain_type"] == expected_domain_type


def test_classify_question_sets_unknown_domain_for_unknown_question():
    result = classifier.classify_question("今天天气如何？")

    assert result["question_type"] == classifier.UNKNOWN
    assert result["route"] == "fallback"
    assert result["domain_type"] == classifier.UNKNOWN_DOMAIN
