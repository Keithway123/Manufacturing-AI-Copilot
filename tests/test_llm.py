import pytest
from manufacturing_ai_copilot.rag.llm import generate_answer_with_qwen
from manufacturing_ai_copilot.rag.llm import get_answer_instruction


def test_generate_answer_with_qwen_raises_when_matches_are_empty():
    with pytest.raises(ValueError):
        generate_answer_with_qwen(question="测试问题", matches=[])


def test_get_answer_instruction_returns_domain_instruction():
    instruction = get_answer_instruction("equipment_sop")

    assert "设备SOP" in instruction


def test_get_answer_instruction_falls_back_to_general_instruction():
    instruction = get_answer_instruction("not_exist_domain")

    assert "知识库内容" in instruction
