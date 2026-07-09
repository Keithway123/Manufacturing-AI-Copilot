import pytest
from manufacturing_ai_copilot.rag.llm import generate_answer_with_qwen


def test_generate_answer_with_qwen_raises_when_matches_are_empty():
    with pytest.raises(ValueError):
        generate_answer_with_qwen(question="测试问题", matches=[])
