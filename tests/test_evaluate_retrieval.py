from manufacturing_ai_copilot.rag.evaluation import (
    get_hit_rank,
    get_max_score,
    load_eval_questions,
)


def test_get_max_score_returns_highest_score():
    matches = [
        {"score": 0.4},
        {"score": 0.8},
        {"score": 0.6},
    ]

    assert get_max_score(matches) == 0.8


def test_get_max_score_returns_none_when_no_score_exists():
    matches = [{}, {"score": None}]

    assert get_max_score(matches) is None


def test_get_hit_rank_returns_one_based_rank():
    matches = [
        {"metadata": {"doc_id": "wrong_doc"}},
        {"metadata": {"doc_id": "expected_doc"}},
    ]

    assert get_hit_rank(matches, "expected_doc") == 2


def test_get_hit_rank_returns_none_when_expected_doc_is_missing():
    matches = [
        {"metadata": {"doc_id": "wrong_doc"}},
    ]

    assert get_hit_rank(matches, "expected_doc") is None


def test_load_eval_questions_reads_yaml_file(tmp_path):
    eval_file = tmp_path / "questions.yaml"
    eval_file.write_text(
        """
- id: test_case
  question: 测试问题
  expected_doc_id: smt_alarm_sop
""".strip(),
        encoding="utf-8",
    )

    questions = load_eval_questions(eval_file)

    assert questions == [
        {
            "id": "test_case",
            "question": "测试问题",
            "expected_doc_id": "smt_alarm_sop",
        }
    ]
