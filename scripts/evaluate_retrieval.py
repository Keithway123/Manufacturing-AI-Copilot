import yaml

from pathlib import Path
from manufacturing_ai_copilot.rag.query_engine import retrieve_matches
from manufacturing_ai_copilot.core.config import MIN_RETRIEVAL_SCORE


def get_max_score(matches: list[dict]) -> float | None:
    scores = [match.get("score") for match in matches if match.get("score") is not None]

    if not scores:
        return None

    return max(scores)


def load_eval_questions(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) or []


def get_hit_rank(matches: list[dict], expected_doc_id: str) -> int | None:
    for index, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        if metadata.get("doc_id") == expected_doc_id:
            return index

    return None


def evaluate_question(
    question_item: dict,
    storage_dir: Path,
    top_k: int,
) -> dict:
    matches = retrieve_matches(
        storage_dir=storage_dir,
        question=question_item["question"],
        similarity_top_k=top_k,
    )

    expected_doc_id = question_item["expected_doc_id"]
    max_score = get_max_score(matches)

    if expected_doc_id is None:
        passed = max_score is None or max_score < MIN_RETRIEVAL_SCORE

        return {
            "id": question_item["id"],
            "question": question_item["question"],
            "expected_doc_id": expected_doc_id,
            "type": "no_answer",
            "hit": passed,
            "rank": None,
            "max_score": max_score,
            "retrieved_doc_ids": [
                match.get("metadata", {}).get("doc_id") for match in matches
            ],
        }

    hit_rank = get_hit_rank(
        matches=matches,
        expected_doc_id=expected_doc_id,
    )

    return {
        "id": question_item["id"],
        "question": question_item["question"],
        "expected_doc_id": expected_doc_id,
        "type": "answerable",
        "hit": hit_rank is not None,
        "rank": hit_rank,
        "max_score": max_score,
        "retrieved_doc_ids": [
            match.get("metadata", {}).get("doc_id") for match in matches
        ],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    eval_path = project_root / "data" / "eval" / "questions.yaml"
    storage_dir = project_root / "storage"
    top_k = 3

    questions = load_eval_questions(eval_path)

    results = []
    for question_item in questions:
        result = evaluate_question(
            question_item=question_item,
            storage_dir=storage_dir,
            top_k=top_k,
        )
        results.append(result)

    answerable_results = [
        result for result in results if result["type"] == "answerable"
    ]
    no_answer_results = [result for result in results if result["type"] == "no_answer"]

    hit_count = sum(1 for result in answerable_results if result["hit"])
    top1_count = sum(1 for result in answerable_results if result["rank"] == 1)
    no_answer_pass_count = sum(1 for result in no_answer_results if result["hit"])
    failed_results = [result for result in results if not result["hit"]]

    answerable_count = len(answerable_results)
    no_answer_count = len(no_answer_results)

    for result in results:
        status = "PASS" if result["hit"] else "FAIL"
        print(
            f"{status} {result['id']} "
            f"type={result['type']} "
            f"rank={result['rank']} "
            f"max_score={result['max_score']} "
            f"retrieved={result['retrieved_doc_ids']}"
        )

    print(f"\nHit@{top_k}: {hit_count}/{answerable_count}")
    print(f"Top1 Accuracy: {top1_count}/{answerable_count}")
    print(f"No-answer Pass: " f"{no_answer_pass_count}/{no_answer_count}")

    if failed_results:
        print("\nFailed case:")
        for result in failed_results:
            print(
                f"- {result['id']}"
                f"expected={result['expected_doc_id']} "
                f"retrieved={result['retrieved_doc_ids']} "
            )


if __name__ == "__main__":
    main()
