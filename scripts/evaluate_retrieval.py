import yaml

from pathlib import Path
from manufacturing_ai_copilot.rag.query_engine import retrieve_matches


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

    hit_rank = get_hit_rank(
        matches=matches,
        expected_doc_id=question_item["expected_doc_id"],
    )

    return {
        "id": question_item["id"],
        "question": question_item["question"],
        "expected_doc_id": question_item["expected_doc_id"],
        "hit": hit_rank is not None,
        "rank": hit_rank,
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

    hit_count = sum(1 for result in results if result["hit"])
    total_count = len(results)

    for result in results:
        status = "PASS" if result["hit"] else "FAIL"
        print(
            f"{status} {result['id']} "
            f"rank={result['rank']} "
            f"retrieved={result['retrieved_doc_ids']}"
        )

    print(f"\nHit@{top_k}: {hit_count}/{total_count}")


if __name__ == "__main__":
    main()
