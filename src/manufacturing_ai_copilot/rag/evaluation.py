from pathlib import Path

import yaml


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
