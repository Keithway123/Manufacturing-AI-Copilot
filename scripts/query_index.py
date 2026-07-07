import sys
from pathlib import Path

from manufacturing_ai_copilot.rag.query_engine import query_index


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Usage: python -m scripts.query_index <queston>")

    question = " ".join(sys.argv[1:])

    project_root = Path(__file__).resolve().parents[1]

    storage_dir = project_root / "storage"

    answer = query_index(storage_dir=storage_dir, question=question)

    print(answer)


if __name__ == "__main__":
    main()
