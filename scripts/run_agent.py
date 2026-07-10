import argparse
from pathlib import Path

from manufacturing_ai_copilot.agent.graph import run_agent
from manufacturing_ai_copilot.core.config import DEFAULT_RETRIEVAL_TOP_K, STORAGE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangGraph agent from CLI")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)
    parser.add_argument("--department", default=None)
    parser.add_argument("--storage-dir", type=Path, default=STORAGE_DIR)

    args = parser.parse_args()

    result = run_agent(
        question=args.question,
        storage_dir=args.storage_dir,
        top_k=args.top_k,
        department=args.department,
    )

    print(result["answer"])
    print()
    print("Sources:")
    for source in result["sources"]:
        print(
            f"- {source.get('title')}"
            f"({source.get('document')})"
            f"score={source.get('score')}"
        )


if __name__ == "__main__":
    main()
