import argparse
from pathlib import Path

from manufacturing_ai_copilot.agent.graph import run_agent
from manufacturing_ai_copilot.core.config import DEFAULT_RETRIEVAL_TOP_K, STORAGE_DIR
from manufacturing_ai_copilot.rag.llm import get_answer_instruction


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
    print(f"Question Type: {result['question_type']}")
    print(f"Domain Type: {result['domain_type']}")
    print(f"Route: {result['route']}")
    print(f"Answer Strategy: {get_answer_instruction(result['domain_type'])}")

    if result["tool_result"]:
        print(f"Tool Result: {result['tool_result']}")

    print()

    print(result["answer"])
    print()

    print("Sources:")
    if result["sources"]:
        for source in result["sources"]:
            print(
                f"- {source.get('title')} "
                f"({source.get('document')}), "
                f"score={source.get('score')}"
            )
    else:
        print("(none)")


if __name__ == "__main__":
    main()
