import argparse

from manufacturing_ai_copilot.core.config import DEFAULT_RETRIEVAL_TOP_K
from manufacturing_ai_copilot.rag.hybrid_query_engine import (
    DEFAULT_HYBRID_PREFETCH_LIMIT,
    query_hybrid,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the hybrid Qdrant collection with Dense, Sparse, and RRF.",
    )
    parser.add_argument(
        "question",
        help="Question used for hybrid retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
        help="Number of fused results to return.",
    )
    parser.add_argument(
        "--prefetch-limit",
        type=int,
        default=DEFAULT_HYBRID_PREFETCH_LIMIT,
        help="Number of candidates recalled by each retrieval route.",
    )
    args = parser.parse_args()

    results = query_hybrid(
        question=args.question,
        top_k=args.top_k,
        prefetch_limit=args.prefetch_limit,
    )

    if not results:
        print("No hybrid retrieval results.")
        return

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]

        # RRF score is rank-based and is not comparable to the Dense cosine score.
        print(f"\n[{index}] rrf_score={result['score']}")
        print(
            "source="
            f"{result['title']} "
            f"({result['document']}), "
            f"department={metadata.get('department')}, "
            f"version={metadata.get('version')}"
        )
        print("chunk:")
        print(result["content"][:500])


if __name__ == "__main__":
    main()
