import argparse

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from manufacturing_ai_copilot.core.config import (
    MIN_RETRIEVAL_SCORE,
    QDRANT_COLLECTION_NAME,
)
from manufacturing_ai_copilot.rag.embedding import configure_embedding
from manufacturing_ai_copilot.rag.qdrant_client import get_qdrant_client


def load_qdrant_retriever(top_k: int):
    configure_embedding()

    client = get_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    return index.as_retriever(
        similarity_top_k=top_k,
    )


def query_qdrant(
    question: str,
    top_k: int,
    min_score: float,
) -> list[dict]:
    retriever = load_qdrant_retriever(top_k=top_k)
    nodes = retriever.retrieve(question)

    results = []

    for node_with_score in nodes:
        if node_with_score is None:
            continue

        if node_with_score.score < min_score:
            continue

        node = node_with_score.node
        metadata = node.metadata or {}

        results.append(
            {
                "score": node_with_score.score,
                "text": node.get_content(),
                "metadata": metadata,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Qdrant collection without calling LLM.",
    )
    parser.add_argument(
        "question",
        help="Question used for Qdrant semantic retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of similar chunks to retrieve.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_RETRIEVAL_SCORE,
        help="Minimum similarity score to keep.",
    )

    args = parser.parse_args()
    results = query_qdrant(
        question=args.question,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    if not results:
        print("No results above min_score.")
        return

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\n[{index}] score={result['score']}")
        print(
            "source="
            f"{metadata.get('title')} "
            f"({metadata.get('file_name')}), "
            f"department={metadata.get('department')}, "
            f"version={metadata.get('version')}"
        )
        print("chunk:")
        print(result["text"][:500])


if __name__ == "__main__":
    main()
