from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from manufacturing_ai_copilot.rag.embedding import configure_embedding


def load_retriever(storage_dir: Path, similarity_top_k: int = 3):
    configure_embedding()

    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))

    index = load_index_from_storage(storage_context)

    # return index.as_query_engine()
    return index.as_retriever(similarity_top_k=similarity_top_k)


def query_index(storage_dir: Path, question: str) -> str:

    retriever = load_retriever(storage_dir)
    nodes = retriever.retrieve(question)

    lines = []
    for node in nodes:
        metadata = node.node.metadata
        lines.append(f"score: {node.score}")
        lines.append(f"document: {metadata.get('file_name')}")
        lines.append(f"title: {metadata.get('title')}")
        lines.append(node.node.get_content()[:500])
        lines.append("-" * 40)

    return "\n".join(lines)
