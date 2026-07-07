from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from manufacturing_ai_copilot.rag.embedding import configure_embedding


def load_retriever(storage_dir: Path, similarity_top_k: int = 3):
    index_store_path = storage_dir / "index_store.json"
    if not index_store_path.exists():
        raise FileNotFoundError(
            f"RAG index not found: {index_store_path}. Run scripts.build_index first."
        )

    configure_embedding()

    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))

    index = load_index_from_storage(storage_context)

    return index.as_retriever(similarity_top_k=similarity_top_k)


def retrieve_matches(
    storage_dir: Path,
    question: str,
    similarity_top_k: int = 3,
) -> list[dict]:

    retriever = load_retriever(
        storage_dir=storage_dir,
        similarity_top_k=similarity_top_k,
    )

    nodes = retriever.retrieve(question)

    matches = []

    for node in nodes:
        metadata = node.node.metadata

        # 返回结构化结果，方便 FastAPI 直接转成 JSON
        matches.append(
            {
                "score": node.score,
                "document": metadata.get("file_name"),
                "title": metadata.get("title"),
                "content": node.node.get_content(),
                "metadata": metadata,
            }
        )
    return matches


def query_index(storage_dir: Path, question: str, similarity_top_k: int = 3) -> str:
    matches = retrieve_matches(
        storage_dir=storage_dir,
        question=question,
        similarity_top_k=similarity_top_k,
    )

    lines = []

    for match in matches:
        lines.append(f"score:{match['score']}")
        lines.append(f"document:{match['document']}")
        lines.append(f"title:{match['title']}")
        lines.append(match["content"][:500])
        lines.append("-" * 40)

    return "\n".join(lines)


def build_retrieval_answer(matches: list[dict]) -> str:
    if not matches:
        return "未在当前知识库中找到相关内容。"

    top_match = matches[0]
    title = top_match.get("title") or "相关文档"
    content = top_match.get("content") or ""

    preview = content[:600].strip()

    return f"根据《{title}》中的相关内容: \n\n{preview}"


def chat_with_retrieval(
    storage_dir: Path,
    question: str,
    similarity_top_k: int = 3,
) -> dict:

    matches = retrieve_matches(
        storage_dir=storage_dir,
        question=question,
        similarity_top_k=similarity_top_k,
    )

    answer = build_retrieval_answer(matches)

    sources = []
    for match in matches:
        sources.append(
            {
                "title": match.get("title"),
                "document": match.get("document"),
                "score": match.get("score"),
            }
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
