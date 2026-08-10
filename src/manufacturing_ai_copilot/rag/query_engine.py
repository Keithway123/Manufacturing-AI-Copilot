from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.qdrant import QdrantVectorStore
from manufacturing_ai_copilot.rag.embedding import configure_embedding
from manufacturing_ai_copilot.rag.llm import generate_answer_with_qwen
from manufacturing_ai_copilot.core.config import (
    DEFAULT_RETRIEVAL_TOP_K,
    MIN_RETRIEVAL_SCORE,
    NO_ANSWER_MESSAGE,
    QDRANT_COLLECTION_NAME,
)
from manufacturing_ai_copilot.rag.qdrant_client import get_qdrant_client


# Qdrant /search
def build_match_from_node(node_with_score) -> dict:
    node = node_with_score.node
    metadata = node.metadata or {}

    return {
        "score": node_with_score.score,
        "document": metadata.get("file_name"),
        "title": metadata.get("title"),
        "content": node.get_content(),
        "metadata": metadata,
    }


def load_qdrant_retriever(
    similarity_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    department: str | None = None,
):
    configure_embedding()

    # vector_storage = 向量数据实际存放在哪里、怎么读写它
    vector_storage = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=QDRANT_COLLECTION_NAME,
    )

    # storage_context = 告诉 LlamaIndex：这次索引相关的数据存储组件用哪些
    storage_context = StorageContext.from_defaults(vector_store=vector_storage)

    # VectorStoreIndex-> 基于这个 storage_context 组装索引对象
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_storage,
        storage_context=storage_context,
    )

    return index.as_retriever(
        similarity_top_k=similarity_top_k,
        filters=build_metadata_filters(department),
    )


# 过滤低分的match
def filter_matches_by_score(
    matches: list[dict],
    min_score: float = MIN_RETRIEVAL_SCORE,
) -> list[dict]:
    filtered_matches = []

    for match in matches:
        score = match.get("score")
        if score is not None and score >= min_score:
            filtered_matches.append(match)

    return filtered_matches


def build_metadata_filters(department: str | None) -> MetadataFilters | None:
    if department is None:
        return None

    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="department",
                value=department,
            )
        ]
    )


def load_retriever(
    storage_dir: Path,
    similarity_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    department: str | None = None,
):
    index_store_path = storage_dir / "index_store.json"
    if not index_store_path.exists():
        raise FileNotFoundError(
            f"RAG index not found: {index_store_path}. Run scripts.build_index first."
        )

    configure_embedding()

    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))

    index = load_index_from_storage(storage_context)

    return index.as_retriever(
        similarity_top_k=similarity_top_k,
        filters=build_metadata_filters(department),
    )


def retrieve_matches(
    storage_dir: Path,
    question: str,
    similarity_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    department: str | None = None,
) -> list[dict]:

    retriever = load_retriever(
        storage_dir=storage_dir,
        similarity_top_k=similarity_top_k,
        department=department,
    )

    nodes = retriever.retrieve(question)

    matches = [build_match_from_node(node) for node in nodes]
    return matches


def query_index(
    storage_dir: Path,
    question: str,
    similarity_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> str:
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


# def build_retrieval_answer(matches: list[dict]) -> str:
#     # 在未接LLM的时候测试 /chat 接口
#     if not matches:
#         return "未在当前知识库中找到相关内容。"

#     top_match = matches[0]
#     title = top_match.get("title") or "相关文档"
#     content = top_match.get("content") or ""

#     preview = content[:600].strip()


#     return f"根据《{title}》中的相关内容: \n\n{preview}"


def build_source(match: dict, doc_id: str) -> dict:
    metadata = match.get("metadata", {})

    return {
        "doc_id": doc_id,
        "title": match.get("title"),
        "document": match.get("document"),
        "department": metadata.get("department"),
        "version": metadata.get("version"),
        "score": match.get("score"),
    }


# 按doc_id去重
def build_sources(matches: list[dict]) -> list[dict]:

    sources_by_doc_id = {}

    for match in matches:
        metadata = match.get("metadata", {})
        doc_id = metadata.get("doc_id") or match.get("document")
        score = match.get("score")

        if doc_id not in sources_by_doc_id:
            sources_by_doc_id[doc_id] = build_source(match, doc_id)
            continue

        current_score = sources_by_doc_id[doc_id]["score"]
        if score is not None and (current_score is None or score > current_score):
            sources_by_doc_id[doc_id] = build_source(match, doc_id)
    return list(sources_by_doc_id.values())


def chat_with_retrieval(
    storage_dir: Path,
    question: str,
    similarity_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    department: str | None = None,
    domain_type: str = "general_knowledge",
) -> dict:

    matches = retrieve_matches(
        storage_dir=storage_dir,
        question=question,
        similarity_top_k=similarity_top_k,
        department=department,
    )
    # answer = build_retrieval_answer(matches)
    filtered_matches = filter_matches_by_score(matches)

    if not filtered_matches:
        return {
            "question": question,
            "answer": NO_ANSWER_MESSAGE,
            "sources": [],
            "retrieval": {
                "top_k": similarity_top_k,
                "min_score": MIN_RETRIEVAL_SCORE,
                "retrieved_count": len(matches),
                "used_count": 0,
            },
        }

    answer = generate_answer_with_qwen(
        question=question,
        matches=filtered_matches,
        domain_type=domain_type,
    )

    sources = build_sources(filtered_matches)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieval": {
            "top_k": similarity_top_k,
            "min_score": MIN_RETRIEVAL_SCORE,
            "retrieved_count": len(matches),
            "used_count": len(filtered_matches),
        },
    }


def retrieve_qdrant_raw_matches(
    question: str,
    similarity_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    department: str | None = None,
) -> list[dict]:
    retriever = load_qdrant_retriever(
        similarity_top_k=similarity_top_k,
        department=department,
    )

    nodes = retriever.retrieve(question)

    return [build_match_from_node(node) for node in nodes]


def retrieve_qdrant_matches(
    question: str,
    similarity_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    department: str | None = None,
) -> list[dict]:
    matches = retrieve_qdrant_raw_matches(
        question=question,
        similarity_top_k=similarity_top_k,
        department=department,
    )

    return filter_matches_by_score(matches)


def chat_with_qdrant_retrieval(
    question: str,
    similarity_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    department: str | None = None,
    domain_type: str = "general_knowledge",
) -> dict:
    matches = retrieve_qdrant_raw_matches(
        question=question,
        similarity_top_k=similarity_top_k,
        department=department,
    )

    filtered_matches = filter_matches_by_score(matches)

    if not filtered_matches:
        return {
            "question": question,
            "answer": NO_ANSWER_MESSAGE,
            "sources": [],
            "retrieval": {
                "top_k": similarity_top_k,
                "min_score": MIN_RETRIEVAL_SCORE,
                "retrieved_count": len(matches),
                "used_count": 0,
            },
        }

    answer = generate_answer_with_qwen(
        question=question,
        matches=filtered_matches,
        domain_type=domain_type,
    )

    sources = build_sources(filtered_matches)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieval": {
            "top_k": similarity_top_k,
            "min_score": MIN_RETRIEVAL_SCORE,
            "retrieved_count": len(matches),
            "used_count": len(filtered_matches),
        },
    }
