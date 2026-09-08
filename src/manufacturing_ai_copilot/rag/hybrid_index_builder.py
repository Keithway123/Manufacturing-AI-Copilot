from pathlib import Path

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode

from manufacturing_ai_copilot.rag.index_builder import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    load_markdown_documents,
)

from qdrant_client import QdrantClient, models
from manufacturing_ai_copilot.core.config import (
    QDRANT_DISTANCE,
    QDRANT_HYBRID_COLLECTION_NAME,
    QDRANT_HYBRID_URL,
    QDRANT_VECTOR_SIZE,
)
from manufacturing_ai_copilot.rag.embedding import configure_embedding
from manufacturing_ai_copilot.rag.qdrant_client import get_qdrant_client
from manufacturing_ai_copilot.rag.sparse_embedding import encode_sparse

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# Build hybrid index Collection
def recreate_hybrid_collection() -> QdrantClient:
    client = get_qdrant_client(QDRANT_HYBRID_URL)

    if client.collection_exists(QDRANT_HYBRID_COLLECTION_NAME):
        client.delete_collection(
            collection_name=QDRANT_HYBRID_COLLECTION_NAME,
        )

    client.create_collection(
        collection_name=QDRANT_HYBRID_COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=QDRANT_VECTOR_SIZE,
                distance=models.Distance(QDRANT_DISTANCE),
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )
    return client


# 组装Point Struct
def build_hybrid_point(
    point_id: int | str,
    text: str,
    metadata: dict,
    dense_vector: list[float],
) -> models.PointStruct:

    if len(dense_vector) != QDRANT_VECTOR_SIZE:
        raise ValueError(
            "Dense vector dimension mismatch: "
            f"expected {QDRANT_VECTOR_SIZE}, "
            f"got {len(dense_vector)}"
        )

    sparse_vector = encode_sparse(text)

    payload = dict(metadata)
    payload["text"] = text

    return models.PointStruct(
        id=point_id,
        vector={
            DENSE_VECTOR_NAME: dense_vector,
            SPARSE_VECTOR_NAME: sparse_vector,
        },
        payload=payload,
    )


def build_hybrid_nodes(raw_dir: Path) -> list[BaseNode]:
    documents = load_markdown_documents(raw_dir)

    # SentenceSplitter 是一个以 token 大小为约束、优先保持段落和完整句子、
    # 必要时递归细分并支持 overlap 的文本切分器。
    splitter = SentenceSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    return splitter.get_nodes_from_documents(documents)


def embed_hybrid_nodes(nodes: list[BaseNode]) -> list[list[float]]:
    if not nodes:
        return []

    configure_embedding()

    # 保持与原 Dense 基线相同的 embedding 输入格式。
    texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]

    return Settings.embed_model.get_text_embedding_batch(texts)


def build_hybrid_index(raw_dir: Path) -> int:
    nodes = build_hybrid_nodes(raw_dir)
    dense_vectors = embed_hybrid_nodes(nodes)

    if len(nodes) != len(dense_vectors):
        raise ValueError("Nodes and dense vectors count mismatch")

    points = []

    for node, dense_vector in zip(nodes, dense_vectors):
        text = node.get_content(metadata_mode=MetadataMode.NONE)

        point = build_hybrid_point(
            point_id=node.node_id,
            text=text,
            metadata=node.metadata,
            dense_vector=dense_vector,
        )
        points.append(point)

    client = recreate_hybrid_collection()

    if points:
        client.upsert(
            collection_name=QDRANT_HYBRID_COLLECTION_NAME,
            points=points,
            wait=True,
        )

    return len(points)
