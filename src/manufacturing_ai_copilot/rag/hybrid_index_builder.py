from qdrant_client import QdrantClient, models
from manufacturing_ai_copilot.core.config import (
    QDRANT_DISTANCE,
    QDRANT_HYBRID_COLLECTION_NAME,
    QDRANT_HYBRID_URL,
    QDRANT_VECTOR_SIZE,
)
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
