from qdrant_client import QdrantClient, models
from manufacturing_ai_copilot.core.config import (
    QDRANT_DISTANCE,
    QDRANT_HYBRID_COLLECTION_NAME,
    QDRANT_HYBRID_URL,
    QDRANT_VECTOR_SIZE,
)
from manufacturing_ai_copilot.rag.qdrant_client import get_qdrant_client

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


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
