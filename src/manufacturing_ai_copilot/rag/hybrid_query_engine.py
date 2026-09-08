from llama_index.core import Settings
from qdrant_client import models

from manufacturing_ai_copilot.core.config import (
    DEFAULT_RETRIEVAL_TOP_K,
    QDRANT_HYBRID_COLLECTION_NAME,
    QDRANT_HYBRID_URL,
)
from manufacturing_ai_copilot.rag.embedding import configure_embedding
from manufacturing_ai_copilot.rag.hybrid_index_builder import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)
from manufacturing_ai_copilot.rag.qdrant_client import get_qdrant_client
from manufacturing_ai_copilot.rag.sparse_embedding import encode_sparse

DEFAULT_HYBRID_PREFETCH_LIMIT = 20


def query_hybrid(
    question: str,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    prefetch_limit: int = DEFAULT_HYBRID_PREFETCH_LIMIT,
) -> list[dict]:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Question must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if prefetch_limit < top_k:
        raise ValueError("prefetch_limit must be greater than or equal to top_k")

    configure_embedding()
    dense_query = Settings.embed_model.get_query_embedding(normalized_question)
    sparse_query = encode_sparse(normalized_question)
    if not sparse_query.indices:
        raise ValueError("Question must contain Chinese characters, letters, or numbers")

    client = get_qdrant_client(QDRANT_HYBRID_URL)
    response = client.query_points(
        collection_name=QDRANT_HYBRID_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=sparse_query,
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            {
                "id": str(point.id),
                "score": point.score,
                "document": payload.get("file_name"),
                "title": payload.get("title"),
                "content": payload.get("text", ""),
                "metadata": payload,
            }
        )

    return results
