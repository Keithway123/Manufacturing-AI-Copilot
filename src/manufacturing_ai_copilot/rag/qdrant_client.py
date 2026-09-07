from qdrant_client import QdrantClient

from manufacturing_ai_copilot.core.config import QDRANT_URL


def get_qdrant_client(url: str | None = None) -> QdrantClient:
    # 这里只统一创建客户端，不执行健康检查或管理collection。
    target_url = QDRANT_URL if url is None else url
    return QdrantClient(url=target_url)
