from qdrant_client import QdrantClient

from manufacturing_ai_copilot.core.config import QDRANT_URL


def get_qdrant_client() -> QdrantClient:
    # 这里只统一创建客户端，不执行健康检查或管理collection。
    return QdrantClient(url=QDRANT_URL)
