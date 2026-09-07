import os
from pathlib import Path

SERVICE_NAME = "manufacturing-ai-copilot"
VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STORAGE_DIR = PROJECT_ROOT / "storage"

DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "manufacturing.db"
DATABASE_URL = os.getenv(
    "DATABASE_URL", f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://127.0.0.1:6333",  # 本地python调试默认访问宿主机映射端口
)

QDRANT_COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "manufacturing_knowledge",
)

QDRANT_HYBRID_URL = os.getenv(
    "QDRANT_HYBRID_URL",
    "http://127.0.0.1:6335",
)

QDRANT_HYBRID_COLLECTION_NAME = os.getenv(
    "QDRANT_HYBRID_COLLECTION_NAME",
    "manufacturing_knowledge_hybrid",
)
QDRANT_VECTOR_SIZE = 1024
QDRANT_DISTANCE = "Cosine"


LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.8-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://ws-5gwqjx9a2tapu6yk.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

DEFAULT_RETRIEVAL_TOP_K = 3
MAX_RETRIEVAL_TOP_K = 10
MIN_RETRIEVAL_SCORE = 0.6

NO_ANSWER_MESSAGE = "当前知识库未找到足够相关的内容，请换个问法或补充资料。"
