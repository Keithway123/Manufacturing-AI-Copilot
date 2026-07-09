from pathlib import Path

SERVICE_NAME = "manufacturing-ai-copilot"
VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STORAGE_DIR = PROJECT_ROOT / "storage"

LLM_MODEL = "qwen3.7-plus"
EMBEDDING_MODEL = "text-embedding-v3"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DEFAULT_RETRIEVAL_TOP_K = 3
MAX_RETRIEVAL_TOP_K = 10
MIN_RETRIEVAL_SCORE = 0.6

NO_ANSWER_MESSAGE = "当前知识库未找到足够相关的内容，请换个问法或补充资料。"
