from pathlib import Path

SERVICE_NAME = "manufacturing-ai-copilot"
VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STORAGE_DIR = PROJECT_ROOT / "storage"

LLM_MODEL = "qwen3.7-plus"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
