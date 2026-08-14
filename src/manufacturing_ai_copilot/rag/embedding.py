from llama_index.core import Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding
from manufacturing_ai_copilot.core.config import EMBEDDING_MODEL


def configure_embedding() -> None:
    Settings.embed_model = DashScopeEmbedding(model_name=EMBEDDING_MODEL)
