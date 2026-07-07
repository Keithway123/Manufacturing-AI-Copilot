from llama_index.core import Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding


def configure_embedding() -> None:
    Settings.embed_model = DashScopeEmbedding(model_name="text-embedding-v3")
