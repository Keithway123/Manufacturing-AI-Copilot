from pathlib import Path

import yaml
from manufacturing_ai_copilot.rag.embedding import configure_embedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from manufacturing_ai_copilot.core.config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_DISTANCE,
    QDRANT_VECTOR_SIZE,
)
from manufacturing_ai_copilot.rag.qdrant_client import get_qdrant_client

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def _parse_markdown_with_metadata(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")

    if not text.startswith("---"):
        raise ValueError(f"Missing front matter: {path}")

    _, metadata_text, content = text.split("---", 2)
    metadata = yaml.safe_load(metadata_text) or {}
    metadata["file_name"] = path.name

    return Document(
        text=content.strip(),
        metadata=metadata,
    )


def load_markdown_documents(raw_dir: Path) -> list[Document]:
    markdown_files = sorted(raw_dir.glob("*.md"))
    if not markdown_files:
        raise ValueError(f"No Markdown files found in {raw_dir}")

    return [_parse_markdown_with_metadata(path) for path in markdown_files]


def recreate_qdrant_collection() -> QdrantClient:
    client = get_qdrant_client()

    if client.collection_exists(QDRANT_COLLECTION_NAME):
        client.delete_collection(
            collection_name=QDRANT_COLLECTION_NAME,
        )

    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=QDRANT_VECTOR_SIZE,
            distance=Distance(QDRANT_DISTANCE),
        ),
    )
    return client


def build_index(raw_dir: Path) -> None:
    configure_embedding()
    documents = load_markdown_documents(raw_dir)
    client = recreate_qdrant_collection()

    splitter = SentenceSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    # QdrantVectorStore负责把LlamaIndex节点转换并写入Qdrant。
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
    )

    # 将QdrantVectorStore注册为本次构建使用的向量存储。
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    # 文档解析、切块和 Embedding 逻辑不变；
    # 生成的向量与 metadata 将通过 storage_context 写入 Qdrant。
    VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        storage_context=storage_context,
    )
