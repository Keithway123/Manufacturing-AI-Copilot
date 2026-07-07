from pathlib import Path

import yaml
from llama_index.core import Document, VectorStoreIndex
from manufacturing_ai_copilot.rag.embedding import configure_embedding
from llama_index.core.node_parser import SentenceSplitter

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


def build_index(raw_dir: Path, storage_dir: Path) -> None:
    configure_embedding()
    documents = load_markdown_documents(raw_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    splitter = SentenceSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
    )
    index.storage_context.persist(persist_dir=str(storage_dir))
