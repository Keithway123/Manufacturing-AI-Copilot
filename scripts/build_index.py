from pathlib import Path

from manufacturing_ai_copilot.core.config import QDRANT_COLLECTION_NAME
from manufacturing_ai_copilot.rag.index_builder import build_index


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"

    build_index(raw_dir=raw_dir)
    print(f"Index built successfully: " f"Qdrant collection '{QDRANT_COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
