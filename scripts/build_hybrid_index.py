from pathlib import Path

from manufacturing_ai_copilot.core.config import (
    QDRANT_HYBRID_COLLECTION_NAME,
)
from manufacturing_ai_copilot.rag.hybrid_index_builder import (
    build_hybrid_index,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"

    point_count = build_hybrid_index(raw_dir)

    print(
        "Hybrid index built successfully: "
        f"collection='{QDRANT_HYBRID_COLLECTION_NAME}', "
        f"points={point_count}"
    )


if __name__ == "__main__":
    main()
