from pathlib import Path

from manufacturing_ai_copilot.rag.index_builder import build_index


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    storage_dir = project_root / "storage"

    build_index(raw_dir=raw_dir, storage_dir=storage_dir)
    print(f"Index built successfully: {storage_dir}")


if __name__ == "__main__":
    main()
