"""
CLI tool for batch-ingesting documents into the vector store before
starting the API server. Useful for pre-loading a large corpus.

Usage:
    python scripts/ingest.py report.pdf
    python scripts/ingest.py docs/               # all supported files under docs/
    python scripts/ingest.py --config path/to/config.yaml report.pdf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.ingestion.chunker import split_documents
from src.ingestion.loader import SUPPORTED, load_file
from src.retrieval.embedder import get_embeddings
from src.retrieval.store import DocumentStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Loading embedding model…")
    embeddings = get_embeddings(cfg["embeddings"]["model"])
    store = DocumentStore(cfg["storage"]["vector_store_path"], embeddings)

    target = Path(args.path)
    files: list[Path] = []
    if target.is_dir():
        for ext in SUPPORTED:
            files.extend(sorted(target.rglob(f"*{ext}")))
    elif target.is_file():
        files = [target]
    else:
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print(f"No supported files found. Supported types: {sorted(SUPPORTED)}")
        sys.exit(0)

    for fpath in files:
        try:
            print(f"Ingesting {fpath.name}…", end=" ", flush=True)
            docs = load_file(str(fpath))
            chunks = split_documents(
                docs,
                chunk_size=cfg["retrieval"]["chunk_size"],
                chunk_overlap=cfg["retrieval"]["chunk_overlap"],
            )
            doc_id = store.add_document(chunks, fpath.name)
            print(f"done  ({len(chunks)} chunks, id={doc_id[:8]}…)")
        except Exception as exc:
            print(f"FAILED — {exc}")

    total = len(store.list_documents())
    print(f"\nTotal documents in store: {total}")


if __name__ == "__main__":
    main()
