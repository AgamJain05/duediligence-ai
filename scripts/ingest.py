"""
scripts/ingest.py  ─  One-time ingestion script
════════════════════════════════════════════════
Run this ONCE (or whenever documents change) to:

  1. Load all PDFs from  data/raw/
  2. Extract text page by page
  3. Chunk text into fixed-size pieces (500 tokens, 50 token overlap)
  4. Generate embeddings for every chunk  (BGE-base-en-v1.5, runs locally)
  5. Build a FAISS IndexFlatIP  (cosine similarity)
  6. Save the index to  data/index/faiss.index
  7. Save chunk metadata to  data/index/chunks.json

After this script completes, app.py can answer questions without
re-embedding anything.

Usage:
    cd duediligence-ai
    python scripts/ingest.py

    # Or point at a custom directory:
    python scripts/ingest.py --data-dir /path/to/pdfs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Make sure we can import from src/ regardless of where we run from ─────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion   import load_documents_from_directory
from src.chunking    import chunk_documents
from src.embeddings  import embed_texts
from src.vector_store import build_index, save_index

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR  = PROJECT_ROOT / "data" / "raw"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"
CHUNKS_FILE       = DEFAULT_INDEX_DIR / "chunks.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the DueDiligence-AI vector store."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing PDF files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"Where to save the FAISS index and chunks (default: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target tokens per chunk (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap tokens between consecutive chunks (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bar = "=" * 62
    print(f"\n{bar}")
    print("  DueDiligence-AI  |  Phase 1 Ingestion")
    print(bar)

    # ── Step 1: Load documents ─────────────────────────────────────────────────
    print(f"\n[STEP 1/5]  Loading PDFs from: {args.data_dir}")
    args.data_dir.mkdir(parents=True, exist_ok=True)

    pages = load_documents_from_directory(str(args.data_dir))
    if not pages:
        print(
            "\n[ERROR] No text pages extracted.\n"
            f"  -> Place PDF files in: {args.data_dir}\n"
            "  -> Then re-run this script.\n"
        )
        sys.exit(1)

    print(f"  [OK] {len(pages)} pages loaded.")

    # ── Step 2: Chunk ──────────────────────────────────────────────────────────
    print(
        f"\n[STEP 2/5]  Chunking "
        f"(size={args.chunk_size} tokens, overlap={args.chunk_overlap} tokens)..."
    )
    chunks = chunk_documents(
        pages,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    if not chunks:
        print("[ERROR] Chunking produced zero chunks. Check your PDFs.")
        sys.exit(1)

    print(f"  [OK] {len(chunks)} chunks created.")

    # ── Step 3: Embed ──────────────────────────────────────────────────────────
    print(
        f"\n[STEP 3/5]  Generating embeddings for {len(chunks)} chunks...\n"
        "  (BGE-base-en-v1.5 runs locally; first run downloads ~438 MB)\n"
    )
    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    print(f"\n  [OK] Embeddings shape: {embeddings.shape}  (dtype={embeddings.dtype})")

    # ── Step 4: Build + save FAISS index ──────────────────────────────────────
    print(f"\n[STEP 4/5]  Building FAISS index...")
    index = build_index(embeddings)
    save_index(index, args.index_dir)
    print(f"  [OK] Index saved to {args.index_dir / 'faiss.index'}")

    # ── Step 5: Save chunk metadata ────────────────────────────────────────────
    print(f"\n[STEP 5/5]  Saving chunk metadata...")
    chunks_path = args.index_dir / "chunks.json"
    args.index_dir.mkdir(parents=True, exist_ok=True)

    with open(chunks_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)

    print(f"  [OK] {len(chunks)} chunks saved to {chunks_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{bar}")
    print("  Ingestion complete!")
    print(f"  Documents : {len(set(c['source'] for c in chunks))}")
    print(f"  Pages     : {len(pages)}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Vectors   : {index.ntotal}  (dim={embeddings.shape[1]})")
    print("")
    print("  Run the app:  python app.py")
    print(bar + "\n")


if __name__ == "__main__":
    main()
