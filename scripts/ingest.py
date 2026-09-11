"""
scripts/ingest.py  ─  Ingestion script (Phase 1 + Phase 2)
═══════════════════════════════════════════════════════════
Supports two chunking strategies selectable via --strategy flag:

  --strategy fixed            Phase 1: fixed-size 500-token chunks (default)
  --strategy structure_aware  Phase 2: structure-aware chunks with metadata

Each strategy writes to its own index files so both can coexist:

  Phase 1 (fixed):
    data/index/chunks.json          ← Phase 1 chunk store
    data/index/faiss.index          ← Phase 1 FAISS index

  Phase 2 (structure_aware):
    data/index/chunks_structure_aware.json
    data/index/faiss_structure_aware.index

WHY SEPARATE FILES?
───────────────────
Keeping both indices lets compare_chunking.py load and query BOTH
strategies in the same run for a direct A/B comparison.
This is the whole point of Phase 2: measurable comparison.

Usage:
    cd duediligence-ai

    # Phase 1 (unchanged behavior):
    python scripts/ingest.py

    # Phase 2:
    python scripts/ingest.py --strategy structure_aware

    # Phase 2 with custom settings:
    python scripts/ingest.py --strategy structure_aware --chunk-size 400

    # Both (run twice):
    python scripts/ingest.py --strategy fixed
    python scripts/ingest.py --strategy structure_aware
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Make src/ importable ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding (cp1252 can't handle some Unicode chars)
import io as _io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Phase 1 imports ───────────────────────────────────────────────────────────
# NOTE: We import directly from the module file to avoid collision with the new
# src/ingestion/ package. The Phase 1 module is src/ingestion.py (a single file)
# while Phase 2 lives in src/ingestion/ (a package). Python resolves the PACKAGE
# first when both exist, so we import Phase 1's function using importlib.
import importlib.util as _ilu
_p1_spec = _ilu.spec_from_file_location(
    "src.ingestion_p1",
    PROJECT_ROOT / "src" / "ingestion.py",
)
_p1_mod = _ilu.module_from_spec(_p1_spec)
_p1_spec.loader.exec_module(_p1_mod)
p1_load = _p1_mod.load_documents_from_directory
from src.chunking        import chunk_documents                             # noqa: E402
from src.embeddings      import embed_texts                                 # noqa: E402
from src.vector_store    import build_index, save_index                     # noqa: E402

# ── Phase 2 imports (lazy to avoid import errors if pdfplumber missing) ───────
# Imported inside the function to give a clear error message.

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR  = PROJECT_ROOT / "data" / "raw"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the DueDiligence-AI vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py                              # Phase 1 fixed chunking
  python scripts/ingest.py --strategy structure_aware   # Phase 2 smart chunking
  python scripts/ingest.py --strategy fixed             # Phase 1 explicit
        """,
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "structure_aware"],
        default="fixed",
        help="Chunking strategy: 'fixed' (Phase 1) or 'structure_aware' (Phase 2).",
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
        help="Overlap tokens between consecutive chunks — Phase 1 only (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bar = "=" * 64
    phase_label = "Phase 1 — Fixed-size Chunking" if args.strategy == "fixed" \
                  else "Phase 2 — Structure-aware Chunking"

    print(f"\n{bar}")
    print(f"  DueDiligence-AI  |  {phase_label}")
    print(f"  Strategy: {args.strategy}   |   Chunk size: {args.chunk_size} tokens")
    print(bar)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.index_dir.mkdir(parents=True, exist_ok=True)

    if args.strategy == "fixed":
        _run_fixed_strategy(args)
    else:
        _run_structure_aware_strategy(args)


# ── Phase 1: fixed-size ───────────────────────────────────────────────────────

def _run_fixed_strategy(args: argparse.Namespace) -> None:
    """Exactly the original Phase 1 pipeline. Nothing changes here."""

    print(f"\n[STEP 1/5]  Loading PDFs from: {args.data_dir}")
    pages = p1_load(str(args.data_dir))
    if not pages:
        print(f"\n[ERROR] No text pages found in {args.data_dir}")
        sys.exit(1)
    print(f"  [OK] {len(pages)} pages loaded.")

    print(f"\n[STEP 2/5]  Chunking (size={args.chunk_size}, overlap={args.chunk_overlap})...")
    chunks = chunk_documents(pages, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    if not chunks:
        print("[ERROR] Chunking produced zero chunks.")
        sys.exit(1)
    print(f"  [OK] {len(chunks)} chunks created.")

    print(f"\n[STEP 3/5]  Generating embeddings for {len(chunks)} chunks...")
    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    print(f"\n  [OK] Embeddings shape: {embeddings.shape}  (dtype={embeddings.dtype})")

    print(f"\n[STEP 4/5]  Building FAISS index...")
    index = build_index(embeddings)
    index_path = args.index_dir / "faiss.index"
    save_index(index, args.index_dir)
    # Override filename for fixed strategy (no suffix needed)
    print(f"  [OK] Index saved to {index_path}")

    print(f"\n[STEP 5/5]  Saving chunk metadata...")
    chunks_path = args.index_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    print(f"  [OK] {len(chunks)} chunks saved to {chunks_path}")

    _print_summary("fixed", len(set(c["source"] for c in chunks)), len(pages), chunks, index)


# ── Phase 2: structure-aware ──────────────────────────────────────────────────

def _run_structure_aware_strategy(args: argparse.Namespace) -> None:
    """Phase 2 pipeline: parse → normalize → detect → chunk → validate → embed → index."""

    try:
        from src.ingestion.pdf_parser  import parse_pdf
        from src.ingestion.chunking    import build_structured_chunks
        from src.ingestion.validation  import validate_chunks, print_validation_summary
    except ImportError as exc:
        print(
            f"\n[ERROR] Phase 2 imports failed: {exc}\n"
            "  Make sure pdfplumber is installed: pip install pdfplumber>=0.11.0\n"
        )
        sys.exit(1)

    # ── Step 1: Parse PDFs ─────────────────────────────────────────────────────
    print(f"\n[STEP 1/6]  Parsing PDFs from: {args.data_dir}")

    pdf_files = sorted(args.data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[ERROR] No PDF files found in {args.data_dir}")
        sys.exit(1)

    print(f"  Found {len(pdf_files)} PDF file(s).")

    all_chunks_raw = []
    total_pages    = 0

    for pdf_path in pdf_files:
        doc = parse_pdf(pdf_path)
        total_pages += doc.total_pages

        # ── Step 2: Structure-aware chunking ──────────────────────────────────
        raw_chunks = build_structured_chunks(
            doc,
            target_tokens=args.chunk_size,
        )
        all_chunks_raw.extend(raw_chunks)

    print(f"  [OK] {total_pages} pages parsed, {len(all_chunks_raw)} raw chunks.")

    # ── Step 3: Validate ──────────────────────────────────────────────────────
    print(f"\n[STEP 3/6]  Validating {len(all_chunks_raw)} chunks...")
    valid_chunks, summary = validate_chunks(all_chunks_raw)
    print_validation_summary(summary)

    if not valid_chunks:
        print("[ERROR] No valid chunks after validation.")
        sys.exit(1)

    # ── Step 4: Convert to dicts for embedding and JSON ───────────────────────
    chunk_dicts = [c.to_dict() for c in valid_chunks]

    print(f"\n[STEP 4/6]  Generating embeddings for {len(chunk_dicts)} chunks...")
    texts      = [c["text"] for c in chunk_dicts]
    embeddings = embed_texts(texts)
    print(f"\n  [OK] Embeddings shape: {embeddings.shape}  (dtype={embeddings.dtype})")

    # ── Step 5: Build + save FAISS index ──────────────────────────────────────
    print(f"\n[STEP 5/6]  Building FAISS index...")
    index      = build_index(embeddings)
    index_path = args.index_dir / "faiss_structure_aware.index"

    import faiss
    faiss.write_index(index, str(index_path))
    print(f"  [OK] Index saved to {index_path}")

    # ── Step 6: Save chunk metadata ────────────────────────────────────────────
    print(f"\n[STEP 6/6]  Saving chunk metadata...")
    chunks_path = args.index_dir / "chunks_structure_aware.json"
    with open(chunks_path, "w", encoding="utf-8") as fh:
        json.dump(chunk_dicts, fh, ensure_ascii=False, indent=2)
    print(f"  [OK] {len(chunk_dicts)} chunks saved to {chunks_path}")

    _print_summary("structure_aware", len(pdf_files), total_pages, chunk_dicts, index)


def _print_summary(strategy, n_docs, n_pages, chunks, index) -> None:
    bar = "=" * 64
    print(f"\n{bar}")
    print("  Ingestion complete!")
    print(f"  Strategy  : {strategy}")
    print(f"  Documents : {n_docs}")
    print(f"  Pages     : {n_pages}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Vectors   : {index.ntotal}  (dim={index.d})")
    print()
    if strategy == "fixed":
        print("  To use Phase 2: python scripts/ingest.py --strategy structure_aware")
        print("  To compare    : python scripts/compare_chunking.py")
    else:
        print("  Inspect structure : python scripts/inspect_document.py <pdf>")
        print("  Inspect chunks    : python scripts/inspect_chunks.py --strategy structure_aware")
        print("  Compare strategies: python scripts/compare_chunking.py")
    print(bar + "\n")


if __name__ == "__main__":
    main()
