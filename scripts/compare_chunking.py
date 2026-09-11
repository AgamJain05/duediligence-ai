"""
scripts/compare_chunking.py  ─  Phase 1 vs Phase 2 comparison
══════════════════════════════════════════════════════════════
Runs the same 10 due-diligence queries against BOTH chunking strategies
and prints a side-by-side result table.

WHAT THIS SCRIPT SHOWS YOU
───────────────────────────
For each of the 10 questions, you will see:
  - Top-3 retrieved chunks from Phase 1 (fixed)
  - Top-3 retrieved chunks from Phase 2 (structure-aware)
  - For each chunk: score, source, page, section (Phase 2 only), token count

WHAT TO LOOK FOR
────────────────
1. Does Phase 2 retrieve chunks that include the section heading context?
   (Better: "Section: Risk Factors > Supply Chain" is present in chunk text)

2. Does Phase 2 retrieve table chunks for financial questions?
   (Phase 1 might scatter table data across arbitrary token windows)

3. Are Phase 2 similarity scores comparable to Phase 1?
   (Ideally similar — we haven't changed the embedding model or FAISS)

4. Do Phase 2 chunks give MORE COMPLETE answers with less fragmentation?

PREREQUISITES
─────────────
Both indexes must exist. Run:
    python scripts/ingest.py --strategy fixed
    python scripts/ingest.py --strategy structure_aware

Usage:
    python scripts/compare_chunking.py
    python scripts/compare_chunking.py --top-k 5
    python scripts/compare_chunking.py --no-color
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings   import embed_query
from src.vector_store import search

DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"

# Fix Windows console encoding
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── 10 canonical evaluation questions ────────────────────────────────────────
EVAL_QUESTIONS = [
    "What was OrionVault's revenue in FY2024?",
    "What are the main risks related to customer concentration?",
    "Why did operating margin improve?",
    "What does Sentinel Copilot do?",
    "Which geographic market generates the most revenue?",
    "What happened to OrionVault in 2021?",
    "What are the company's main supply-chain risks?",
    "What is OrionVault's strategy for expanding into healthcare?",
    "What percentage of revenue came from its largest customer?",
    "What information is provided about future growth?",
]

# ── ANSI colors ───────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_DIM    = "\033[2m"
_BLUE   = "\033[94m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Phase 1 vs Phase 2 chunking on standard questions.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per question per strategy (default: 3)",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory containing index files (default: {DEFAULT_INDEX_DIR})",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output",
    )
    p.add_argument(
        "--text-length",
        type=int,
        default=250,
        help="Characters of chunk text to preview (default: 250)",
    )
    return p.parse_args()


def _c(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{code}{text}{_RESET}"


def _load_index_and_chunks(
    index_dir: Path,
    strategy: str,
) -> Optional[tuple]:
    """Load FAISS index and chunk list for a strategy. Returns None if not found."""
    import faiss

    if strategy == "fixed":
        idx_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.json"
    else:
        idx_path = index_dir / "faiss_structure_aware.index"
        chunks_path = index_dir / "chunks_structure_aware.json"

    if not idx_path.exists() or not chunks_path.exists():
        return None

    index  = faiss.read_index(str(idx_path))
    with open(chunks_path, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)

    return index, chunks


def _retrieve(
    query: str,
    index,
    chunks: List[Dict],
    top_k: int,
) -> List[Dict]:
    """Run a query against an index and return enriched result dicts."""
    query_vec    = embed_query(query)
    raw_results  = search(index, query_vec, top_k=top_k)
    results = []
    for faiss_idx, score in raw_results:
        if faiss_idx >= len(chunks):
            continue
        chunk = chunks[faiss_idx]
        results.append({
            "score":        score,
            "chunk_id":     chunk.get("chunk_id", "?"),
            "source":       chunk.get("source_filename") or chunk.get("source", "?"),
            "page_start":   chunk.get("page_start") or chunk.get("page_num", "?"),
            "page_end":     chunk.get("page_end", None),
            "section":      chunk.get("section", ""),
            "section_path": chunk.get("section_path") or [],
            "content_type": chunk.get("content_type", "text"),
            "tokens":       chunk.get("chunk_token_count", "?"),
            "text":         chunk.get("text", ""),
        })
    return results


def _print_chunk_result(
    rank: int,
    r: Dict,
    strategy: str,
    text_length: int,
    use_color: bool,
) -> None:
    """Print a single retrieved chunk in a readable format."""
    page_str = f"p{r['page_start']}"
    if r["page_end"] and r["page_end"] != r["page_start"]:
        page_str += f"–{r['page_end']}"

    score_color = _GREEN if r["score"] >= 0.70 else _YELLOW if r["score"] >= 0.50 else _RED

    print(
        f"    [{rank}] "
        + _c(f"score={r['score']:.4f}", score_color, use_color)
        + f"  |  {page_str}"
        + f"  |  {r['tokens']} tok"
        + (f"  |  {_c(r['content_type'], _CYAN, use_color)}" if strategy == "structure_aware" else "")
    )

    if strategy == "structure_aware" and r["section_path"]:
        path_str = " > ".join(r["section_path"])
        print(f"         " + _c(f"Section: {path_str}", _DIM + _CYAN, use_color))

    text_preview = r["text"].strip()[:text_length].replace("\n", " ↵ ")
    ellipsis = "…" if len(r["text"].strip()) > text_length else ""
    print(f"         {text_preview}{ellipsis}")


def main() -> None:
    args = parse_args()
    use_color = not args.no_color

    # ── Load both indexes ──────────────────────────────────────────────────────
    print(f"\nLoading indexes from {args.index_dir}...")

    result_fixed = _load_index_and_chunks(args.index_dir, "fixed")
    result_sa    = _load_index_and_chunks(args.index_dir, "structure_aware")

    if result_fixed is None:
        print(
            "[WARNING] Phase 1 (fixed) index not found.\n"
            "  Run: python scripts/ingest.py --strategy fixed"
        )
    if result_sa is None:
        print(
            "[WARNING] Phase 2 (structure_aware) index not found.\n"
            "  Run: python scripts/ingest.py --strategy structure_aware"
        )

    if result_fixed is None and result_sa is None:
        print("[ERROR] No indexes found. Run ingest.py for at least one strategy.")
        sys.exit(1)

    fixed_index, fixed_chunks = result_fixed if result_fixed else (None, [])
    sa_index, sa_chunks       = result_sa    if result_sa    else (None, [])

    if fixed_index:
        print(f"  ✓ Phase 1 (fixed):           {fixed_index.ntotal} vectors, {len(fixed_chunks)} chunks")
    if sa_index:
        print(f"  ✓ Phase 2 (structure_aware): {sa_index.ntotal} vectors, {len(sa_chunks)} chunks")

    print(f"\n{'═' * 72}")
    print(f"  CHUNKING COMPARISON  —  top_k={args.top_k}  |  {len(EVAL_QUESTIONS)} questions")
    print(f"{'═' * 72}")

    for q_idx, question in enumerate(EVAL_QUESTIONS, start=1):
        print(f"\n{'─' * 72}")
        print(
            _c(f"  Q{q_idx:02d}: {question}", _BOLD, use_color)
        )
        print(f"{'─' * 72}")

        # ── Phase 1 results ──────────────────────────────────────────────────
        if fixed_index:
            print(_c("\n  ── PHASE 1 (fixed-size chunks) ──", _YELLOW, use_color))
            fixed_results = _retrieve(question, fixed_index, fixed_chunks, args.top_k)
            if fixed_results:
                for rank, r in enumerate(fixed_results, start=1):
                    _print_chunk_result(rank, r, "fixed", args.text_length, use_color)
                    print()
            else:
                print("    (no results)")
        else:
            print("\n  [Phase 1 index not available]")

        # ── Phase 2 results ──────────────────────────────────────────────────
        if sa_index:
            print(_c("\n  ── PHASE 2 (structure-aware chunks) ──", _GREEN, use_color))
            sa_results = _retrieve(question, sa_index, sa_chunks, args.top_k)
            if sa_results:
                for rank, r in enumerate(sa_results, start=1):
                    _print_chunk_result(rank, r, "structure_aware", args.text_length, use_color)
                    print()
            else:
                print("    (no results)")
        else:
            print("\n  [Phase 2 index not available]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  COMPARISON COMPLETE")
    print(f"{'═' * 72}")
    print("""
  What to observe:
  ─────────────────────────────────────────────────────────────────────
  1. Phase 2 chunks include "Section: X > Y" showing WHERE in the document
     the content came from. Phase 1 chunks only show source + page number.

  2. Financial questions (Q1, Q3, Q9): does Phase 2 retrieve TABLE chunks?
     Look for content_type=table in Phase 2 results. Phase 1 may fragment
     tables across arbitrary token windows.

  3. Conceptual questions (Q2, Q4, Q7): Phase 2 should retrieve chunks
     that contain BOTH the heading AND the content — not just one or the other.

  4. Similarity scores: if Phase 2 scores are significantly lower, it may
     mean the chunks are too long (diluting the embedding signal) or that
     the section prefix text is dominating the embedding.

  5. For Q6 (2021 timeline): does either strategy retrieve the history/
     milestone section? If not, this reveals a retrieval gap to fix later.
  ─────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
