"""
scripts/inspect_retrieval.py  -  Interactive retrieval debugger
================================================================
Shows the full retrieval trace for a single query across all pipeline stages.

Usage:
    python scripts/inspect_retrieval.py "What was FY2024 revenue?" --mode HYBRID_RERANKED
    python scripts/inspect_retrieval.py "Sentinel Copilot" --mode KEYWORD_ONLY
    python scripts/inspect_retrieval.py "key risks" --mode HYBRID --filter-section "Risk"

Arguments:
    query        : The query string (positional)
    --mode       : VECTOR_ONLY | KEYWORD_ONLY | HYBRID | HYBRID_RERANKED (default: HYBRID_RERANKED)
    --top-k      : Final number of results (default: 5)
    --sem-k      : Semantic candidate pool size (default: 9 = all)
    --kw-k       : Keyword candidate pool size (default: 9 = all)
    --filter-section    : Only show chunks from sections matching this substring
    --filter-type       : Only show chunks of this content_type
    --filter-page-start : Only chunks starting at or after this page
    --filter-page-end   : Only chunks ending at or before this page
    --no-text    : Suppress chunk text (show metadata only)
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# ── Windows encoding fix ──────────────────────────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid  import HybridRetriever
from src.retrieval.models  import (
    VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED, RetrievalResult
)

# ── ANSI colors (disabled on Windows if not supported) ───────────────────────
_USE_COLOR = sys.stdout.isatty()
_RESET  = "\033[0m"    if _USE_COLOR else ""
_BOLD   = "\033[1m"    if _USE_COLOR else ""
_CYAN   = "\033[36m"   if _USE_COLOR else ""
_GREEN  = "\033[32m"   if _USE_COLOR else ""
_YELLOW = "\033[33m"   if _USE_COLOR else ""
_RED    = "\033[31m"   if _USE_COLOR else ""
_DIM    = "\033[2m"    if _USE_COLOR else ""


def _header(title: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n  {_BOLD}{title}{_RESET}\n{bar}"


def _section_header(n: int, title: str) -> str:
    return f"\n{_BOLD}{_CYAN}[{n}] {title}{_RESET}\n" + "-" * 60


def _fmt_score(label: str, value: float | None) -> str:
    if value is None:
        return f"{label}=N/A"
    return f"{label}={value:.4f}"


def _print_result_list(
    results: list[RetrievalResult],
    show_text: bool = True,
    text_preview: int = 120,
) -> None:
    if not results:
        print("  (no results)")
        return

    for r in results:
        scores = []
        if r.semantic_score is not None:
            scores.append(_fmt_score("sem", r.semantic_score))
        if r.keyword_score is not None:
            scores.append(_fmt_score("kw", r.keyword_score))
        if r.fusion_score is not None:
            scores.append(_fmt_score("rrf", r.fusion_score))
        if r.rerank_score is not None:
            scores.append(_fmt_score("rerank", r.rerank_score))

        ranks = []
        if r.semantic_rank is not None:
            ranks.append(f"sem_rank={r.semantic_rank}")
        if r.keyword_rank is not None:
            ranks.append(f"kw_rank={r.keyword_rank}")

        score_str = "  ".join(scores) if scores else ""
        rank_str  = "  ".join(ranks)  if ranks  else ""

        print(f"  {_BOLD}{_GREEN}{r.chunk_id}{_RESET}  {score_str}")
        print(f"      Section: {r.section or '(none)'}")
        print(f"      Pages: {r.page_start}-{r.page_end}  |  Type: {r.content_type}  |  {rank_str}")
        if show_text and r.text:
            preview = r.text.replace("\n", " ")[:text_preview]
            ellipsis = "..." if len(r.text) > text_preview else ""
            print(f"      {_DIM}\"{preview}{ellipsis}\"{_RESET}")
        print()


def _print_fused_table(results: list[RetrievalResult]) -> None:
    if not results:
        print("  (no results)")
        return
    print(f"  {'chunk_id':<14}  {'sem_rank':>8}  {'kw_rank':>7}  {'rrf_score':>10}")
    print("  " + "-" * 46)
    for r in results:
        sr  = str(r.semantic_rank) if r.semantic_rank else "   -"
        kr  = str(r.keyword_rank)  if r.keyword_rank  else "   -"
        rrf = f"{r.fusion_score:.6f}" if r.fusion_score else "   -"
        print(f"  {r.chunk_id:<14}  {sr:>8}  {kr:>7}  {rrf:>10}")


def run_inspection(args: argparse.Namespace) -> None:
    # Build filters dict
    filters = {}
    if args.filter_section:
        filters["section"] = args.filter_section
    if args.filter_type:
        filters["content_type"] = args.filter_type
    if args.filter_page_start is not None:
        filters["page_start"] = args.filter_page_start
    if args.filter_page_end is not None:
        filters["page_end"] = args.filter_page_end

    print(_header(f"RETRIEVAL INSPECTOR  --  Mode: {args.mode}"))
    print(f"\n  Query     : {_BOLD}{_YELLOW}{args.query}{_RESET}")
    print(f"  Mode      : {args.mode}")
    print(f"  Top-K     : {args.top_k}")
    print(f"  Sem-K     : {args.sem_k}")
    print(f"  KW-K      : {args.kw_k}")
    if filters:
        print(f"  Filters   : {filters}")

    # ── Build retriever ───────────────────────────────────────────────────────
    print("\n  [Loading retrieval components...]\n")
    retriever = HybridRetriever(
        mode           = args.mode,
        semantic_top_k = args.sem_k,
        keyword_top_k  = args.kw_k,
        final_top_k    = args.top_k,
        filters        = filters,
        index_dir      = str(PROJECT_ROOT / "data" / "index"),
    )

    # ── Run retrieval ─────────────────────────────────────────────────────────
    result = retriever.search(args.query)

    show_text = not args.no_text

    # ── Section 1: Semantic results ───────────────────────────────────────────
    if args.mode != KEYWORD_ONLY:
        sem_results = result.debug.get("semantic_results", [])
        print(_section_header(1, f"SEMANTIC RESULTS  ({len(sem_results)} candidates)"))
        _print_result_list(sem_results, show_text=show_text)

    # ── Section 2: Keyword results ────────────────────────────────────────────
    if args.mode != VECTOR_ONLY:
        kw_results = result.debug.get("keyword_results", [])
        print(_section_header(2, f"KEYWORD (BM25) RESULTS  ({len(kw_results)} candidates)"))
        _print_result_list(kw_results, show_text=show_text)

    # ── Section 3: Fused results ──────────────────────────────────────────────
    if args.mode in {HYBRID, HYBRID_RERANKED}:
        fused = result.debug.get("fused_results", [])
        print(_section_header(3, f"FUSED RESULTS (RRF k=60)  ({len(fused)} candidates)"))
        _print_fused_table(fused)

    # ── Section 3b: After metadata filter ────────────────────────────────────
    if filters:
        filtered = result.debug.get("filtered_results", [])
        n_removed = len(result.debug.get("fused_results") or result.debug.get("semantic_results") or []) - len(filtered)
        print(_section_header("3b", f"AFTER METADATA FILTER  ({len(filtered)} remain, {n_removed} removed)"))
        _print_result_list(filtered, show_text=False)

    # ── Section 4: Reranked results ────────────────────────────────────────────
    if args.mode == HYBRID_RERANKED:
        reranked = result.debug.get("reranked_results", [])
        print(_section_header(4, f"RERANKED RESULTS  (top {len(reranked)})"))

        # Show movement vs fused
        fused     = result.debug.get("filtered_results") or result.debug.get("fused_results") or []
        fused_ids = [r.chunk_id for r in fused]

        for rank, r in enumerate(reranked, start=1):
            old_pos = fused_ids.index(r.chunk_id) + 1 if r.chunk_id in fused_ids else None
            delta_str = ""
            if old_pos is not None:
                d = old_pos - rank
                if d > 0:
                    delta_str = f"  {_GREEN}[UP {d}]{_RESET}"
                elif d < 0:
                    delta_str = f"  {_RED}[DOWN {abs(d)}]{_RESET}"
                else:
                    delta_str = "  [same]"

            print(f"  Rank {rank:2d}  {_BOLD}{r.chunk_id}{_RESET}  rerank={r.rerank_score:.4f}{delta_str}")
            print(f"         Section: {r.section}")
            if show_text and r.text:
                preview = r.text.replace("\n", " ")[:100]
                print(f"         {_DIM}{preview}...{_RESET}")
            print()

    # ── Section 5: Final context ───────────────────────────────────────────────
    print(_section_header(5, f"FINAL CONTEXT  ({len(result.results)} chunks -> LLM)"))
    for rank, r in enumerate(result.results, start=1):
        best = r.best_score()
        print(f"\n  [{rank}] chunk_id={_BOLD}{r.chunk_id}{_RESET}  score={best:.4f}  type={r.content_type}  pages={r.page_start}-{r.page_end}")
        print(f"       Section: {r.section}")
        if show_text and r.text:
            lines = r.text.split("\n")[:5]
            for line in lines:
                print(f"       {line[:100]}")
            if len(r.text.split("\n")) > 5:
                print(f"       ...")

    # ── Section 6: Latency breakdown ──────────────────────────────────────────
    lat = result.latency
    print(_section_header(6, "LATENCY BREAKDOWN"))
    if lat.embedding_ms:
        print(f"  Query embedding   : {lat.embedding_ms:.1f} ms")
    if lat.semantic_ms:
        print(f"  FAISS search      : {lat.semantic_ms:.1f} ms")
    if lat.keyword_ms:
        print(f"  BM25 search       : {lat.keyword_ms:.1f} ms")
    if lat.fusion_ms:
        print(f"  RRF fusion        : {lat.fusion_ms:.1f} ms")
    if lat.filter_ms:
        print(f"  Metadata filter   : {lat.filter_ms:.1f} ms")
    if lat.rerank_ms:
        print(f"  Cross-encoder     : {lat.rerank_ms:.1f} ms")
    print(f"  TOTAL             : {_BOLD}{lat.total_ms:.1f} ms{_RESET}")

    print("\n" + "=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect retrieval trace for a single query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="The query string to inspect")
    parser.add_argument("--mode", default=HYBRID_RERANKED,
                        choices=[VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sem-k", type=int, default=9,
                        help="Semantic candidate pool size")
    parser.add_argument("--kw-k",  type=int, default=9,
                        help="Keyword candidate pool size")
    parser.add_argument("--filter-section", default=None)
    parser.add_argument("--filter-type",    default=None)
    parser.add_argument("--filter-page-start", type=int, default=None)
    parser.add_argument("--filter-page-end",   type=int, default=None)
    parser.add_argument("--no-text", action="store_true",
                        help="Suppress chunk text previews")

    args = parser.parse_args()
    run_inspection(args)


if __name__ == "__main__":
    main()
