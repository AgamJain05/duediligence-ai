"""
scripts/inspect_research.py  ─  Phase 5 Full Research Trace
════════════════════════════════════════════════════════════
Runs the complete Phase 5 pipeline and shows every step.

Output includes:
  • Query classification + confidence
  • Routing decision (simple vs complex)
  • Research plan (tasks, sub-queries, dependencies)
  • Task execution (query, filters, latency, chunk count per task)
  • Evidence summary per task (top 2 chunks)
  • Evidence deduplication count
  • Final structured context (truncated for display)
  • Final LLM answer
  • Full latency breakdown
  • Phase 4 vs Phase 5 comparison hint

Usage:
    python scripts/inspect_research.py "What is FY2025 revenue?"
    python scripts/inspect_research.py "Compare US and European growth strategies."
    python scripts/inspect_research.py "How did risk profile change from 2024 to 2026?"
    python scripts/inspect_research.py --query "..." --no-answer  # skip LLM generation
    python scripts/inspect_research.py --query "..." --pg          # use PostgreSQL
    python scripts/inspect_research.py --query "..." --faiss       # use FAISS (Phase 3)
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.planning.models import PlanningConfig
from src.orchestration.graph import ResearchGraph
from src.orchestration.state import initial_state

BAR  = "=" * 68
DASH = "─" * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full Phase 5 research pipeline trace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--query", dest="query_flag", default=None)
    parser.add_argument("--no-answer", action="store_true", help="Skip LLM generation")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM classifier/planner")
    parser.add_argument("--pg", action="store_true", help="Use PostgreSQL backend")
    parser.add_argument("--faiss", action="store_true", help="Force FAISS backend")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def _get_db_session():
    try:
        from src.db.connection import get_db, ping_db
        if ping_db():
            ctx = get_db()
            return ctx.__enter__(), ctx
    except Exception:
        pass
    return None, None


def print_section(title: str) -> None:
    print(f"\n{BAR}")
    print(f"  {title}")
    print(BAR)


def print_dict_rows(d: dict, indent: int = 2) -> None:
    pad = " " * indent
    for k, v in d.items():
        print(f"{pad}{k:25s}: {v}")


def main() -> None:
    args  = parse_args()
    query = args.query or args.query_flag

    if not query:
        print("Usage: inspect_research.py <query>  or  --query <query>")
        sys.exit(1)

    # ── Config ────────────────────────────────────────────────────────────────
    config = PlanningConfig(
        max_task_retrieval    = args.top_k,
        allow_llm_classifier  = not args.no_llm,
        allow_llm_planner     = not args.no_llm,
    )

    # ── Backend selection ─────────────────────────────────────────────────────
    db_session = None
    db_ctx     = None

    if not args.faiss:
        db_session, db_ctx = _get_db_session()
        if db_session is None:
            print("  [INFO] PostgreSQL unavailable — falling back to FAISS.")

    print(f"\n{BAR}")
    print("  Phase 5 Research Inspector — DueDiligence-AI")
    print(BAR)
    print(f"  Query   : {query}")
    print(f"  Backend : {'PostgreSQL' if db_session else 'FAISS (Phase 3)'}")
    print(f"  LLM     : {'enabled' if not args.no_llm else 'disabled (rules only)'}")

    # ── Run graph with node-by-node streaming ─────────────────────────────────
    rg    = ResearchGraph(config=config, db_session=db_session)
    t_all = time.perf_counter()

    print_section("STREAMING NODE EXECUTION")
    state = initial_state(query)
    for event in rg.stream(query):
        for node_name, node_output in event.items():
            if isinstance(node_output, dict):
                state.update(node_output)
            keys = list(node_output.keys()) if isinstance(node_output, dict) else []
            print(f"  [node: {node_name}]  updated keys: {keys}")

    total_ms = (time.perf_counter() - t_all) * 1000

    # ── Analysis ──────────────────────────────────────────────────────────────
    print_section("QUERY ANALYSIS")
    analysis = state.get("analysis", {})
    print_dict_rows({
        "query_type":             analysis.get("query_type", "?"),
        "requires_decomposition": analysis.get("requires_decomposition", "?"),
        "confidence":             f"{analysis.get('confidence', 0):.0%}",
        "method":                 analysis.get("classifier_method", "?"),
        "reasoning":              analysis.get("classifier_reasoning", "?"),
        "entities":               ", ".join(analysis.get("entities", []) or []) or "—",
        "topics":                 ", ".join(analysis.get("topics", []) or []) or "—",
        "temporal":               ", ".join(analysis.get("temporal_constraints", []) or []) or "—",
    })

    # ── Route ─────────────────────────────────────────────────────────────────
    route = state.get("route", "simple")
    print_section(f"ROUTING DECISION: {route.upper()} PATH")
    if route == "simple":
        print("  → Direct retrieval. No planning, no decomposition.")
    else:
        print("  → Multi-step: plan → execute tasks → aggregate → generate.")

    # ── Research plan ─────────────────────────────────────────────────────────
    plan_dict = state.get("research_plan")
    if plan_dict and route == "complex":
        print_section("RESEARCH PLAN")
        print(f"  Objective : {plan_dict.get('objective', '—')}")
        tasks = plan_dict.get("tasks", [])
        print(f"  Tasks     : {len(tasks)}")
        for t in tasks:
            deps = ", ".join(t.get("depends_on", [])) or "none (parallel)"
            print(f"\n  ┌─ {t.get('task_id','?').upper()}: {t.get('description','')}")
            print(f"  │  Sub-query : {t.get('query','')}")
            print(f"  │  Filters   : {t.get('filters') or '(none)'}")
            print(f"  │  Top-K     : {t.get('top_k', 8)}")
            print(f"  └─ Deps      : {deps}")

    # ── Task execution ────────────────────────────────────────────────────────
    task_results = state.get("task_results", {})
    evidence_by_task = state.get("evidence_by_task", {})
    if task_results:
        print_section("TASK EXECUTION RESULTS")
        tasks_meta = {}
        if plan_dict:
            tasks_meta = {t["task_id"]: t for t in plan_dict.get("tasks", [])}

        for tid, records in task_results.items():
            t_meta = tasks_meta.get(tid, {})
            status = t_meta.get("status", "DONE")
            lat    = t_meta.get("latency_ms", 0)
            err    = t_meta.get("error")
            err_str = f"  error={err}" if err else ""
            print(f"\n  Task: {tid}  status={status}  chunks={len(records)}  latency={lat:.0f}ms{err_str}")
            print(f"  Sub-query: {t_meta.get('query', '—')}")
            for j, rec in enumerate(records[:2], start=1):
                if isinstance(rec, dict):
                    sim = rec.get("semantic_score")
                    sim_str = f" sim={sim:.3f}" if sim else ""
                    print(f"    [{j}]{sim_str}  p{rec.get('page_start','?')}  {rec.get('text','')[:80].replace(chr(10),' ')}...")

    # ── Evidence summary ──────────────────────────────────────────────────────
    evidence = state.get("evidence", [])
    print_section("AGGREGATED EVIDENCE")
    print(f"  Total unique chunks: {len(evidence)}")
    duplicates_removed = sum(len(r) for r in task_results.values()) - len(evidence)
    if duplicates_removed > 0:
        print(f"  Duplicates removed : {duplicates_removed}")
    for i, rec in enumerate(evidence[:3], start=1):
        if isinstance(rec, dict):
            sim = rec.get("semantic_score")
            print(f"  [{i}] task={rec.get('task_id','?')}  "
                  f"doc={rec.get('document','?')[:30]}  "
                  f"p={rec.get('page_start','?')}  "
                  f"sim={sim:.3f if sim else '—'}")

    # ── Context preview ───────────────────────────────────────────────────────
    context = state.get("final_context", "")
    if context:
        print_section("FINAL CONTEXT (PREVIEW)")
        print(context[:1200])
        if len(context) > 1200:
            print(f"  ... [{len(context) - 1200} more characters] ...")

    # ── Answer ────────────────────────────────────────────────────────────────
    if not args.no_answer:
        answer = state.get("answer", "")
        if answer:
            print_section("FINAL ANSWER")
            print(answer)

    # ── Latency ───────────────────────────────────────────────────────────────
    print_section("LATENCY BREAKDOWN")
    latency = state.get("latency", {})
    for k, v in latency.items():
        print(f"  {k:30s}: {v:.0f}ms")
    print(f"  {'TOTAL (wall clock)':30s}: {total_ms:.0f}ms")

    # ── Errors ────────────────────────────────────────────────────────────────
    errors = state.get("errors", [])
    if errors:
        print_section("WARNINGS / ERRORS")
        for e in errors:
            print(f"  ⚠ {e}")

    print(f"\n{BAR}\n")

    if db_ctx:
        try:
            db_ctx.__exit__(None, None, None)
        except Exception:
            pass


if __name__ == "__main__":
    main()
