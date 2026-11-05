"""
scripts/inspect_plan.py  ─  Phase 5 Plan Inspector
════════════════════════════════════════════════════
Shows query classification + research plan WITHOUT executing retrieval.

Use this to understand:
  • How a query is classified
  • Whether the planner routes simple vs complex
  • What sub-queries are generated
  • What metadata filters are proposed
  • What dependencies exist between tasks

Usage:
    python scripts/inspect_plan.py "What is OrionVault's FY2025 revenue?"
    python scripts/inspect_plan.py "Compare US and European strategies and risks."
    python scripts/inspect_plan.py "How did the risk profile change from 2024 to 2026?"
    python scripts/inspect_plan.py --no-llm "complex query"   # rules only
    python scripts/inspect_plan.py --batch                    # run all questions in planning_questions.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.planning.analyzer import QueryAnalyzer
from src.planning.planner import ResearchPlanner
from src.planning.validator import PlanValidator, PlanValidationError
from src.planning.models import PlanningConfig

QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "planning_questions.json"
BAR = "=" * 68


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Phase 5 query classification and research plan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", default=None, help="Query to inspect")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM classifier and planner")
    parser.add_argument("--batch", action="store_true", help="Run all planning_questions.json")
    parser.add_argument("--type", choices=["SIMPLE","MULTI_HOP","COMPARISON","TEMPORAL","CROSS_DOCUMENT","MULTI_TOPIC"],
                        help="Filter batch by query type")
    parser.add_argument("--json-out", action="store_true", help="Output JSON instead of human-readable")
    return parser.parse_args()


def _print_analysis(analysis) -> None:
    print(f"\n{'─'*68}")
    print(f"  QUERY CLASSIFICATION")
    print(f"{'─'*68}")
    print(f"  Query      : {analysis.original_query}")
    print(f"  Type       : {analysis.query_type.value}")
    print(f"  Decompose? : {'YES → complex path' if analysis.requires_decomposition else 'NO → simple path'}")
    print(f"  Confidence : {analysis.confidence:.0%}")
    print(f"  Method     : {analysis.classifier_method}")
    print(f"  Reasoning  : {analysis.classifier_reasoning}")
    if analysis.entities:
        print(f"  Entities   : {', '.join(analysis.entities)}")
    if analysis.topics:
        print(f"  Topics     : {', '.join(analysis.topics)}")
    if analysis.temporal_constraints:
        print(f"  Temporal   : {', '.join(analysis.temporal_constraints)}")
    if analysis.document_type_hints:
        print(f"  Doc hints  : {', '.join(analysis.document_type_hints)}")
    print(f"  Est. tasks : {analysis.expected_subquery_count}")


def _print_plan(plan, analysis) -> None:
    route = "COMPLEX (multi-step)" if analysis.requires_decomposition else "SIMPLE (direct retrieval)"
    print(f"\n{'─'*68}")
    print(f"  RESEARCH PLAN  [{route}]")
    print(f"{'─'*68}")
    print(f"  Plan ID   : {plan.plan_id}")
    print(f"  Objective : {plan.objective}")
    print(f"  Tasks     : {len(plan.tasks)}")
    if plan.notes:
        print(f"  Notes     : {plan.notes}")

    for task in plan.tasks:
        dep_str = ", ".join(task.depends_on) if task.depends_on else "none (independent)"
        print(f"\n  ┌─ {task.task_id.upper()}")
        print(f"  │  Description : {task.description}")
        print(f"  │  Purpose     : {task.purpose or '—'}")
        print(f"  │  Sub-query   : {task.query}")
        print(f"  │  Filters     : {task.filters or '(none)'}")
        print(f"  │  Top-K       : {task.top_k}")
        print(f"  └─ Depends on  : {dep_str}")

    # Parallel / dependent breakdown
    independent = [t.task_id for t in plan.tasks if not t.depends_on]
    dependent   = [t.task_id for t in plan.tasks if t.depends_on]
    print(f"\n  Parallelizable : {', '.join(independent) if independent else '—'}")
    if dependent:
        print(f"  Sequential     : {', '.join(dependent)}")


def inspect_query(query: str, config: PlanningConfig) -> None:
    print(f"\n{BAR}")
    print("  Phase 5 Plan Inspector — DueDiligence-AI")
    print(BAR)

    analyzer  = QueryAnalyzer(config)
    planner   = ResearchPlanner(config)
    validator = PlanValidator(config)

    analysis = analyzer.analyze(query)
    _print_analysis(analysis)

    plan = planner.plan(analysis)
    try:
        plan = validator.validate_and_correct(plan)
    except PlanValidationError as e:
        print(f"\n  [VALIDATION ERROR] {e}")
        return

    _print_plan(plan, analysis)
    print(f"\n{BAR}\n")


def inspect_batch(config: PlanningConfig, type_filter: str | None) -> None:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = [q for q in json.load(f) if "question" in q]

    if type_filter:
        questions = [q for q in questions if q.get("expected_type") == type_filter]

    print(f"\n{BAR}")
    print(f"  Batch Plan Inspection — {len(questions)} questions")
    print(BAR)

    analyzer  = QueryAnalyzer(config)
    planner   = ResearchPlanner(config)
    validator = PlanValidator(config)

    results = []
    for q in questions:
        query    = q["question"]
        analysis = analyzer.analyze(query)
        plan     = planner.plan(analysis)
        try:
            plan = validator.validate_and_correct(plan)
            plan_ok = True
        except PlanValidationError:
            plan_ok = False

        expected_type = q.get("expected_type", "")
        type_match    = analysis.query_type.value == expected_type
        min_tasks     = q.get("minimum_expected_tasks", 1)
        max_tasks     = q.get("maximum_expected_tasks", 99)
        task_ok       = min_tasks <= len(plan.tasks) <= max_tasks

        status = "✓" if type_match and task_ok and plan_ok else "✗"
        print(
            f"  {status} [{q['id']:4s}] "
            f"expect={expected_type:14s}  got={analysis.query_type.value:14s}  "
            f"tasks={len(plan.tasks)} (exp {min_tasks}–{max_tasks})  "
            f"{'plan_ok' if plan_ok else 'PLAN_FAIL'}"
        )
        results.append({
            "id": q["id"], "type_match": type_match,
            "task_ok": task_ok, "plan_ok": plan_ok,
        })

    n = len(results)
    type_acc  = sum(1 for r in results if r["type_match"]) / n
    task_acc  = sum(1 for r in results if r["task_ok"])   / n
    plan_acc  = sum(1 for r in results if r["plan_ok"])   / n

    print(f"\n  {'─'*48}")
    print(f"  Type accuracy  : {type_acc:.0%}  ({int(type_acc*n)}/{n})")
    print(f"  Task count OK  : {task_acc:.0%}  ({int(task_acc*n)}/{n})")
    print(f"  Plan valid     : {plan_acc:.0%}  ({int(plan_acc*n)}/{n})")
    print(f"\n{BAR}\n")


def main() -> None:
    args = parse_args()

    config = PlanningConfig(
        allow_llm_classifier=not args.no_llm,
        allow_llm_planner=not args.no_llm,
    )

    if args.batch:
        inspect_batch(config, args.type)
    elif args.query:
        inspect_query(args.query, config)
    else:
        print("Usage: inspect_plan.py <query>  or  --batch")
        sys.exit(1)


if __name__ == "__main__":
    main()
