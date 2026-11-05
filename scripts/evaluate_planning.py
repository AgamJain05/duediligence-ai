"""
scripts/evaluate_planning.py  ─  Phase 5 Planning Evaluation
═════════════════════════════════════════════════════════════
Runs the planning evaluation over data/evaluation/planning_questions.json
and reports the five planning metrics.

METRICS
────────
1. Classification Accuracy
   Percentage of queries correctly classified by type.
   Uses expected_type from JSON ground truth.

2. Task Count Accuracy
   Percentage of plans where task count falls within [min_expected, max_expected].

3. Decomposition Routing Accuracy
   SIMPLE queries → simple path (requires_decomposition=False).
   Complex queries → complex path (requires_decomposition=True).

4. Plan Validity Rate
   Percentage of plans that pass validation (no cycles, no duplicates, etc).

5. Over-decomposition Rate
   Percentage of plans with more tasks than maximum_expected_tasks.
   HIGH = planner is over-splitting queries unnecessarily.

USAGE:
    python scripts/evaluate_planning.py
    python scripts/evaluate_planning.py --type COMPARISON
    python scripts/evaluate_planning.py --no-llm
    python scripts/evaluate_planning.py --save
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
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
from src.planning.models import PlanningConfig, QueryType

QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "planning_questions.json"
RESULTS_DIR    = PROJECT_ROOT / "data" / "evaluation"
BAR = "=" * 72


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 planning quality.")
    parser.add_argument("--type", choices=[t.value for t in QueryType], default=None,
                        help="Filter by expected query type")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--save",   action="store_true", help="Save results JSON")
    parser.add_argument("--verbose", action="store_true", help="Show per-question details")
    return parser.parse_args()


def run_evaluation(questions: list, config: PlanningConfig, verbose: bool) -> list:
    analyzer  = QueryAnalyzer(config)
    planner   = ResearchPlanner(config)
    validator = PlanValidator(config)

    results = []
    for q in questions:
        if "_comment" in q:
            continue
        query = q.get("question", "")
        if not query:
            continue

        t0       = time.perf_counter()
        analysis = analyzer.analyze(query)
        plan_raw = planner.plan(analysis)
        try:
            plan     = validator.validate_and_correct(plan_raw)
            plan_ok  = True
        except PlanValidationError:
            plan     = plan_raw
            plan_ok  = False
        latency_ms = (time.perf_counter() - t0) * 1000

        expected_type = q.get("expected_type", "SIMPLE")
        min_tasks     = q.get("minimum_expected_tasks", 1)
        max_tasks     = q.get("maximum_expected_tasks", 6)
        actual_tasks  = len(plan.tasks)

        type_correct     = analysis.query_type.value == expected_type
        task_count_ok    = min_tasks <= actual_tasks <= max_tasks
        over_decomposed  = actual_tasks > max_tasks
        # Routing correctness: SIMPLE should NOT decompose, others should
        expected_decompose = expected_type != "SIMPLE"
        routing_correct    = analysis.requires_decomposition == expected_decompose

        row = {
            "id":                q.get("id", "?"),
            "question":          query[:60],
            "expected_type":     expected_type,
            "actual_type":       analysis.query_type.value,
            "type_correct":      type_correct,
            "routing_correct":   routing_correct,
            "min_tasks":         min_tasks,
            "max_tasks":         max_tasks,
            "actual_tasks":      actual_tasks,
            "task_count_ok":     task_count_ok,
            "over_decomposed":   over_decomposed,
            "plan_valid":        plan_ok,
            "confidence":        analysis.confidence,
            "classifier_method": analysis.classifier_method,
            "latency_ms":        latency_ms,
        }
        results.append(row)

        if verbose:
            status = "✓" if type_correct and task_count_ok and plan_ok else "✗"
            print(
                f"  {status} [{row['id']:4s}] "
                f"expect={expected_type:14s}  got={analysis.query_type.value:14s}  "
                f"tasks={actual_tasks}(exp {min_tasks}–{max_tasks})  "
                f"route={'✓' if routing_correct else '✗'}  "
                f"{'ok' if plan_ok else 'INVALID'}"
            )

    return results


def print_report(results: list, config: PlanningConfig) -> None:
    n = len(results)
    if n == 0:
        print("No results.")
        return

    type_acc   = sum(1 for r in results if r["type_correct"])   / n
    task_acc   = sum(1 for r in results if r["task_count_ok"])  / n
    route_acc  = sum(1 for r in results if r["routing_correct"]) / n
    plan_valid = sum(1 for r in results if r["plan_valid"])      / n
    over_dec   = sum(1 for r in results if r["over_decomposed"]) / n
    avg_lat    = sum(r["latency_ms"] for r in results) / n
    avg_tasks  = sum(r["actual_tasks"] for r in results) / n

    print(f"\n{BAR}")
    print("  PLANNING EVALUATION RESULTS")
    print(BAR)
    print(f"  {'Metric':<40} {'Value':>8}")
    print(f"  {'─'*40}   {'─'*8}")
    print(f"  {'Classification Accuracy':40} {type_acc:>8.1%}")
    print(f"  {'Routing Accuracy (simple/complex)':40} {route_acc:>8.1%}")
    print(f"  {'Task Count Accuracy':40} {task_acc:>8.1%}")
    print(f"  {'Plan Validity Rate':40} {plan_valid:>8.1%}")
    print(f"  {'Over-decomposition Rate':40} {over_dec:>8.1%}  ← lower is better")
    print(f"  {'Average Task Count':40} {avg_tasks:>8.1f}")
    print(f"  {'Average Planning Latency':40} {avg_lat:>7.0f}ms")
    print(f"  {'Total Questions':40} {n:>8}")

    # By type breakdown
    print(f"\n  {'─'*60}")
    print(f"  By Query Type:")
    types = sorted({r["expected_type"] for r in results})
    for qt in types:
        subset = [r for r in results if r["expected_type"] == qt]
        ta = sum(1 for r in subset if r["type_correct"]) / len(subset)
        print(f"    {qt:16s}: {len(subset):2d} questions  type_acc={ta:.0%}")

    print(f"\n{BAR}\n")


def main() -> None:
    args = parse_args()

    config = PlanningConfig(
        allow_llm_classifier=not args.no_llm,
        allow_llm_planner=not args.no_llm,
    )

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Filter by type
    if args.type:
        questions = [q for q in questions if q.get("expected_type") == args.type]
        questions = [q for q in questions if "question" in q]

    print(f"\n{BAR}")
    print("  Phase 5 Planning Evaluation — DueDiligence-AI")
    print(f"  LLM: {'enabled' if not args.no_llm else 'disabled'}   Questions: {len([q for q in questions if 'question' in q])}")
    print(BAR)

    results = run_evaluation(questions, config, verbose=args.verbose)
    print_report(results, config)

    if args.save and results:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"planning_eval_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
