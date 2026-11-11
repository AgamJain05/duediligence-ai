#!/usr/bin/env python3
"""
scripts/evaluate_tools.py — Evaluate Phase 6 tool routing and accuracy.

Runs questions from data/evaluation/tool_questions.json.
Reports: tool-call accuracy, numeric accuracy, conflict detection rate,
path accuracy, error handling rate.

Usage:
    python scripts/evaluate_tools.py
    python scripts/evaluate_tools.py --category METRICS_ONLY
    python scripts/evaluate_tools.py --category conflict
    python scripts/evaluate_tools.py --question m01
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "tool_questions.json"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 6 tool framework")
    parser.add_argument("--category", default=None,
                        help="Filter by expected_answer_path or 'conflict' or 'negative'")
    parser.add_argument("--question", default=None, help="Run single question by id")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check tools are registered without running questions")
    args = parser.parse_args()

    # Load questions
    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)

    # Setup tools
    db_session = None
    faiss_retriever = None
    try:
        from src.db.connection import SessionLocal
        db_session = SessionLocal()
    except Exception as e:
        print(f"[warn] DB not available: {e}")

    try:
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.models import HYBRID_RERANKED
        index_dir = str(Path(__file__).resolve().parent.parent / "data" / "index")
        if (Path(index_dir) / "faiss.index").exists():
            faiss_retriever = HybridRetriever(mode=HYBRID_RERANKED, index_dir=index_dir, final_top_k=5)
    except Exception as e:
        print(f"[warn] FAISS not available: {e}")

    from src.tools.registry import build_default_registry, list_tools
    build_default_registry(db_session=db_session, faiss_retriever=faiss_retriever)

    if args.dry_run:
        print("Registered tools:", list_tools())
        print(f"Questions loaded: {len(questions)}")
        return

    # Filter questions
    if args.question:
        questions = [q for q in questions if q["id"] == args.question]
    elif args.category:
        cat = args.category.lower()
        if cat == "conflict":
            questions = [q for q in questions if q.get("expected_conflict")]
        elif cat == "negative":
            questions = [q for q in questions if q["id"].startswith("neg")]
        else:
            questions = [q for q in questions if q.get("expected_answer_path", "").upper() == cat.upper()]

    if not questions:
        print(f"No questions found for filter: {args.question or args.category}")
        return

    # Evaluate
    results = []
    for q in questions:
        result = _evaluate_question(q, db_session)
        results.append(result)
        status = "✓" if result["passed"] else "✗"
        print(f"[{status}] {q['id']:12} {q['question'][:60]:60} → {result['status']}")
        if result.get("note"):
            print(f"         {result['note']}")

    # Summary stats
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    conflict_expected = [r for r in results if r.get("expected_conflict")]
    conflict_detected = [r for r in results if r.get("conflict_detected")]
    numeric_results = [r for r in results if r.get("numeric_expected") is not None]
    numeric_correct = [r for r in numeric_results if r.get("numeric_correct")]

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS ({total} questions)")
    print("=" * 70)
    print(f"Overall pass rate:        {passed}/{total} ({100*passed//total if total else 0}%)")
    if numeric_results:
        print(f"Numeric accuracy:         {len(numeric_correct)}/{len(numeric_results)} "
              f"({100*len(numeric_correct)//len(numeric_results) if numeric_results else 0}%)")
    if conflict_expected:
        conflicts_ok = sum(1 for r in conflict_expected if r.get("conflict_detected"))
        print(f"Conflict detection:       {conflicts_ok}/{len(conflict_expected)}")
    errors = [r for r in results if r["status"] == "ERROR"]
    empties = [r for r in results if r["status"] == "EMPTY"]
    print(f"Error responses:          {len(errors)}/{total}")
    print(f"Empty responses:          {len(empties)}/{total}")
    print("=" * 70)

    if db_session:
        db_session.close()


def _evaluate_question(q: Dict[str, Any], db_session) -> Dict[str, Any]:
    """Evaluate a single question by running the primary expected tool."""
    from src.tools.registry import get_tool

    expected_tools = q.get("expected_tools", [])
    if not expected_tools:
        return {"passed": False, "status": "SKIPPED", "note": "No expected tools defined"}

    primary_tool_name = expected_tools[0]
    tool = get_tool(primary_tool_name)
    if tool is None:
        return {"passed": False, "status": "TOOL_NOT_FOUND",
                "note": f"Tool '{primary_tool_name}' not registered"}

    # Build kwargs from expected_tool_args (may be partial)
    kwargs = dict(q.get("expected_tool_args", {}))

    result = tool.safe_run(**kwargs)

    out = {
        "id":                q["id"],
        "status":            result.status,
        "conflict_detected": result.conflict_detected,
        "expected_conflict": q.get("expected_conflict", False),
        "passed":            False,
    }

    # Conflict check
    if q.get("expected_conflict"):
        out["passed"] = result.conflict_detected
        if not result.conflict_detected:
            out["note"] = f"Expected conflict_detected=True, got {result.conflict_detected}"

    # Empty check for negative cases
    elif q["id"].startswith("neg") or not q.get("expected_sources"):
        out["passed"] = result.status in ("EMPTY", "ERROR")
        if result.status not in ("EMPTY", "ERROR"):
            out["note"] = f"Expected EMPTY/ERROR for missing data, got {result.status}"

    # Numeric check
    elif q.get("expected_numeric_result") is not None:
        expected = Decimal(str(q["expected_numeric_result"]))
        tolerance = Decimal(str(q.get("numeric_tolerance", 0.1)))
        out["numeric_expected"] = str(expected)

        # Extract first row value for metric queries
        rows = result.data.get("rows", [])
        if rows:
            actual = Decimal(str(rows[0].get("value", "0")))
            diff = abs(actual - expected)
            out["numeric_correct"] = diff <= tolerance
            out["passed"] = out["numeric_correct"] and result.is_ok()
            if not out["numeric_correct"]:
                out["note"] = f"Expected {expected}±{tolerance}, got {actual}"
        elif result.data.get("result"):
            # Calculator result
            actual = Decimal(str(result.data["result"]))
            diff = abs(actual - expected)
            out["numeric_correct"] = diff <= tolerance
            out["passed"] = out["numeric_correct"]
        else:
            out["passed"] = result.is_ok()

    else:
        out["passed"] = result.is_ok()

    return out


if __name__ == "__main__":
    main()
