"""
src/orchestration/context_builder_p6.py — Phase 6 structured context builder.

Produces a context string with clearly separated sections:
  1. STRUCTURED DATA (from search_financial_data)
  2. COMPUTATIONS (from calculate_metric / compare_periods)
  3. DOCUMENT EVIDENCE (from search_documents)
  4. FAILED / EMPTY TASKS

The LLM is explicitly instructed to use ONLY numbers from STRUCTURED DATA
and COMPUTATIONS — never to do its own arithmetic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.planning.models_p6 import ToolAwareResearchPlan, ToolTask, ToolTaskType
from src.tools.base import ToolResult

_SEP = "═" * 68
_TASK_SEP = "─" * 60


def build_p6_context(
    query: str,
    plan: ToolAwareResearchPlan,
    task_results: Dict[str, ToolResult],
    max_chars: int = 12_000,
) -> str:
    """
    Build the Phase 6 LLM context string.

    Numbers in the answer MUST come from STRUCTURED DATA or COMPUTATIONS sections only.
    """
    parts = [_SEP, "RESEARCH QUESTION", query, _SEP, ""]

    metric_rows = []
    computation_rows = []
    doc_sections = []
    failed_tasks = []

    for task in plan.tasks:
        result = task_results.get(task.task_id)
        if result is None:
            failed_tasks.append((task, None))
            continue

        if not result.is_ok():
            failed_tasks.append((task, result))
            continue

        if task.task_type in (ToolTaskType.QUERY_METRICS,):
            _collect_metric_rows(task, result, metric_rows)

        elif task.task_type in (ToolTaskType.CALCULATE, ToolTaskType.COMPARE_PERIODS):
            _collect_computation(task, result, computation_rows)

        elif task.task_type == ToolTaskType.RETRIEVE_DOCUMENTS:
            _collect_doc_section(task, result, doc_sections)

    # Section 1: Structured data
    if metric_rows:
        parts.append("STRUCTURED DATA (from financial database)")
        parts.append(_TASK_SEP)
        for row_text in metric_rows:
            parts.append(row_text)
        parts.append("")

    # Section 2: Computations
    if computation_rows:
        parts.append("COMPUTATIONS (deterministic calculator)")
        parts.append(_TASK_SEP)
        for c in computation_rows:
            parts.append(c)
        parts.append("")

    # Section 3: Document evidence
    if doc_sections:
        parts.append("DOCUMENT EVIDENCE")
        parts.append(_TASK_SEP)
        for section_text in doc_sections:
            parts.append(section_text)
        parts.append("")

    # Section 4: Failures
    if failed_tasks:
        parts.append("FAILED / EMPTY TASKS")
        parts.append(_TASK_SEP)
        for task, result in failed_tasks:
            if result is None:
                parts.append(f"- {task.task_id} ({task.tool}): no result")
            else:
                err = result.error
                parts.append(f"- {task.task_id} ({task.tool}): {err.code if err else result.status} — {err.message if err else ''}")
        parts.append("")

    # Instructions
    parts.extend([
        _SEP, "INSTRUCTIONS",
        "Answer the research question using ONLY the evidence above.",
        "• Use ONLY numbers from STRUCTURED DATA or COMPUTATIONS. Do NOT invent or calculate your own numbers.",
        "• If a calculation was needed but is not in COMPUTATIONS, state that it was not computed.",
        "• If sources conflict, report the conflict and cite each source. Do NOT choose a winner.",
        "• If a task failed or returned no data, say so explicitly.",
        "• State units and fiscal periods for every number.",
        "• Cite every factual claim with its source (Source N).",
        _SEP,
    ])

    context = "\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[Context truncated]"
    return context


def _collect_metric_rows(task: ToolTask, result: ToolResult, out: List[str]) -> None:
    rows = result.data.get("rows", [])
    conflict = result.data.get("conflict_detected", False)

    header = f"[{task.task_id}] {task.description}"
    if conflict:
        header += "  ⚠ CONFLICT: Multiple sources report different values"
    out.append(header)

    for i, row in enumerate(rows):
        conflict_marker = " ← CONFLICT" if conflict and len(rows) > 1 else ""
        out.append(
            f"  {row.get('metric_name','?')} | "
            f"{row.get('company_id','?')} | "
            f"FY{row.get('fiscal_year','?')}/{row.get('fiscal_period','?')} | "
            f"{row.get('value','?')} {row.get('unit','?')} ({row.get('scale','?')}) | "
            f"Source: {row.get('source_text','?')[:80]}{conflict_marker}"
        )
    out.append("")


def _collect_computation(task: ToolTask, result: ToolResult, out: List[str]) -> None:
    d = result.data
    out.append(f"[{task.task_id}] {task.description}")
    if d.get("formula"):
        out.append(f"  Formula: {d['formula']}")
    if d.get("result"):
        out.append(f"  Result:  {d['result']} {d.get('unit','')}")
    if d.get("reconciliation"):
        rec = d["reconciliation"]
        out.append(f"  Reconciliation: {rec['classification']} — {rec['note']}")
    out.append("")


def _collect_doc_section(task: ToolTask, result: ToolResult, out: List[str]) -> None:
    chunks = result.data.get("chunks", [])
    out.append(f"[{task.task_id}] {task.description}")
    for i, c in enumerate(chunks, 1):
        source = c.get("source", "?")
        page = c.get("page", "?")
        section = c.get("section", "")
        out.append(f"  [Source {i}] {source} | p.{page}" + (f" | {section}" if section else ""))
        out.append(f"  {c.get('text','')[:400]}")
        out.append("")
