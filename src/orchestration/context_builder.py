"""
src/orchestration/context_builder.py  ─  Structured Context Builder
═══════════════════════════════════════════════════════════════════════
Phase 5: Builds the final LLM generation context from aggregated evidence.

DESIGN:
  Context is NOT a flat blob. It's structured by research task so the LLM
  understands which evidence answers which sub-question.

FORMAT:
  ══════════════════════════════════════════════════════════════════
  RESEARCH QUESTION
  [original query]
  ══════════════════════════════════════════════════════════════════

  [TASK 1: description]
  Purpose: [why this was retrieved]
  ────────────────────────────────────────────────────────────────
  [Source 1] Document | Page | Section
  [chunk text]

  [Source 2] ...

  [TASK 2: ...]
  ...
  ══════════════════════════════════════════════════════════════════
  INSTRUCTIONS: Use the evidence above to answer the research question.
  ...

LIMITS:
  - max_context_chars controls total context size
  - Chunks are prioritized by best_score() (rerank > fusion > semantic)
  - Chunks are deduplicated across tasks (same chunk_id, first task wins)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.planning.models import EvidenceRecord, PlanningConfig, ResearchPlan

_SEPARATOR = "═" * 68
_TASK_SEP  = "─" * 60


def _format_evidence_record(rec: EvidenceRecord, idx: int) -> str:
    """Format a single evidence record for the context block."""
    source_line = (
        f"[Source {idx}] "
        f"{rec.document or rec.source or 'Unknown'}"
        f" | page {rec.page_start}"
        f"{f'–{rec.page_end}' if rec.page_end and rec.page_end != rec.page_start else ''}"
        f" | {rec.section or 'n/a'}"
    )
    score_info = ""
    if rec.semantic_score is not None:
        score_info = f" | sim={rec.semantic_score:.3f}"
    if rec.rerank_score is not None:
        score_info += f" | rerank={rec.rerank_score:.3f}"

    return f"{source_line}{score_info}\n{rec.text.strip()}"


def build_structured_context(
    query: str,
    evidence_by_task: Dict[str, List[EvidenceRecord]],
    plan: Optional[ResearchPlan],
    config: Optional[PlanningConfig] = None,
) -> str:
    """
    Build the final context string for LLM generation.

    Args:
        query:            Original user query.
        evidence_by_task: {task_id: [EvidenceRecord]} from aggregator.
        plan:             ResearchPlan for task descriptions (may be None).
        config:           PlanningConfig for size limits.

    Returns:
        Formatted context string.
    """
    config = config or PlanningConfig.default()
    parts  = []

    # ── Header ────────────────────────────────────────────────────────────────
    parts.append(_SEPARATOR)
    parts.append("RESEARCH QUESTION")
    parts.append(query)
    parts.append(_SEPARATOR)
    parts.append("")

    # ── Evidence sections ─────────────────────────────────────────────────────
    global_source_idx = 1

    for task_id, records in evidence_by_task.items():
        if not records:
            continue

        # Get task description from plan
        task_desc    = task_id
        task_purpose = ""
        if plan:
            task = plan.get_task(task_id)
            if task:
                task_desc    = task.description
                task_purpose = task.purpose

        parts.append(f"[{task_id.upper()}: {task_desc}]")
        if task_purpose:
            parts.append(f"Purpose: {task_purpose}")
        parts.append(_TASK_SEP)

        for rec in records:
            parts.append(_format_evidence_record(rec, global_source_idx))
            global_source_idx += 1
            parts.append("")

        parts.append("")

    # ── Instructions ──────────────────────────────────────────────────────────
    parts.append(_SEPARATOR)
    parts.append("INSTRUCTIONS")
    parts.append(
        "Answer the research question using ONLY the evidence passages above.\n"
        "• Cite each factual claim with its source number, e.g. (Source 1).\n"
        "• If the evidence does not support a claim, state that explicitly.\n"
        "• Do NOT invent facts, numbers, or names not present in the evidence.\n"
        "• Structure your answer clearly. Use headings or bullet points as appropriate."
    )
    parts.append(_SEPARATOR)

    context = "\n".join(parts)

    # ── Context size limit ────────────────────────────────────────────────────
    if len(context) > config.max_context_chars:
        context = context[:config.max_context_chars]
        context += f"\n\n[Context truncated at {config.max_context_chars} characters]"

    return context


def build_simple_context(
    query: str,
    evidence: List[EvidenceRecord],
    config: Optional[PlanningConfig] = None,
) -> str:
    """
    Build context for the SIMPLE path (single retrieval, no task structure).
    Compatible with the Phase 4 generator format.
    """
    config = config or PlanningConfig.default()

    parts = []
    parts.append(_SEPARATOR)
    parts.append("QUESTION")
    parts.append(query)
    parts.append(_SEPARATOR)
    parts.append("")

    for i, rec in enumerate(evidence, start=1):
        parts.append(_format_evidence_record(rec, i))
        parts.append("")

    parts.append(_SEPARATOR)
    parts.append(
        "Answer the question using ONLY the context above. "
        "Cite sources. Do not invent facts."
    )
    parts.append(_SEPARATOR)

    context = "\n".join(parts)
    if len(context) > config.max_context_chars:
        context = context[:config.max_context_chars] + "\n[...truncated]"
    return context


# ── Evidence Aggregator ───────────────────────────────────────────────────────

def aggregate_evidence(
    task_results: Dict[str, List[Any]],  # {task_id: [EvidenceRecord or dict]}
    plan: Optional[ResearchPlan],
    config: Optional[PlanningConfig] = None,
) -> tuple[List[EvidenceRecord], Dict[str, List[EvidenceRecord]]]:
    """
    Aggregate all task results into:
      1. Flat deduplicated evidence list (sorted by score)
      2. Evidence grouped by task_id

    Deduplication: same chunk_id appears only once (first task wins).
    Total evidence capped at config.max_total_evidence.

    Returns: (evidence_flat, evidence_by_task)
    """
    config = config or PlanningConfig.default()

    seen_chunk_ids: set[str] = set()
    evidence_by_task: Dict[str, List[EvidenceRecord]] = {}
    all_evidence: List[EvidenceRecord] = []

    # Iterate in task order (preserves plan task ordering)
    task_order = [t.task_id for t in plan.tasks] if plan else list(task_results.keys())

    for task_id in task_order:
        records_raw = task_results.get(task_id, [])
        task_evidence = []

        for r in records_raw:
            # Handle both EvidenceRecord objects and plain dicts
            if isinstance(r, dict):
                try:
                    rec = EvidenceRecord(**r)
                except Exception:
                    continue
            elif isinstance(r, EvidenceRecord):
                rec = r
            else:
                continue

            if rec.chunk_id in seen_chunk_ids:
                continue  # deduplicate

            seen_chunk_ids.add(rec.chunk_id)
            task_evidence.append(rec)

        # Sort within task by best score descending
        task_evidence.sort(key=lambda r: r.best_score(), reverse=True)
        evidence_by_task[task_id] = task_evidence
        all_evidence.extend(task_evidence)

    # Sort flat list by score descending
    all_evidence.sort(key=lambda r: r.best_score(), reverse=True)

    # Cap total
    all_evidence = all_evidence[:config.max_total_evidence]

    # Also cap per-task lists if we hit total cap
    if len(all_evidence) < sum(len(v) for v in evidence_by_task.values()):
        kept_ids = {r.chunk_id for r in all_evidence}
        evidence_by_task = {
            tid: [r for r in recs if r.chunk_id in kept_ids]
            for tid, recs in evidence_by_task.items()
        }

    return all_evidence, evidence_by_task
