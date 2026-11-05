"""
src/orchestration/state.py  ─  LangGraph State Schema
═══════════════════════════════════════════════════════
Phase 5: The ResearchState TypedDict is the single source of truth
passed between LangGraph nodes.

DESIGN CONSTRAINTS (LangGraph-specific):
  - State must be a TypedDict (or Pydantic model, but TypedDict is simpler)
  - All values must be JSON-serializable (no Python objects, no generators)
  - LangGraph merges state by key; nodes return partial dicts
  - Lists use the Annotated + add_messages reducer pattern for sequences

WHY SERIALIZABLE?
  LangGraph can checkpoint state to disk/db for resumability.
  Non-serializable objects (locks, file handles, generators) break this.
  We store Pydantic model dumps (dicts), not the model instances.

STATE LIFECYCLE:
  1. User query → input_state (only original_query is set)
  2. analyze_query node → fills analysis dict
  3. route_query node → fills route field
  4. (simple path) retrieve_simple → fills task_results, evidence
  5. (complex path) plan_research → fills research_plan dict
  6. validate_plan → may update research_plan (corrected)
  7. execute_tasks → fills task_results dict
  8. aggregate_evidence → fills evidence list
  9. build_context → fills final_context string
  10. generate_answer → fills answer string

ROUTE VALUES:
  "simple"  → direct retrieval (no planning)
  "complex" → full planning + decomposition + parallel retrieval
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    """
    Complete state flowing through the Phase 5 research graph.

    Keys populated by each node:

    [input]
      original_query:    str                  ← user's verbatim question

    [analyze_query]
      analysis:          dict                 ← QueryAnalysis.model_dump()

    [route_query]
      route:             str                  ← "simple" | "complex"

    [plan_research / retrieve_simple]
      research_plan:     Optional[dict]       ← ResearchPlan.model_dump()

    [execute_tasks / retrieve_simple]
      task_results:      Dict[str, list]      ← {task_id: [RetrievalResult.to_dict()]}

    [aggregate_evidence]
      evidence:          List[dict]           ← [EvidenceRecord.model_dump()]
      evidence_by_task:  Dict[str, List[dict]] ← {task_id: [EvidenceRecord]}

    [build_context]
      final_context:     str                  ← formatted LLM context string

    [generate_answer]
      answer:            str                  ← LLM answer

    [latency tracking]
      latency:           Dict[str, float]     ← {phase: ms}

    [error tracking]
      errors:            List[str]            ← non-fatal errors/warnings
      fatal_error:       Optional[str]        ← fatal error (stops graph)
    """
    original_query:    str
    analysis:          Dict[str, Any]
    route:             str
    research_plan:     Optional[Dict[str, Any]]
    task_results:      Dict[str, List[Dict[str, Any]]]
    evidence:          List[Dict[str, Any]]
    evidence_by_task:  Dict[str, List[Dict[str, Any]]]
    final_context:     str
    answer:            str
    latency:           Dict[str, float]
    errors:            List[str]
    fatal_error:       Optional[str]


def initial_state(query: str) -> ResearchState:
    """
    Create an empty state dict for a new research session.
    All optional fields initialized to safe defaults.
    """
    return ResearchState(
        original_query=query,
        analysis={},
        route="simple",
        research_plan=None,
        task_results={},
        evidence=[],
        evidence_by_task={},
        final_context="",
        answer="",
        latency={},
        errors=[],
        fatal_error=None,
    )


def merge_latency(existing: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    """Merge latency records (additive for same keys)."""
    merged = dict(existing)
    for k, v in new.items():
        merged[k] = merged.get(k, 0.0) + v
    return merged
