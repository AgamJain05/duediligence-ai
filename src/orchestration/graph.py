"""
src/orchestration/graph.py  ─  LangGraph Research Graph
════════════════════════════════════════════════════════
Phase 5: The main LangGraph StateGraph defining the complete
research workflow with conditional routing between simple and
complex paths.

GRAPH STRUCTURE:
  START
    ↓
  analyze_query           ← QueryAnalyzer.analyze()
    ↓
  route_query             ← Conditional edge (simple vs complex)
    ├── "simple" ──────→ retrieve_simple ─────────────────→ build_context
    └── "complex" ─────→ plan_research                          ↓
                              ↓                           generate_answer
                        validate_plan                          ↓
                              ↓                               END
                        execute_tasks
                              ↓
                        aggregate_evidence
                              ↓
                        build_context
                              ↓
                        generate_answer
                              ↓
                             END

WHY LANGGRAPH FOR THIS?
────────────────────────
LangGraph is introduced in Phase 5 because:
  1. The workflow now has BRANCHING (simple vs complex path)
  2. Multiple nodes need to pass STATE between them
  3. Execution order depends on graph topology, not code structure
  4. State is serializable → checkpointing possible in future phases

LangGraph's StateGraph is the minimal tool that handles these without
introducing a full agent framework.

IMPORTANT:
  Retrieval logic stays in src/db/repositories/retrieval.py (Phase 4).
  LangGraph only handles ORCHESTRATION — it doesn't know about vectors.

DB SESSION MANAGEMENT:
  The graph receives a db_session at construction time (or None for FAISS).
  Sessions are not stored in LangGraph state (not JSON-serializable).
  They are captured in closures by the node functions.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import StateGraph, END, START

from src.orchestration.state import ResearchState, initial_state, merge_latency
from src.planning.models import (
    PlanningConfig,
    QueryAnalysis,
    QueryType,
    ResearchPlan,
    EvidenceRecord,
)
from src.planning.analyzer import QueryAnalyzer
from src.planning.planner import ResearchPlanner
from src.planning.validator import PlanValidator, PlanValidationError


# ── Node builder utilities ────────────────────────────────────────────────────

def _make_search_fn(db_session, config: PlanningConfig):
    """Create the appropriate search function based on available backend."""
    from src.orchestration.executor import make_pg_search_fn, make_faiss_search_fn
    if db_session is not None:
        return make_pg_search_fn(db_session, config.retrieval_mode)
    return make_faiss_search_fn(top_k=config.max_task_retrieval)


def _safe_errors(state: ResearchState) -> List[str]:
    return list(state.get("errors") or [])


# ── Node: analyze_query ───────────────────────────────────────────────────────

def make_analyze_query_node(config: PlanningConfig):
    """Factory: returns the analyze_query node function."""
    analyzer = QueryAnalyzer(config)

    def analyze_query(state: ResearchState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["original_query"]

        analysis = analyzer.analyze(query)
        ms       = (time.perf_counter() - t0) * 1000

        return {
            "analysis": analysis.model_dump_json_safe(),
            "latency":  merge_latency(state.get("latency", {}), {"analyze_ms": ms}),
        }

    return analyze_query


# ── Node: route_query ─────────────────────────────────────────────────────────

def route_query(state: ResearchState) -> str:
    """
    Conditional edge function: decides which path to take.

    Returns "simple" or "complex".
    LangGraph calls this to choose the next node.
    """
    analysis_dict = state.get("analysis", {})
    requires      = analysis_dict.get("requires_decomposition", False)
    return "complex" if requires else "simple"


# ── Node: retrieve_simple ─────────────────────────────────────────────────────

def make_retrieve_simple_node(config: PlanningConfig, search_fn: Callable):
    """Factory: returns the retrieve_simple node for the SIMPLE path."""

    def retrieve_simple(state: ResearchState) -> Dict[str, Any]:
        """Single retrieval call. No planning, no decomposition."""
        from src.orchestration.executor import _result_to_evidence
        import asyncio

        t0    = time.perf_counter()
        query = state["original_query"]
        errors = _safe_errors(state)

        try:
            # Run async search_fn synchronously
            raw = asyncio.run(
                search_fn(query, {}, config.max_task_retrieval)
            )
            evidence = [_result_to_evidence(r, "task_1") for r in raw]
        except Exception as exc:
            errors.append(f"Simple retrieval error: {exc}")
            evidence = []

        ms = (time.perf_counter() - t0) * 1000

        return {
            "task_results":     {"task_1": [e.model_dump() for e in evidence]},
            "evidence":         [e.model_dump() for e in evidence],
            "evidence_by_task": {"task_1": [e.model_dump() for e in evidence]},
            "errors":           errors,
            "latency":          merge_latency(state.get("latency", {}), {"retrieve_simple_ms": ms}),
        }

    return retrieve_simple


# ── Node: plan_research ───────────────────────────────────────────────────────

def make_plan_research_node(config: PlanningConfig):
    """Factory: returns the plan_research node for the COMPLEX path."""
    planner   = ResearchPlanner(config)
    validator = PlanValidator(config)

    def plan_research(state: ResearchState) -> Dict[str, Any]:
        t0     = time.perf_counter()
        errors = _safe_errors(state)

        analysis = QueryAnalysis.model_validate(state["analysis"])

        try:
            plan = planner.plan(analysis)
            plan = validator.validate_and_correct(plan)
        except PlanValidationError as exc:
            errors.append(f"Plan validation failed: {exc}")
            # Fall back to simple 1-task plan
            from src.planning.planner import _plan_simple
            plan = _plan_simple(analysis, config)
            plan.notes += " [fallback after validation failure]"

        ms = (time.perf_counter() - t0) * 1000

        return {
            "research_plan": plan.model_dump_json_safe(),
            "errors":        errors,
            "latency":       merge_latency(state.get("latency", {}), {"plan_ms": ms}),
        }

    return plan_research


# ── Node: execute_tasks ───────────────────────────────────────────────────────

def make_execute_tasks_node(config: PlanningConfig, search_fn: Callable):
    """Factory: returns the execute_tasks node."""

    def execute_tasks(state: ResearchState) -> Dict[str, Any]:
        from src.orchestration.executor import TaskExecutor
        import asyncio

        t0     = time.perf_counter()
        errors = _safe_errors(state)

        plan_dict = state.get("research_plan")
        if not plan_dict:
            errors.append("No research plan found in state — cannot execute tasks.")
            return {"task_results": {}, "errors": errors}

        plan = ResearchPlan.model_validate(plan_dict)

        executor = TaskExecutor(config=config, search_fn=search_fn)
        raw_results = asyncio.run(executor.execute(plan))

        # Serialize: EvidenceRecord → dict
        task_results = {
            tid: [r.model_dump() for r in records]
            for tid, records in raw_results.items()
        }

        ms = (time.perf_counter() - t0) * 1000

        # Update plan task statuses back to state
        updated_plan = plan.model_dump_json_safe()

        return {
            "task_results":  task_results,
            "research_plan": updated_plan,
            "errors":        errors,
            "latency":       merge_latency(state.get("latency", {}), {"execute_ms": ms}),
        }

    return execute_tasks


# ── Node: aggregate_evidence ──────────────────────────────────────────────────

def make_aggregate_evidence_node(config: PlanningConfig):
    """Factory: returns the aggregate_evidence node."""

    def aggregate_evidence(state: ResearchState) -> Dict[str, Any]:
        from src.orchestration.context_builder import aggregate_evidence as _aggregate

        t0     = time.perf_counter()
        errors = _safe_errors(state)

        task_results_raw = state.get("task_results", {})
        plan_dict        = state.get("research_plan")
        plan             = ResearchPlan.model_validate(plan_dict) if plan_dict else None

        # Deserialize dicts to EvidenceRecord
        task_results = {}
        for tid, recs in task_results_raw.items():
            task_results[tid] = [
                EvidenceRecord(**r) if isinstance(r, dict) else r
                for r in recs
            ]

        evidence_flat, evidence_by_task = _aggregate(task_results, plan, config)
        ms = (time.perf_counter() - t0) * 1000

        return {
            "evidence":         [r.model_dump() for r in evidence_flat],
            "evidence_by_task": {
                tid: [r.model_dump() for r in recs]
                for tid, recs in evidence_by_task.items()
            },
            "errors":  errors,
            "latency": merge_latency(state.get("latency", {}), {"aggregate_ms": ms}),
        }

    return aggregate_evidence


# ── Node: build_context ───────────────────────────────────────────────────────

def make_build_context_node(config: PlanningConfig):
    """Factory: returns the build_context node (shared by simple + complex paths)."""

    def build_context(state: ResearchState) -> Dict[str, Any]:
        from src.orchestration.context_builder import (
            build_structured_context, build_simple_context
        )

        t0      = time.perf_counter()
        route   = state.get("route", "simple")
        query   = state["original_query"]
        errors  = _safe_errors(state)

        evidence_by_task_raw = state.get("evidence_by_task", {})
        plan_dict            = state.get("research_plan")
        plan = ResearchPlan.model_validate(plan_dict) if plan_dict else None

        evidence_by_task = {
            tid: [EvidenceRecord(**r) if isinstance(r, dict) else r for r in recs]
            for tid, recs in evidence_by_task_raw.items()
        }

        if route == "simple" or not plan or len(evidence_by_task) <= 1:
            flat = [
                EvidenceRecord(**r) if isinstance(r, dict) else r
                for r in state.get("evidence", [])
            ]
            context = build_simple_context(query, flat, config)
        else:
            context = build_structured_context(query, evidence_by_task, plan, config)

        ms = (time.perf_counter() - t0) * 1000

        return {
            "final_context": context,
            "errors":        errors,
            "latency":       merge_latency(state.get("latency", {}), {"context_ms": ms}),
        }

    return build_context


# ── Node: generate_answer ─────────────────────────────────────────────────────

def make_generate_answer_node():
    """Factory: returns the generate_answer node."""

    def generate_answer(state: ResearchState) -> Dict[str, Any]:
        import os
        from openai import OpenAI

        t0      = time.perf_counter()
        query   = state["original_query"]
        context = state.get("final_context", "")
        errors  = _safe_errors(state)

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        model   = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")

        if not api_key:
            return {
                "answer":  "[ERROR] OPENROUTER_API_KEY not set.",
                "errors":  errors,
                "latency": merge_latency(state.get("latency", {}), {"generate_ms": 0.0}),
            }

        try:
            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            system = (
                "You are a meticulous financial and business analyst specializing in "
                "company due-diligence reports. Answer using ONLY the evidence provided. "
                "Cite sources as (Source N). Do not invent facts."
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": f"{context}\n\nQUESTION: {query}"},
                ],
                max_tokens=1200,
                temperature=0.1,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as exc:
            answer = f"[ERROR] Generation failed: {exc}"
            errors.append(str(exc))

        ms = (time.perf_counter() - t0) * 1000

        return {
            "answer":  answer,
            "errors":  errors,
            "latency": merge_latency(state.get("latency", {}), {"generate_ms": ms}),
        }

    return generate_answer


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    config: Optional[PlanningConfig] = None,
    db_session=None,
):
    """
    Build and compile the Phase 5 research graph.

    Args:
        config:     PlanningConfig (uses default if None).
        db_session: SQLAlchemy session for Phase 4 retrieval.
                    If None, falls back to Phase 3 FAISS.

    Returns:
        Compiled LangGraph (callable as graph.invoke(state)).
    """
    config    = config or PlanningConfig.default()
    search_fn = _make_search_fn(db_session, config)

    # ── Build nodes ────────────────────────────────────────────────────────────
    analyze_query_node    = make_analyze_query_node(config)
    retrieve_simple_node  = make_retrieve_simple_node(config, search_fn)
    plan_research_node    = make_plan_research_node(config)
    execute_tasks_node    = make_execute_tasks_node(config, search_fn)
    aggregate_node        = make_aggregate_evidence_node(config)
    build_context_node    = make_build_context_node(config)
    generate_answer_node  = make_generate_answer_node()

    # ── Define route_with_state ────────────────────────────────────────────────
    # LangGraph conditional edges call this function; it receives state and
    # returns a string that maps to a node name via the edge dict.
    def _route(state: ResearchState) -> str:
        route = route_query(state)
        # Also update route in state — we do this by returning a dict from
        # a wrapper, but LangGraph edge functions don't update state.
        # Route is stored in state by a thin wrapper node instead.
        return route

    # ── Wrapper to write route into state ─────────────────────────────────────
    def set_route(state: ResearchState) -> Dict[str, Any]:
        route = route_query(state)
        return {"route": route}

    # ── Assemble graph ─────────────────────────────────────────────────────────
    graph = StateGraph(ResearchState)

    graph.add_node("analyze_query",      analyze_query_node)
    graph.add_node("set_route",          set_route)
    graph.add_node("retrieve_simple",    retrieve_simple_node)
    graph.add_node("plan_research",      plan_research_node)
    graph.add_node("execute_tasks",      execute_tasks_node)
    graph.add_node("aggregate_evidence", aggregate_node)
    graph.add_node("build_context",      build_context_node)
    graph.add_node("generate_answer",    generate_answer_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "set_route")

    # Conditional routing after set_route
    graph.add_conditional_edges(
        "set_route",
        lambda state: state.get("route", "simple"),
        {
            "simple":  "retrieve_simple",
            "complex": "plan_research",
        },
    )

    # Simple path
    graph.add_edge("retrieve_simple", "build_context")

    # Complex path
    graph.add_edge("plan_research",      "execute_tasks")
    graph.add_edge("execute_tasks",      "aggregate_evidence")
    graph.add_edge("aggregate_evidence", "build_context")

    # Shared terminal
    graph.add_edge("build_context",   "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


class ResearchGraph:
    """
    High-level interface for the Phase 5 research graph.

    Usage:
        rg = ResearchGraph(config, db_session)
        result = rg.run("Compare US and European expansion strategies.")
        print(result.answer)
    """

    def __init__(
        self,
        config: Optional[PlanningConfig] = None,
        db_session=None,
    ) -> None:
        self.config    = config or PlanningConfig.default()
        self.db_session = db_session
        self._graph    = build_graph(self.config, db_session)

    def run(self, query: str) -> ResearchState:
        """Execute the full research pipeline and return final state."""
        state = initial_state(query)
        result = self._graph.invoke(state)
        return result

    def stream(self, query: str):
        """Stream state updates from each node."""
        state = initial_state(query)
        yield from self._graph.stream(state)
