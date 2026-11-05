"""
src/planning/models.py  ─  Phase 5 Data Models
════════════════════════════════════════════════
All Pydantic dataclasses for the planning layer.

DESIGN: Everything must be JSON-serializable.
LangGraph state stores these as dicts; Pydantic model_dump() handles that.

WHY PYDANTIC?
─────────────
  1. Field validation at construction time (not silent None bugs)
  2. .model_dump() gives clean JSON-serializable dicts for LangGraph state
  3. .model_validate(dict) reconstructs from LangGraph state
  4. Explicit defaults + optional fields = self-documenting schema

QUERY TYPES
────────────
See QueryType enum. Each maps to a routing decision in the graph.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ── Query Types ───────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    """
    Taxonomy of query complexity and structure.

    SIMPLE:
      Single information need, one retrieval call sufficient.
      Example: "What is OrionVault's FY2025 revenue?"

    MULTI_HOP:
      Answer to question B depends on answer to question A.
      Example: "What factors caused margin improvement and what risks could reverse it?"

    COMPARISON:
      Two or more entities/strategies/positions to compare side-by-side.
      Example: "Compare OrionVault's US and European growth strategies."

    TEMPORAL:
      Requires retrieving information from two distinct time periods.
      Example: "How did OrionVault's risk profile change from 2024 to 2026?"

    CROSS_DOCUMENT:
      Requires evidence from two structurally distinct source types.
      Example: "Compare management commentary on Sentinel Copilot with formal risk disclosures."

    MULTI_TOPIC:
      Multiple independent topics that each need dedicated retrieval.
      Example: "What are growth priorities and the major risks associated with each?"
    """
    SIMPLE         = "SIMPLE"
    MULTI_HOP      = "MULTI_HOP"
    COMPARISON     = "COMPARISON"
    TEMPORAL       = "TEMPORAL"
    CROSS_DOCUMENT = "CROSS_DOCUMENT"
    MULTI_TOPIC    = "MULTI_TOPIC"


# Query types that require multi-step planning
COMPLEX_QUERY_TYPES = {
    QueryType.MULTI_HOP,
    QueryType.COMPARISON,
    QueryType.TEMPORAL,
    QueryType.CROSS_DOCUMENT,
    QueryType.MULTI_TOPIC,
}


class TaskStatus(str, Enum):
    """Execution status of a single research task."""
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    DONE      = "DONE"
    FAILED    = "FAILED"
    SKIPPED   = "SKIPPED"   # dependency failed, so this was skipped


# ── QueryAnalysis ─────────────────────────────────────────────────────────────

class QueryAnalysis(BaseModel):
    """
    Structured output of the QueryAnalyzer.

    Captures all information the planner needs to decide whether decomposition
    is needed and what the research tasks should target.

    Fields:
        original_query:          The user's verbatim question.
        query_type:              Classified query type.
        requires_decomposition:  True = use complex multi-step path.
        entities:                Named entities extracted (companies, products, places).
        topics:                  High-level topic areas (growth, risk, finance…).
        temporal_constraints:    Date or period references ["2024", "FY2026", "2024-2026"].
        document_type_hints:     Suggested doc types to filter ["risk_memo", "annual_review"].
        expected_subquery_count: Estimated number of retrieval tasks needed (≥1).
        classifier_method:       How the type was determined ("rules" | "llm" | "hybrid").
        classifier_reasoning:    Short explanation of why this type was chosen.
        confidence:              0.0–1.0 classifier confidence.
    """
    original_query:          str
    query_type:              QueryType
    requires_decomposition:  bool
    entities:                List[str]          = Field(default_factory=list)
    topics:                  List[str]          = Field(default_factory=list)
    temporal_constraints:    List[str]          = Field(default_factory=list)
    document_type_hints:     List[str]          = Field(default_factory=list)
    expected_subquery_count: int                = 1
    classifier_method:       str                = "rules"
    classifier_reasoning:    str                = ""
    confidence:              float              = 1.0

    @model_validator(mode="after")
    def _sync_decomposition_flag(self) -> "QueryAnalysis":
        """Ensure SIMPLE queries are never decomposed."""
        if self.query_type == QueryType.SIMPLE:
            self.requires_decomposition = False
            self.expected_subquery_count = 1
        return self

    def model_dump_json_safe(self) -> Dict[str, Any]:
        """Dump to a JSON-serializable dict (enums as strings)."""
        d = self.model_dump()
        d["query_type"] = self.query_type.value
        return d


# ── ResearchTask ──────────────────────────────────────────────────────────────

class ResearchTask(BaseModel):
    """
    A single atomic retrieval task within a ResearchPlan.

    DESIGN PRINCIPLE:
    The task specifies WHAT to retrieve and WHY.
    The retrieval layer (search_chunks) decides HOW to retrieve.

    Fields:
        task_id:      Unique identifier within the plan (e.g., "task_1").
        description:  Human-readable purpose — what this task discovers.
        query:        Retrieval-optimized sub-query (NOT the original query).
        purpose:      Why this information is needed for the overall objective.
        depends_on:   task_ids that must complete before this runs.
        filters:      Optional metadata filters for Phase 4 retrieval.
        top_k:        Number of chunks to retrieve (default 8, max configurable).
        status:       Execution status (updated by executor).
        result_count: Number of chunks retrieved (set after execution).
        latency_ms:   Retrieval latency in ms (set after execution).
        error:        Error message if task failed.
    """
    task_id:      str
    description:  str
    query:        str
    purpose:      str                         = ""
    depends_on:   List[str]                   = Field(default_factory=list)
    filters:      Dict[str, Any]              = Field(default_factory=dict)
    top_k:        int                         = 8
    status:       TaskStatus                  = TaskStatus.PENDING
    result_count: int                         = 0
    latency_ms:   float                       = 0.0
    error:        Optional[str]               = None

    def is_ready(self, completed_ids: set[str]) -> bool:
        """Return True if all dependencies have completed successfully."""
        return all(dep in completed_ids for dep in self.depends_on)

    def model_dump_json_safe(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["status"] = self.status.value
        return d


# ── ResearchPlan ──────────────────────────────────────────────────────────────

class ResearchPlan(BaseModel):
    """
    Structured multi-step research plan produced by the ResearchPlanner.

    The plan is the contract between the planner and the executor:
      - Planner fills in tasks
      - Validator checks correctness
      - Executor runs tasks respecting depends_on
      - Aggregator collects results

    Fields:
        plan_id:    UUID (auto-generated).
        query_id:   Links back to the QueryAnalysis.
        objective:  One-sentence statement of the overall research goal.
        tasks:      Ordered list of ResearchTasks.
        version:    Plan version (incremented on correction passes).
        notes:      Planner notes or warnings.
    """
    plan_id:   str                = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    query_id:  str                = ""
    objective: str
    tasks:     List[ResearchTask] = Field(default_factory=list)
    version:   int                = 1
    notes:     str                = ""

    @property
    def task_ids(self) -> List[str]:
        return [t.task_id for t in self.tasks]

    @property
    def independent_tasks(self) -> List[ResearchTask]:
        """Tasks with no dependencies — can run in parallel."""
        return [t for t in self.tasks if not t.depends_on]

    @property
    def dependent_tasks(self) -> List[ResearchTask]:
        """Tasks that depend on at least one other task."""
        return [t for t in self.tasks if t.depends_on]

    def get_task(self, task_id: str) -> Optional[ResearchTask]:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def model_dump_json_safe(self) -> Dict[str, Any]:
        return {
            "plan_id":   self.plan_id,
            "query_id":  self.query_id,
            "objective": self.objective,
            "tasks":     [t.model_dump_json_safe() for t in self.tasks],
            "version":   self.version,
            "notes":     self.notes,
        }


# ── PlanningConfig ────────────────────────────────────────────────────────────

class PlanningConfig(BaseModel):
    """
    Configurable limits and behaviour for the planning layer.

    These are intentionally conservative defaults to prevent over-planning.

    Fields:
        max_tasks:              Maximum tasks per plan (rejects plans over this).
        max_task_retrieval:     Maximum chunks per task.
        max_total_evidence:     Maximum chunks across all tasks (after dedup).
        max_context_chars:      Maximum characters in final LLM context string.
        task_timeout_sec:       Per-task retrieval timeout.
        allow_llm_planner:      Use LLM-based planning (vs pure rule-based).
        allow_llm_classifier:   Use LLM for query classification.
        max_correction_passes:  Max times to retry an invalid plan.
        retrieval_mode:         Default Phase 4 retrieval mode for tasks.
    """
    max_tasks:             int   = 6
    max_task_retrieval:    int   = 10
    max_total_evidence:    int   = 30
    max_context_chars:     int   = 12_000
    task_timeout_sec:      float = 30.0
    allow_llm_planner:     bool  = True
    allow_llm_classifier:  bool  = True
    max_correction_passes: int   = 1
    retrieval_mode:        str   = "POSTGRES_HNSW"

    @classmethod
    def default(cls) -> "PlanningConfig":
        return cls()

    @classmethod
    def fast(cls) -> "PlanningConfig":
        """Faster config for testing/demos: smaller limits."""
        return cls(
            max_tasks=4,
            max_task_retrieval=5,
            max_total_evidence=15,
            max_context_chars=6_000,
            task_timeout_sec=15.0,
        )


# ── EvidenceRecord ────────────────────────────────────────────────────────────

class EvidenceRecord(BaseModel):
    """
    A single retrieved chunk with task provenance attached.

    This is what the aggregator collects after all tasks complete.
    Serializable, deduplicated by chunk_id.
    """
    chunk_id:           str
    task_id:            str
    text:               str
    source:             str               = ""
    document:           str               = ""
    document_type:      str               = ""
    version_label:      str               = ""
    effective_date:     Optional[str]     = None
    is_current:         bool              = False
    section:            str               = ""
    section_path:       List[str]         = Field(default_factory=list)
    page_start:         int               = 0
    page_end:           int               = 0
    content_type:       str               = "text"
    semantic_score:     Optional[float]   = None
    rerank_score:       Optional[float]   = None

    def best_score(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score
        return self.semantic_score or 0.0

    def model_dump_json_safe(self) -> Dict[str, Any]:
        return self.model_dump()
