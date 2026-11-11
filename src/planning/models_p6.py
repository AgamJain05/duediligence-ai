"""
src/planning/models_p6.py — Phase 6 extensions to the planning data models.

Adds tool-aware fields to ResearchTask and answer_path routing to QueryAnalysis.
Backward compatible: all new fields are Optional with defaults.

Import these alongside the Phase 5 models:
    from src.planning.models import ResearchTask, QueryAnalysis
    from src.planning.models_p6 import ToolTaskType, AnswerPath, ToolTask, ToolAwareQueryAnalysis
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.planning.models import (
    QueryAnalysis, ResearchTask, ResearchPlan, TaskStatus, PlanningConfig
)


# ── Task types ────────────────────────────────────────────────────────────────

class ToolTaskType(str, Enum):
    """Phase 6: typed task kinds the executor dispatches to specific tools."""
    RETRIEVE_DOCUMENTS = "RETRIEVE_DOCUMENTS"   # → search_documents
    QUERY_METRICS      = "QUERY_METRICS"        # → search_financial_data
    CALCULATE          = "CALCULATE"            # → calculate_metric
    COMPARE_PERIODS    = "COMPARE_PERIODS"      # → compare_periods
    RETRIEVE_SOURCE    = "RETRIEVE_SOURCE"      # → retrieve_source


TOOL_FOR_TASK_TYPE: Dict[ToolTaskType, str] = {
    ToolTaskType.RETRIEVE_DOCUMENTS: "search_documents",
    ToolTaskType.QUERY_METRICS:      "search_financial_data",
    ToolTaskType.CALCULATE:          "calculate_metric",
    ToolTaskType.COMPARE_PERIODS:    "compare_periods",
    ToolTaskType.RETRIEVE_SOURCE:    "retrieve_source",
}


# ── Answer path ───────────────────────────────────────────────────────────────

class AnswerPath(str, Enum):
    """
    High-level routing decision for the query.

    DOCS_ONLY:         Uses search_documents only. No metric tools.
    METRICS_ONLY:      Single metric lookup. Skips planner.
    METRICS_PLUS_CALC: Metric lookup(s) + calculator. No document search.
    HYBRID:            Mix of document search + metric lookup + calculation.
    """
    DOCS_ONLY        = "DOCS_ONLY"
    METRICS_ONLY     = "METRICS_ONLY"
    METRICS_PLUS_CALC = "METRICS_PLUS_CALC"
    HYBRID           = "HYBRID"


# ── Extended QueryAnalysis ────────────────────────────────────────────────────

class ToolAwareQueryAnalysis(BaseModel):
    """
    Phase 6 extension of QueryAnalysis. Wraps the Phase 5 analysis and adds
    tool routing fields. Used when Phase 6 tools are available.
    """
    # Phase 5 fields (re-exposed for convenience)
    original_query:         str
    query_type:             str                   # QueryType value
    requires_decomposition: bool
    entities:               List[str]             = Field(default_factory=list)
    topics:                 List[str]             = Field(default_factory=list)
    temporal_constraints:   List[str]             = Field(default_factory=list)
    confidence:             float                 = 1.0
    classifier_method:      str                   = "rules"
    classifier_reasoning:   str                   = ""

    # Phase 6 additions
    needs_documents:   bool           = True
    needs_metrics:     bool           = False
    needs_calculation: bool           = False
    metrics_mentioned: List[str]      = Field(default_factory=list)   # canonical names
    fiscal_periods:    List[str]      = Field(default_factory=list)   # e.g. ["FY2024", "FY2025"]
    companies:         List[str]      = Field(default_factory=list)
    answer_path:       AnswerPath     = AnswerPath.DOCS_ONLY

    def model_dump_json_safe(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["query_type"]   = str(self.query_type)
        d["answer_path"]  = self.answer_path.value
        return d

    @classmethod
    def from_phase5(
        cls,
        analysis: QueryAnalysis,
        needs_documents: bool = True,
        needs_metrics: bool = False,
        needs_calculation: bool = False,
        metrics_mentioned: Optional[List[str]] = None,
        fiscal_periods: Optional[List[str]] = None,
        companies: Optional[List[str]] = None,
        answer_path: AnswerPath = AnswerPath.DOCS_ONLY,
    ) -> "ToolAwareQueryAnalysis":
        return cls(
            original_query=analysis.original_query,
            query_type=analysis.query_type.value,
            requires_decomposition=analysis.requires_decomposition,
            entities=analysis.entities,
            topics=analysis.topics,
            temporal_constraints=analysis.temporal_constraints,
            confidence=analysis.confidence,
            classifier_method=analysis.classifier_method,
            classifier_reasoning=analysis.classifier_reasoning,
            needs_documents=needs_documents,
            needs_metrics=needs_metrics,
            needs_calculation=needs_calculation,
            metrics_mentioned=metrics_mentioned or [],
            fiscal_periods=fiscal_periods or [],
            companies=companies or [],
            answer_path=answer_path,
        )


# ── Tool-aware task ───────────────────────────────────────────────────────────

class ToolTask(BaseModel):
    """
    Phase 6 ResearchTask with tool routing.

    Extends the Phase 5 ResearchTask concept. The executor uses task_type
    and tool_args to dispatch to the correct tool — not free-text queries.

    input_bindings: maps this task's argument names to upstream task outputs.
      e.g. {"old_value": "task_1.result.value", "new_value": "task_2.result.value"}
    """
    task_id:        str
    task_type:      ToolTaskType
    tool:           str                        # must exist in registry
    tool_args:      Dict[str, Any]             = Field(default_factory=dict)
    description:    str
    purpose:        str                        = ""
    depends_on:     List[str]                  = Field(default_factory=list)
    input_bindings: Dict[str, str]             = Field(default_factory=dict)
    status:         TaskStatus                 = TaskStatus.PENDING
    result_count:   int                        = 0
    latency_ms:     float                      = 0.0
    error:          Optional[str]              = None
    tool_result:    Optional[Dict[str, Any]]   = None   # populated after execution

    def is_ready(self, completed_ids: set) -> bool:
        return all(dep in completed_ids for dep in self.depends_on)

    def model_dump_json_safe(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["task_type"] = self.task_type.value
        d["status"]    = self.status.value
        return d


# ── Tool-aware plan ───────────────────────────────────────────────────────────

class ToolAwareResearchPlan(BaseModel):
    """Phase 6 research plan with typed tool tasks."""
    plan_id:   str             = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    objective: str
    tasks:     List[ToolTask]  = Field(default_factory=list)
    version:   int             = 1
    notes:     str             = ""

    @property
    def task_ids(self) -> List[str]:
        return [t.task_id for t in self.tasks]

    @property
    def independent_tasks(self) -> List[ToolTask]:
        return [t for t in self.tasks if not t.depends_on]

    def get_task(self, task_id: str) -> Optional[ToolTask]:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def model_dump_json_safe(self) -> Dict[str, Any]:
        return {
            "plan_id":   self.plan_id,
            "objective": self.objective,
            "tasks":     [t.model_dump_json_safe() for t in self.tasks],
            "version":   self.version,
            "notes":     self.notes,
        }
