"""
src/tools/base.py — Tool interface, ToolResult, ToolError for Phase 6.

Every tool returns a ToolResult — never raises into the graph.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolError(BaseModel):
    code: str
    message: str


class Provenance(BaseModel):
    """Source attribution for a numeric value or document passage."""
    source_type: str              # "metric_row" | "document_chunk" | "computation"
    document:    Optional[str] = None
    page:        Optional[int] = None
    section:     Optional[str] = None
    source_text: Optional[str] = None
    metric_id:   Optional[int] = None
    chunk_id:    Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None


class ToolResult(BaseModel):
    """Serializable result returned by every tool call."""
    tool:        str
    status:      str                        # SUCCESS | EMPTY | ERROR | PARTIAL | CONFLICT
    data:        Dict[str, Any]             = Field(default_factory=dict)
    error:       Optional[ToolError]        = None
    provenance:  List[Provenance]           = Field(default_factory=list)
    latency_ms:  float                      = 0.0
    conflict_detected: bool                 = False

    def is_ok(self) -> bool:
        return self.status in ("SUCCESS", "PARTIAL", "CONFLICT")

    def model_dump_json_safe(self) -> Dict[str, Any]:
        """Decimal values → str for JSON serialization."""
        d = self.model_dump()
        return _make_json_safe(d)


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(i) for i in obj]
    return obj


class BaseTool(ABC):
    """Abstract base for all Phase 6 tools."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def run(self, **kwargs) -> ToolResult: ...

    def safe_run(self, **kwargs) -> ToolResult:
        """
        Execute run() and catch any unexpected exception.
        Tools must NEVER propagate raw exceptions into the graph.
        """
        import time
        t0 = time.perf_counter()
        try:
            result = self.run(**kwargs)
            result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return result
        except Exception as exc:
            return ToolResult(
                tool=self.name,
                status="ERROR",
                error=ToolError(code="UNEXPECTED_ERROR", message=str(exc)),
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
