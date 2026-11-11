"""
src/tools/source_retrieval.py — Retrieve full source text by chunk_id or metric_id.
"""
from __future__ import annotations

from typing import Optional

from src.tools.base import BaseTool, ToolError, ToolResult, Provenance


class SourceRetrievalTool(BaseTool):
    name = "retrieve_source"
    description = (
        "Given a chunk_id or metric_id, return the full source text, document, "
        "version, section, and page. Used for traceability and citation verification."
    )

    def __init__(self, db_session=None):
        self._session = db_session

    def run(
        self,
        chunk_id: Optional[str] = None,
        metric_id: Optional[int] = None,
    ) -> ToolResult:
        if chunk_id is None and metric_id is None:
            return ToolResult(
                tool=self.name, status="ERROR",
                error=ToolError(code="MISSING_INPUT", message="Provide chunk_id or metric_id."),
            )

        if self._session is None:
            return ToolResult(
                tool=self.name, status="ERROR",
                error=ToolError(code="NO_DB", message="Database session not available."),
            )

        if metric_id is not None:
            return self._get_metric_source(metric_id)

        return self._get_chunk_source(chunk_id)

    def _get_metric_source(self, metric_id: int) -> ToolResult:
        from src.financial.models import FinancialMetricP6
        from sqlalchemy import select
        row = self._session.get(FinancialMetricP6, metric_id)
        if row is None:
            return ToolResult(
                tool=self.name, status="EMPTY",
                data={"metric_id": metric_id},
            )
        return ToolResult(
            tool=self.name, status="SUCCESS",
            data={
                "type":            "metric",
                "metric_id":       row.id,
                "metric_name":     row.metric_name,
                "fiscal_year":     row.fiscal_year,
                "fiscal_period":   row.fiscal_period,
                "value":           str(row.value),
                "unit":            row.unit,
                "source_text":     row.source_text,
                "extraction_method": row.extraction_method,
                "document_version_id": row.document_version_id,
                "page":            row.page,
            },
            provenance=[Provenance(
                source_type="metric_row",
                metric_id=row.id,
                fiscal_year=row.fiscal_year,
                fiscal_period=row.fiscal_period,
                source_text=row.source_text,
            )],
        )

    def _get_chunk_source(self, chunk_id: str) -> ToolResult:
        from src.db.models import Chunk
        from sqlalchemy import select
        # chunk_id in Phase 4 is an integer PK; handle both int and string
        try:
            cid = int(chunk_id)
            chunk = self._session.get(Chunk, cid)
        except (ValueError, TypeError):
            chunk = None

        if chunk is None:
            return ToolResult(
                tool=self.name, status="EMPTY",
                data={"chunk_id": chunk_id},
            )

        return ToolResult(
            tool=self.name, status="SUCCESS",
            data={
                "type":       "chunk",
                "chunk_id":   str(chunk.id),
                "text":       chunk.text,
                "page_start": chunk.page_start,
                "page_end":   chunk.page_end,
                "content_type": chunk.content_type,
                "document_version_id": chunk.document_version_id,
            },
            provenance=[Provenance(
                source_type="document_chunk",
                chunk_id=str(chunk.id),
                source_text=chunk.text[:300],
                page=chunk.page_start,
            )],
        )
