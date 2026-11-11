"""
src/financial/models.py — Phase 6 ORM models for financial_metrics_p6 and metric_aliases.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.models import Base


class FinancialMetricP6(Base):
    """
    Detailed financial metric row with full provenance.

    Uniqueness: (company_id, fiscal_year, fiscal_period, metric_name, dimension_type,
                 dimension_value, document_version_id) — enforced via two partial indexes.

    Multiple rows for the same (company, year, period, metric) ARE ALLOWED when
    they come from different document_version_ids (e.g. annual report vs earnings call).
    This is how conflicts are stored and surfaced.
    """
    __tablename__ = "financial_metrics_p6"

    id:                  Mapped[int]            = mapped_column(Integer, primary_key=True)
    company_id:          Mapped[int]            = mapped_column(Integer, ForeignKey("companies.id"), nullable=False)
    fiscal_year:         Mapped[int]            = mapped_column(Integer, nullable=False)
    fiscal_period:       Mapped[str]            = mapped_column(String(5), nullable=False)
    period_end_date:     Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    metric_name:         Mapped[str]            = mapped_column(String(100), nullable=False)
    value:               Mapped[Decimal]        = mapped_column(Numeric(20, 6), nullable=False)
    unit:                Mapped[str]            = mapped_column(String(20), nullable=False)
    scale:               Mapped[str]            = mapped_column(String(20), nullable=False, default="UNITS")
    dimension_type:      Mapped[Optional[str]]  = mapped_column(String(20), nullable=True)
    dimension_value:     Mapped[Optional[str]]  = mapped_column(String(100), nullable=True)
    is_derived:          Mapped[bool]           = mapped_column(Boolean, nullable=False, default=False)
    document_id:         Mapped[Optional[int]]  = mapped_column(Integer, ForeignKey("documents.id"), nullable=True)
    document_version_id: Mapped[Optional[int]]  = mapped_column(Integer, ForeignKey("document_versions.id"), nullable=True)
    section_id:          Mapped[Optional[int]]  = mapped_column(Integer, ForeignKey("sections.id"), nullable=True)
    chunk_id:            Mapped[Optional[int]]  = mapped_column(Integer, ForeignKey("chunks.id"), nullable=True)
    page:                Mapped[Optional[int]]  = mapped_column(Integer, nullable=True)
    extraction_method:   Mapped[str]            = mapped_column(String(20), nullable=False)
    source_text:         Mapped[str]            = mapped_column(Text, nullable=False, default="")
    created_at:          Mapped[datetime]       = mapped_column(DateTime(timezone=True), server_default=func.now())

    company          = relationship("Company")
    document_version = relationship("DocumentVersion")

    def to_dict(self) -> dict:
        return {
            "id":                  self.id,
            "company_id":          self.company_id,
            "fiscal_year":         self.fiscal_year,
            "fiscal_period":       self.fiscal_period,
            "metric_name":         self.metric_name,
            "value":               str(self.value),
            "unit":                self.unit,
            "scale":               self.scale,
            "dimension_type":      self.dimension_type,
            "dimension_value":     self.dimension_value,
            "is_derived":          self.is_derived,
            "document_version_id": self.document_version_id,
            "page":                self.page,
            "extraction_method":   self.extraction_method,
            "source_text":         self.source_text,
        }

    def __repr__(self) -> str:
        dim = f" [{self.dimension_type}={self.dimension_value}]" if self.dimension_type else ""
        return (
            f"<FinancialMetricP6 {self.metric_name}{dim} "
            f"FY{self.fiscal_year}/{self.fiscal_period} "
            f"={self.value} {self.unit} ({self.scale})>"
        )


class MetricAlias(Base):
    """Maps alias strings to canonical metric names."""
    __tablename__ = "metric_aliases"

    alias:          Mapped[str] = mapped_column(String(200), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(100), nullable=False)

    def __repr__(self) -> str:
        return f"<MetricAlias '{self.alias}' \u2192 '{self.canonical_name}'>"
