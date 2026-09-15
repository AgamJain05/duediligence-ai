"""
src/db/models.py  ─  SQLAlchemy ORM models for Phase 4 schema
══════════════════════════════════════════════════════════════
Phase 4: Relational + vector schema for the OrionVault knowledge base.

SCHEMA DESIGN RATIONALE
────────────────────────
The schema is normalized to capture the real-world structure of
due-diligence document management:

  companies → documents → document_versions → chunks
                                           → sections

This hierarchy answers questions like:
  - "Which COMPANY published this chunk?" (join → company)
  - "Which VERSION of the document does this come from?" (document_version)
  - "Was this the CURRENT version at query time?" (is_current flag)
  - "What SECTION did this chunk appear in?" (section hierarchy)

WHY SEPARATE document_versions?
────────────────────────────────
A company's risk memo from 2024 and its updated version from 2026 are
the SAME document but DIFFERENT versions. Keeping them separate lets us:
  1. Return only the current version by default
  2. Return the historically applicable version for temporal queries
  3. Track exactly what changed between versions
  4. Detect duplicate ingestion via checksum

VECTOR STORAGE ON chunks
─────────────────────────
Embedding vectors are stored directly on the chunks table as a pgvector
column (vector(768)). This avoids a join and keeps the schema simple.

Alternative: a separate chunk_embeddings table (one row per model).
This would be needed if testing multiple embedding models simultaneously.
For Phase 4, one model (BAAI/bge-base-en-v1.5) is used throughout.
The embedding_model column records which model produced each embedding,
allowing future migration to a separate table if needed.

DISTANCE METRIC: Cosine via pgvector <=> operator
────────────────────────────────────────────────────
BAAI/bge-base-en-v1.5 produces L2-normalized vectors. For normalized
vectors: cosine_similarity(a,b) = inner_product(a,b). pgvector's <=>
computes cosine DISTANCE = 1 - cosine_similarity. Results are sorted
ascending (smaller distance = more similar).

This is consistent with Phase 1–3 which use FAISS IndexFlatIP (inner
product on normalized vectors = cosine similarity sorted descending).
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ── Embedding dimension ───────────────────────────────────────────────────────
# The dimension MUST match the embedding model configured in EMBEDDING_MODEL.
# BAAI/bge-base-en-v1.5 → 768 dimensions.
# If you change the model, update this value AND re-run migrations.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))


# ── Base class ────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    # Allow legacy relationship annotations (List["Model"]) alongside Mapped[]
    # columns. Without this, SQLAlchemy 2.x requires Mapped[List["Model"]] for
    # all relationship fields, which makes the models more verbose.
    __allow_unmapped__ = True



# ── companies ─────────────────────────────────────────────────────────────────
class Company(Base):
    """
    A company being analyzed.

    slug: URL-safe identifier derived from name.
    Example: "OrionVault Systems" → "orionvault-systems"
    The slug uniquely identifies a company and can be used in filter APIs.
    """
    __tablename__ = "companies"

    id:         Mapped[int]          = mapped_column(Integer, primary_key=True)
    name:       Mapped[str]          = mapped_column(String(255), nullable=False)
    slug:       Mapped[str]          = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime]     = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime]     = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    documents:         List["Document"]        = relationship("Document", back_populates="company")
    financial_metrics: List["FinancialMetric"] = relationship("FinancialMetric", back_populates="company")

    def __repr__(self) -> str:
        return f"<Company id={self.id} name='{self.name}' slug='{self.slug}'>"


# ── sources ───────────────────────────────────────────────────────────────────
class Source(Base):
    """
    Provenance record for a document version.

    WHY: Every document version should trace back to a source — where the
    file came from, when it was retrieved, and a checksum to detect changes.
    For local files, source_uri is the absolute file path.
    For remote files, it would be a URL (not invented — only real paths used).
    """
    __tablename__ = "sources"

    id:           Mapped[int]           = mapped_column(Integer, primary_key=True)
    source_name:  Mapped[str]           = mapped_column(String(512), nullable=False)
    source_uri:   Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_type:  Mapped[str]           = mapped_column(String(50), nullable=False, default="local_file")
    # Examples: "local_file", "s3", "sharepoint", "web_scrape"
    retrieved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    checksum:     Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # SHA-256 hex digest of the source file

    document_versions: List["DocumentVersion"] = relationship("DocumentVersion", back_populates="source")

    def __repr__(self) -> str:
        return f"<Source id={self.id} name='{self.source_name}' type='{self.source_type}'>"


# ── documents ─────────────────────────────────────────────────────────────────
class Document(Base):
    """
    A logical document (may have multiple versions).

    document_type examples:
      annual_review, risk_memo, earnings_call, technical_brief,
      policy, incident_report, case_study

    The type is a free-form string, not an enum, so new types can be added
    without a schema change.
    """
    __tablename__ = "documents"

    id:            Mapped[int]      = mapped_column(Integer, primary_key=True)
    company_id:    Mapped[int]      = mapped_column(Integer, ForeignKey("companies.id"), nullable=False)
    title:         Mapped[str]      = mapped_column(String(512), nullable=False)
    document_type: Mapped[str]      = mapped_column(String(100), nullable=False)
    created_at:    Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:    Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    company:   "Company"              = relationship("Company", back_populates="documents")
    versions:  List["DocumentVersion"] = relationship("DocumentVersion", back_populates="document")

    def __repr__(self) -> str:
        return f"<Document id={self.id} title='{self.title}' type='{self.document_type}'>"


# ── document_versions ─────────────────────────────────────────────────────────
class DocumentVersion(Base):
    """
    One version of a document.

    UNIQUENESS STRATEGY
    ────────────────────
    The pair (document_id, checksum) is unique. This prevents the same file
    from being ingested twice. A new version requires either a new document_id
    (impossible for the same document) or a different checksum (meaning the
    file content changed).

    VERSION PRECEDENCE (for temporal queries)
    ──────────────────────────────────────────
    For "what applied on date D?":
      1. Filter: effective_date <= D
      2. Among results, pick the one with the LATEST effective_date
      3. Tie-break: pick the one with the LATEST published_date
      4. Note: filename ordering is NOT used (unreliable)

    is_current=True marks the version to use for queries with no date context.
    Only ONE version per document should have is_current=True. This is
    enforced by application logic in the repository layer (not a DB constraint,
    because PostgreSQL partial unique indexes can enforce it but the upsert
    logic is cleaner in Python).
    """
    __tablename__ = "document_versions"
    __table_args__ = (
        # Prevent duplicate ingestion of the same file content for the same document
        UniqueConstraint("document_id", "checksum", name="uq_docver_document_checksum"),
    )

    id:              Mapped[int]           = mapped_column(Integer, primary_key=True)
    document_id:     Mapped[int]           = mapped_column(Integer, ForeignKey("documents.id"), nullable=False)
    source_id:       Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("sources.id"), nullable=True)
    version_label:   Mapped[str]           = mapped_column(String(50), nullable=False)
    # Human-readable: "v1", "v2", "2024", "FY2026", etc.
    effective_date:  Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    # Date from which this version is the applicable one
    published_date:  Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    # Date the document was actually published/released
    source_filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    source_uri:      Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    checksum:        Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # SHA-256 hex digest of the PDF file — used for duplicate detection
    is_current:      Mapped[bool]          = mapped_column(Boolean, default=False, nullable=False)
    created_at:      Mapped[datetime]      = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document:         "Document"        = relationship("Document", back_populates="versions")
    source:           Optional["Source"] = relationship("Source", back_populates="document_versions")
    sections:         List["Section"]    = relationship("Section", back_populates="document_version")
    chunks:           List["Chunk"]      = relationship("Chunk", back_populates="document_version")
    financial_metrics: List["FinancialMetric"] = relationship(
        "FinancialMetric", back_populates="source_document_version"
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentVersion id={self.id} label='{self.version_label}' "
            f"is_current={self.is_current} effective={self.effective_date}>"
        )


# ── sections ──────────────────────────────────────────────────────────────────
class Section(Base):
    """
    A section in the document hierarchy (from Phase 2 structure detection).

    Preserves the section tree from Phase 2:
        Risk Factors
          ├── Cloud Infrastructure
          ├── Customer Concentration
          └── Regulatory Risk

    section_path is stored as a JSONB list of strings:
        ["Risk Factors", "Cloud Infrastructure"]

    This allows SQL queries like:
        WHERE section_path @> '["Risk Factors"]'  -- any risk sub-section
    """
    __tablename__ = "sections"

    id:                  Mapped[int]           = mapped_column(Integer, primary_key=True)
    document_version_id: Mapped[int]           = mapped_column(Integer, ForeignKey("document_versions.id"), nullable=False)
    parent_section_id:   Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("sections.id"), nullable=True)
    title:               Mapped[str]           = mapped_column(String(512), nullable=False)
    section_path:        Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    # e.g. ["Risk Factors", "Cloud Infrastructure"]
    section_order:       Mapped[int]           = mapped_column(Integer, default=0, nullable=False)

    # Relationships
    document_version: "DocumentVersion" = relationship("DocumentVersion", back_populates="sections")
    parent:           Optional["Section"] = relationship("Section", remote_side="Section.id")
    chunks:           List["Chunk"]       = relationship("Chunk", back_populates="section")

    def __repr__(self) -> str:
        return f"<Section id={self.id} title='{self.title}' order={self.section_order}>"


# ── chunks ────────────────────────────────────────────────────────────────────
class Chunk(Base):
    """
    A text chunk with its embedding vector.

    WHY chunk_hash UNIQUE?
    ───────────────────────
    chunk_hash is the SHA-256 of the chunk text. If the same text appears
    in two ingestion runs (e.g., document was re-ingested unchanged), the
    UNIQUE constraint on chunk_hash prevents duplicate rows. This is the
    core idempotency mechanism.

    Note: Two chunks with identical text but from different documents will
    collide on chunk_hash. This is intentional — if the text is identical,
    we don't need duplicate embedding rows. In practice, this is rare in
    due-diligence documents.

    If strict per-document deduplication is needed, change the unique
    constraint to (document_version_id, chunk_index) instead.

    EMBEDDING DIMENSION VALIDATION
    ────────────────────────────────
    The embedding column is vector(EMBEDDING_DIM). Before insertion, the
    ingestion pipeline validates:
        len(embedding) == EMBEDDING_DIM
    If the model dimension has changed (e.g., switched from bge-base to
    bge-small), this will raise a clear error instead of silently inserting
    mismatched vectors.

    DISTANCE METRIC: cosine (via <=> operator)
    ────────────────────────────────────────────
    BAAI/bge-base-en-v1.5 produces L2-normalized vectors.
    Cosine distance = 1 - cosine_similarity.
    pgvector <=> computes cosine distance for normalized vectors.
    The HNSW index is built with vector_cosine_ops for this metric.
    """
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("chunk_hash", name="uq_chunk_hash"),
    )

    id:                  Mapped[int]            = mapped_column(Integer, primary_key=True)
    document_version_id: Mapped[int]            = mapped_column(Integer, ForeignKey("document_versions.id"), nullable=False)
    section_id:          Mapped[Optional[int]]  = mapped_column(Integer, ForeignKey("sections.id"), nullable=True)
    chunk_index:         Mapped[int]            = mapped_column(Integer, nullable=False)
    text:                Mapped[str]            = mapped_column(Text, nullable=False)
    content_type:        Mapped[str]            = mapped_column(String(50), nullable=False, default="text")
    # Values: text, table, list, mixed (from Phase 2 ingestion)
    page_start:          Mapped[int]            = mapped_column(Integer, nullable=False, default=0)
    page_end:            Mapped[int]            = mapped_column(Integer, nullable=False, default=0)
    token_count:         Mapped[int]            = mapped_column(Integer, nullable=False, default=0)
    chunk_hash:          Mapped[str]            = mapped_column(String(64), nullable=False)
    # SHA-256 hex digest of text — used for deduplication
    embedding_model:     Mapped[Optional[str]]  = mapped_column(String(256), nullable=True)
    # e.g., "BAAI/bge-base-en-v1.5" — records which model generated this embedding
    embedding:           Mapped[Optional[list]] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    # pgvector column: vector(768) — stores the dense embedding
    created_at:          Mapped[datetime]       = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document_version: "DocumentVersion" = relationship("DocumentVersion", back_populates="chunks")
    section:          Optional["Section"] = relationship("Section", back_populates="chunks")

    def __repr__(self) -> str:
        has_emb = self.embedding is not None
        return (
            f"<Chunk id={self.id} index={self.chunk_index} "
            f"pages={self.page_start}-{self.page_end} "
            f"content_type='{self.content_type}' embedded={has_emb}>"
        )


# ── financial_metrics ─────────────────────────────────────────────────────────
class FinancialMetric(Base):
    """
    Structured financial data — NOT retrieved via vector search.

    WHY RELATIONAL?
    ────────────────
    Revenue for FY2025 is a precise numeric fact. Retrieving it via vector
    similarity risks returning the wrong year or a rounding artifact. Storing
    it relationally allows:
        SELECT value FROM financial_metrics
        WHERE company_id = $id AND metric = 'revenue' AND period = 'FY2025'

    This is faster, more precise, and auditable vs. vector retrieval.

    The combination of relational query (exact fact) + vector retrieval
    (narrative context) is the core Phase 4 architecture.

    period_type examples: "fiscal_year", "quarter", "half_year", "trailing_12m"
    metric examples: "revenue", "gross_profit", "operating_income", "net_income",
                     "operating_margin", "free_cash_flow", "net_revenue_retention",
                     "gross_revenue_retention", "r_and_d_expense", "employee_count"
    """
    __tablename__ = "financial_metrics"

    id:                        Mapped[int]            = mapped_column(Integer, primary_key=True)
    company_id:                Mapped[int]            = mapped_column(Integer, ForeignKey("companies.id"), nullable=False)
    metric:                    Mapped[str]            = mapped_column(String(100), nullable=False)
    period:                    Mapped[str]            = mapped_column(String(50), nullable=False)
    # e.g., "FY2023", "FY2024", "FY2025", "Q1FY2026"
    period_type:               Mapped[str]            = mapped_column(String(50), nullable=False, default="fiscal_year")
    value:                     Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unit:                      Mapped[Optional[str]]  = mapped_column(String(50), nullable=True)
    # e.g., "million", "percent", "count"
    currency:                  Mapped[Optional[str]]  = mapped_column(String(10), nullable=True)
    # e.g., "USD", "EUR"
    source_document_version_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("document_versions.id"), nullable=True
    )

    # Relationships
    company:                "Company"               = relationship("Company", back_populates="financial_metrics")
    source_document_version: Optional["DocumentVersion"] = relationship(
        "DocumentVersion", back_populates="financial_metrics"
    )

    def __repr__(self) -> str:
        return (
            f"<FinancialMetric id={self.id} metric='{self.metric}' "
            f"period='{self.period}' value={self.value} unit='{self.unit}'>"
        )


# ── ingestion_runs ────────────────────────────────────────────────────────────
class IngestionRun(Base):
    """
    Audit log for each ingestion run.

    WHY: Provides a history of what was ingested, when, and with what result.
    Useful for:
      - Diagnosing why a particular chunk appears or doesn't appear
      - Verifying that idempotent re-ingestion didn't create duplicates
      - Comparing ingestion strategies (fixed vs structure_aware vs pg)
    """
    __tablename__ = "ingestion_runs"

    id:               Mapped[int]           = mapped_column(Integer, primary_key=True)
    run_at:           Mapped[datetime]      = mapped_column(DateTime(timezone=True), server_default=func.now())
    strategy:         Mapped[str]           = mapped_column(String(100), nullable=False)
    # e.g., "structure_aware_pg", "fixed_pg"
    pdf_file:         Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    chunks_created:   Mapped[int]           = mapped_column(Integer, default=0)
    chunks_skipped:   Mapped[int]           = mapped_column(Integer, default=0)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status:           Mapped[str]           = mapped_column(String(50), nullable=False, default="success")
    # Values: "success", "error", "partial"
    notes:            Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<IngestionRun id={self.id} status='{self.status}' "
            f"chunks_created={self.chunks_created} at={self.run_at}>"
        )


# ── Index definitions ─────────────────────────────────────────────────────────
# These are created via Alembic migration (not here) so they can be managed
# independently from the schema. Defined here as reference documentation only.

# HNSW index on chunks.embedding (cosine distance, m=16, ef_construction=64)
# Created in migration 0001_initial_schema.py:
#   CREATE INDEX chunks_embedding_hnsw_idx ON chunks
#   USING hnsw (embedding vector_cosine_ops)
#   WITH (m = 16, ef_construction = 64);

# GIN index on chunks.text_tsv (full-text search)
# Created in migration 0001_initial_schema.py:
#   ALTER TABLE chunks ADD COLUMN text_tsv tsvector
#       GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
#   CREATE INDEX chunks_text_tsv_idx ON chunks USING GIN (text_tsv);

# B-tree indexes on foreign keys (auto-created by PostgreSQL for PK, not FK)
# Created in migration:
#   CREATE INDEX idx_chunks_document_version_id ON chunks(document_version_id);
#   CREATE INDEX idx_document_versions_document_id ON document_versions(document_id);
#   CREATE INDEX idx_documents_company_id ON documents(company_id);
