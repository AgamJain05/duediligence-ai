"""
alembic/versions/0001_initial_schema.py  ─  Phase 4 initial database schema
══════════════════════════════════════════════════════════════════════════════
Creates all Phase 4 tables, indexes, constraints, and pgvector extension.

Migration: 0001 (no previous migration)
Created: Phase 4 implementation

WHAT THIS MIGRATION DOES
─────────────────────────
1. Enables pgvector extension (safe with IF NOT EXISTS — Supabase pre-installs it)
2. Creates tables in dependency order:
   companies → sources → documents → document_versions → sections → chunks
   companies → financial_metrics
   (no FK dependencies) → ingestion_runs

3. Creates indexes:
   - HNSW index on chunks.embedding (vector_cosine_ops, m=16, ef_construction=64)
   - GIN index on chunks.text_tsv (full-text search)
   - B-tree indexes on all foreign key columns

HNSW PARAMETERS EXPLAINED
───────────────────────────
m = 16:
  Number of bidirectional links created for each new node during construction.
  This determines graph connectivity. Higher m → better recall at query time,
  more memory consumed. Range: 4–64. Default and recommended starting value: 16.

ef_construction = 64:
  Size of the candidate list during HNSW index BUILD. Controls build quality:
  higher values build a better graph but take longer. 64 is conservative;
  use 128+ for production datasets >100k chunks if recall matters.
  This parameter does NOT affect query latency after the index is built.

ef_search = 100 (set in connection.py):
  Size of the dynamic candidate list during QUERY. This IS the primary
  lever for recall vs latency at query time. The default in pgvector is 40.
  We increase to 100 because this corpus prioritizes recall over speed.
  Set per-session via: SET hnsw.ef_search = 100;
  Our connection.py sets this on every new connection automatically.

FULL-TEXT SEARCH
─────────────────
We add a generated tsvector column to chunks for PostgreSQL native FTS.
This allows comparison between:
  - Phase 3: BM25 (in-memory Python, using rank-bm25)
  - Phase 4: PostgreSQL tsvector/tsquery with GIN index

Both use 'english' configuration for stemming and stopword removal.

DOWNGRADE
─────────
drop_all() removes all tables in reverse dependency order.
The pgvector extension is NOT dropped on downgrade to avoid accidentally
breaking other projects that depend on it.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# Alembic revision identifiers
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. Enable pgvector extension ──────────────────────────────────────────
    # Supabase pre-installs pgvector, but this is idempotent (IF NOT EXISTS)
    # and documents the dependency explicitly.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── 2. Create companies table ─────────────────────────────────────────────
    op.create_table(
        "companies",
        sa.Column("id",         sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("name",       sa.String(255), nullable=False),
        sa.Column("slug",       sa.String(255), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_companies_slug", "companies", ["slug"])

    # ── 3. Create sources table ───────────────────────────────────────────────
    op.create_table(
        "sources",
        sa.Column("id",           sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("source_name",  sa.String(512), nullable=False),
        sa.Column("source_uri",   sa.Text,        nullable=True),
        sa.Column("source_type",  sa.String(50),  nullable=False, server_default="local_file"),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("checksum",     sa.String(64),  nullable=True),
    )

    # ── 4. Create documents table ─────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id",            sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("company_id",    sa.Integer,     sa.ForeignKey("companies.id"), nullable=False),
        sa.Column("title",         sa.String(512), nullable=False),
        sa.Column("document_type", sa.String(100), nullable=False),
        sa.Column("created_at",    sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at",    sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_documents_company_id",    "documents", ["company_id"])
    op.create_index("idx_documents_document_type", "documents", ["document_type"])

    # ── 5. Create document_versions table ─────────────────────────────────────
    op.create_table(
        "document_versions",
        sa.Column("id",              sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("document_id",     sa.Integer,     sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("source_id",       sa.Integer,     sa.ForeignKey("sources.id"),   nullable=True),
        sa.Column("version_label",   sa.String(50),  nullable=False),
        sa.Column("effective_date",  sa.Date,        nullable=True),
        sa.Column("published_date",  sa.Date,        nullable=True),
        sa.Column("source_filename", sa.String(512), nullable=True),
        sa.Column("source_uri",      sa.Text,        nullable=True),
        sa.Column("checksum",        sa.String(64),  nullable=True),
        sa.Column("is_current",      sa.Boolean,     nullable=False, server_default="false"),
        sa.Column("created_at",      sa.DateTime(timezone=True), server_default=sa.func.now()),
        # UNIQUENESS: same document cannot have two versions with identical content
        sa.UniqueConstraint("document_id", "checksum", name="uq_docver_document_checksum"),
    )
    op.create_index("idx_docver_document_id",   "document_versions", ["document_id"])
    op.create_index("idx_docver_effective_date", "document_versions", ["effective_date"])
    op.create_index("idx_docver_is_current",     "document_versions", ["is_current"])

    # ── 6. Create sections table ──────────────────────────────────────────────
    op.create_table(
        "sections",
        sa.Column("id",                  sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("document_version_id", sa.Integer,     sa.ForeignKey("document_versions.id"), nullable=False),
        sa.Column("parent_section_id",   sa.Integer,     sa.ForeignKey("sections.id"),          nullable=True),
        sa.Column("title",               sa.String(512), nullable=False),
        sa.Column("section_path",        JSONB,          nullable=True),
        sa.Column("section_order",       sa.Integer,     nullable=False, server_default="0"),
    )
    op.create_index("idx_sections_document_version_id", "sections", ["document_version_id"])

    # ── 7. Create chunks table with pgvector column ───────────────────────────
    # We use raw SQL for the vector column because SQLAlchemy's Column() does not
    # natively know the pgvector type without pgvector.sqlalchemy registration.
    # The Vector type from pgvector.sqlalchemy IS registered in models.py, but
    # for migration DDL we use raw SQL to be explicit and self-documenting.

    # First create the table without the embedding column
    op.create_table(
        "chunks",
        sa.Column("id",                  sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("document_version_id", sa.Integer,     sa.ForeignKey("document_versions.id"), nullable=False),
        sa.Column("section_id",          sa.Integer,     sa.ForeignKey("sections.id"),          nullable=True),
        sa.Column("chunk_index",         sa.Integer,     nullable=False),
        sa.Column("text",                sa.Text,        nullable=False),
        sa.Column("content_type",        sa.String(50),  nullable=False, server_default="text"),
        sa.Column("page_start",          sa.Integer,     nullable=False, server_default="0"),
        sa.Column("page_end",            sa.Integer,     nullable=False, server_default="0"),
        sa.Column("token_count",         sa.Integer,     nullable=False, server_default="0"),
        sa.Column("chunk_hash",          sa.String(64),  nullable=False),
        sa.Column("embedding_model",     sa.String(256), nullable=True),
        sa.Column("created_at",          sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("chunk_hash", name="uq_chunk_hash"),
    )

    # Add vector column via raw SQL (pgvector syntax)
    # EMBEDDING_DIM = 768 for BAAI/bge-base-en-v1.5
    op.execute("ALTER TABLE chunks ADD COLUMN embedding vector(768)")

    # ── 8. Create HNSW index ──────────────────────────────────────────────────
    # HNSW: Hierarchical Navigable Small World graph.
    # vector_cosine_ops: uses cosine distance (<=> operator).
    # m=16: default bidirectional link count per node (good balance).
    # ef_construction=64: build-time candidate list size (faster build, good quality).
    #
    # WHY HNSW over IVFFlat?
    # HNSW: no training phase needed, better recall at low ef_search,
    #       higher memory usage, faster queries.
    # IVFFlat: needs training (cluster assignment), lower memory, slower at low nprobe.
    # For this corpus size (~30 chunks), the difference is minimal but HNSW is
    # the better default for production use.
    op.execute("""
        CREATE INDEX chunks_embedding_hnsw_idx
        ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # ── 9. Add full-text search support ──────────────────────────────────────
    # Generated tsvector column — automatically updated when text changes.
    # 'english' config: applies English stemming and removes stopwords.
    # STORED: computed at INSERT time and stored physically (not computed at query time).
    op.execute("""
        ALTER TABLE chunks
        ADD COLUMN text_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
    """)

    # GIN index on tsvector — required for efficient full-text search.
    # GIN (Generalized Inverted Index) is optimal for tsvector because it
    # indexes each lexeme separately, allowing O(log N) lookup per word.
    op.execute("CREATE INDEX chunks_text_tsv_idx ON chunks USING GIN (text_tsv)")

    # B-tree indexes on FK columns (not auto-created by PostgreSQL for non-PK FKs)
    op.create_index("idx_chunks_document_version_id", "chunks", ["document_version_id"])
    op.create_index("idx_chunks_section_id",          "chunks", ["section_id"])
    op.create_index("idx_chunks_content_type",         "chunks", ["content_type"])
    op.create_index("idx_chunks_page_start",           "chunks", ["page_start"])

    # ── 10. Create financial_metrics table ────────────────────────────────────
    op.create_table(
        "financial_metrics",
        sa.Column("id",                         sa.Integer,    primary_key=True, autoincrement=True),
        sa.Column("company_id",                 sa.Integer,    sa.ForeignKey("companies.id"), nullable=False),
        sa.Column("metric",                     sa.String(100), nullable=False),
        sa.Column("period",                     sa.String(50),  nullable=False),
        sa.Column("period_type",                sa.String(50),  nullable=False, server_default="fiscal_year"),
        sa.Column("value",                      sa.Float,       nullable=True),
        sa.Column("unit",                       sa.String(50),  nullable=True),
        sa.Column("currency",                   sa.String(10),  nullable=True),
        sa.Column("source_document_version_id", sa.Integer,    sa.ForeignKey("document_versions.id"), nullable=True),
    )
    op.create_index("idx_financial_company_id", "financial_metrics", ["company_id"])
    op.create_index("idx_financial_metric",     "financial_metrics", ["metric"])
    op.create_index("idx_financial_period",     "financial_metrics", ["period"])

    # ── 11. Create ingestion_runs table ───────────────────────────────────────
    op.create_table(
        "ingestion_runs",
        sa.Column("id",               sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("run_at",           sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("strategy",         sa.String(100), nullable=False),
        sa.Column("pdf_file",         sa.String(512), nullable=True),
        sa.Column("chunks_created",   sa.Integer,     nullable=False, server_default="0"),
        sa.Column("chunks_skipped",   sa.Integer,     nullable=False, server_default="0"),
        sa.Column("duration_seconds", sa.Float,       nullable=True),
        sa.Column("status",           sa.String(50),  nullable=False, server_default="success"),
        sa.Column("notes",            sa.Text,        nullable=True),
    )


def downgrade() -> None:
    """Drop all Phase 4 tables in reverse dependency order."""
    # Drop indexes first
    op.execute("DROP INDEX IF EXISTS chunks_text_tsv_idx")
    op.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw_idx")

    # Drop tables in reverse FK order
    op.drop_table("ingestion_runs")
    op.drop_table("financial_metrics")
    op.drop_table("chunks")
    op.drop_table("sections")
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.drop_table("sources")
    op.drop_table("companies")

    # NOTE: pgvector extension is NOT dropped on downgrade.
    # Dropping it would break any other projects using pgvector in the same DB.
