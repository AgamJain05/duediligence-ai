"""
scripts/inspect_db.py  ─  Phase 4 database state inspector
════════════════════════════════════════════════════════════
Reports on the state of the PostgreSQL knowledge base.

Output includes:
  • Companies with document counts
  • Documents with version counts
  • Document versions with is_current flag
  • Section counts per version
  • Chunk counts (total, embedded)
  • Financial metric rows per company
  • Ingestion run history
  • Database health (pgvector available, HNSW index status)

Usage:
    python scripts/inspect_db.py
    python scripts/inspect_db.py --company "OrionVault Systems"
    python scripts/inspect_db.py --verbose
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from sqlalchemy import text
from src.db.connection import get_db, ping_db, check_pgvector
from src.db.models import Company, Document, DocumentVersion, Section, Chunk, FinancialMetric, IngestionRun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Phase 4 PostgreSQL knowledge base.")
    parser.add_argument("--company", type=str, default=None, help="Filter by company name")
    parser.add_argument("--verbose", action="store_true", help="Show chunk-level detail")
    return parser.parse_args()


def print_header(title: str, width: int = 68) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print('=' * width)


def main() -> None:
    args = parse_args()
    bar = "=" * 68

    print(f"\n{bar}")
    print("  DueDiligence-AI  |  Phase 4 Database Inspector")
    print(bar)

    # ── Health check ──────────────────────────────────────────────────────────
    print("\n[1] Database Health")
    print("-" * 40)
    db_ok = ping_db()
    pv_ok = check_pgvector()
    print(f"  PostgreSQL:  {'✓ reachable' if db_ok else '✗ unreachable'}")
    print(f"  pgvector:    {'✓ installed' if pv_ok else '✗ missing'}")

    if not db_ok:
        print("\n[ERROR] Cannot connect. Check DATABASE_URL in .env")
        sys.exit(1)

    with get_db() as db:
        # ── HNSW index status ─────────────────────────────────────────────────
        idx_result = db.execute(text("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'chunks'
              AND indexname = 'chunks_embedding_hnsw_idx'
        """)).fetchone()
        if idx_result:
            print(f"  HNSW index:  ✓ exists")
            if args.verbose:
                print(f"    Definition: {idx_result.indexdef}")
        else:
            print(f"  HNSW index:  ✗ missing (run migrations)")

        # ── Companies ─────────────────────────────────────────────────────────
        print_header("Companies")
        companies = db.query(Company).order_by(Company.name).all()
        if not companies:
            print("  (no companies — run ingest_pg.py)")
        for co in companies:
            if args.company and args.company.lower() not in co.name.lower():
                continue
            doc_count = db.query(Document).filter_by(company_id=co.id).count()
            print(f"  [{co.id}] {co.name}  (slug: {co.slug})  docs: {doc_count}")

        # ── Documents ─────────────────────────────────────────────────────────
        print_header("Documents")
        docs_q = db.query(Document).order_by(Document.document_type, Document.title)
        if args.company:
            docs_q = docs_q.join(Company).filter(Company.name.ilike(f"%{args.company}%"))
        documents = docs_q.all()
        if not documents:
            print("  (no documents)")
        for doc in documents:
            ver_count = db.query(DocumentVersion).filter_by(document_id=doc.id).count()
            print(f"  [{doc.id}] [{doc.document_type}] {doc.title}  versions: {ver_count}")

        # ── Document Versions ─────────────────────────────────────────────────
        print_header("Document Versions")
        versions = (
            db.query(DocumentVersion)
            .join(Document)
            .order_by(Document.title, DocumentVersion.effective_date)
            .all()
        )
        if not versions:
            print("  (no versions)")
        for ver in versions:
            current_flag = " ← CURRENT" if ver.is_current else ""
            chunk_count  = db.query(Chunk).filter_by(document_version_id=ver.id).count()
            emb_count    = db.query(Chunk).filter(
                Chunk.document_version_id == ver.id,
                Chunk.embedding.isnot(None)
            ).count()
            sec_count    = db.query(Section).filter_by(document_version_id=ver.id).count()
            print(
                f"  [{ver.id}] v={ver.version_label}  "
                f"effective={ver.effective_date}  published={ver.published_date}"
                f"{current_flag}"
            )
            print(
                f"       doc_id={ver.document_id}  "
                f"chunks={chunk_count} (embedded={emb_count})  "
                f"sections={sec_count}"
            )
            print(f"       file={ver.source_filename}  checksum={ver.checksum[:16] + '...' if ver.checksum else 'None'}")

        # ── Summary counts ────────────────────────────────────────────────────
        print_header("Corpus Summary")
        total_chunks  = db.query(Chunk).count()
        emb_chunks    = db.query(Chunk).filter(Chunk.embedding.isnot(None)).count()
        total_sections = db.query(Section).count()
        print(f"  Total chunks    : {total_chunks}")
        print(f"  Embedded chunks : {emb_chunks}  ({emb_chunks/max(total_chunks,1)*100:.0f}%)")
        print(f"  Total sections  : {total_sections}")

        # Check embedding dimension
        sample = db.query(Chunk).filter(Chunk.embedding.isnot(None)).first()
        if sample and sample.embedding is not None:
            dim = len(sample.embedding)
            print(f"  Embedding dim   : {dim}  (model: {sample.embedding_model})")

        # ── Financial Metrics ─────────────────────────────────────────────────
        print_header("Financial Metrics")
        fm_rows = (
            db.query(FinancialMetric)
            .join(Company)
            .order_by(Company.name, FinancialMetric.metric, FinancialMetric.period)
            .all()
        )
        if not fm_rows:
            print("  (no financial metrics)")
        else:
            current_co = None
            for fm in fm_rows:
                co_name = fm.company.name if fm.company else f"co_id={fm.company_id}"
                if co_name != current_co:
                    print(f"\n  {co_name}:")
                    current_co = co_name
                val_str = f"{fm.value:,.2f}" if fm.value is not None else "None"
                currency_str = f" {fm.currency}" if fm.currency else ""
                print(f"    {fm.metric:<30} {fm.period:<10} {val_str:>12} {fm.unit or ''}{currency_str}")

        # ── Ingestion Runs ────────────────────────────────────────────────────
        print_header("Ingestion Run History")
        runs = db.query(IngestionRun).order_by(IngestionRun.run_at.desc()).limit(10).all()
        if not runs:
            print("  (no ingestion runs)")
        for run in runs:
            print(
                f"  [{run.id}] {run.run_at.strftime('%Y-%m-%d %H:%M:%S')}  "
                f"status={run.status}  "
                f"created={run.chunks_created}  skipped={run.chunks_skipped}  "
                f"duration={run.duration_seconds:.1f}s"
            )
            if run.notes:
                print(f"       {run.notes}")

        # ── Chunk detail (verbose) ────────────────────────────────────────────
        if args.verbose:
            print_header("Chunk Detail (sample: first 5 per version)")
            for ver in versions:
                print(f"\n  Version [{ver.id}] {ver.version_label}:")
                chunks = (
                    db.query(Chunk)
                    .filter_by(document_version_id=ver.id)
                    .order_by(Chunk.chunk_index)
                    .limit(5)
                    .all()
                )
                for ch in chunks:
                    emb_ok = "✓" if ch.embedding is not None else "✗"
                    print(
                        f"    [{ch.chunk_index}] pages={ch.page_start}-{ch.page_end}  "
                        f"type={ch.content_type}  tokens={ch.token_count}  "
                        f"emb={emb_ok}  hash={ch.chunk_hash[:8]}..."
                    )
                    print(f"         {ch.text[:80].replace(chr(10), ' ')}...")

    print(f"\n{'=' * 68}\n")


if __name__ == "__main__":
    main()
