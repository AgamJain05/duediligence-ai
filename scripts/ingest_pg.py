"""
scripts/ingest_pg.py  ─  Phase 4 PostgreSQL ingestion pipeline
═══════════════════════════════════════════════════════════════════
Ingests a PDF document into the PostgreSQL + pgvector knowledge base.

PIPELINE FLOW
──────────────
PDF file
  ↓ Phase 2 parse (pdfplumber + structure detection)
  ↓ identify/create company
  ↓ identify/create document (title + type)
  ↓ compute file checksum (SHA-256)
  ↓ check: has this checksum been ingested for this document already?
    → YES (same file):  skip, log "already ingested" → EXIT
    → YES, same checksum but different doc: create new version
    → NO:  create document_version record
  ↓ upsert sections from Phase 2 section hierarchy
  ↓ generate embeddings (same model as Phase 1–3)
  ↓ validate embedding dimension == EMBEDDING_DIM
  ↓ upsert chunks with embeddings (skip exact-text duplicates via chunk_hash)
  ↓ extract + seed financial_metrics via regex
  ↓ record ingestion_run audit entry

IDEMPOTENCY CASES
──────────────────
Case 1 — First ingestion:
  All records created. Ingestion run status="success".

Case 2 — Same file ingested again (no change):
  checksum matches existing document_version → skip all chunk/section work.
  Status="skipped_duplicate". Zero new records created.

Case 3 — Modified file (new version):
  checksum differs → create new document_version.
  New chunks inserted; unchanged chunks skipped via chunk_hash.
  Status="new_version".

Case 4 — New document type for same company:
  Different document_type → new Document row + new DocumentVersion.

Usage:
    cd duediligence-ai

    # Ingest Dossier as v1 (2024)
    python scripts/ingest_pg.py \\
        --pdf data/raw/OrionVault_Systems_Due_Diligence_Dossier.pdf \\
        --company "OrionVault Systems" \\
        --document-type annual_review \\
        --version-label v1 \\
        --effective-date 2024-01-01 \\
        --published-date 2024-03-15

    # Ingest Phase3+ corpus as v2 (current)
    python scripts/ingest_pg.py \\
        --pdf data/raw/OrionVault_RAG_Phase3plus_Benchmark_Corpus.pdf \\
        --company "OrionVault Systems" \\
        --document-type annual_review \\
        --version-label v2 \\
        --effective-date 2026-01-01 \\
        --published-date 2026-03-01 \\
        --is-current

    # Dry run (no DB writes)
    python scripts/ingest_pg.py --pdf data/raw/... --dry-run
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import date
from pathlib import Path

# ── Windows encoding fix ──────────────────────────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Phase 2 imports ───────────────────────────────────────────────────────────
from src.ingestion.pdf_parser  import parse_pdf
from src.ingestion.chunking    import build_structured_chunks
from src.ingestion.validation  import validate_chunks, print_validation_summary

# ── Phase 4 imports ───────────────────────────────────────────────────────────
from src.embeddings import embed_texts, get_embedding_dim
from src.db.connection import get_db, ping_db, check_pgvector
from src.db.models import EMBEDDING_DIM, IngestionRun
from src.db.repositories.companies  import get_or_create_company
from src.db.repositories.documents  import (
    get_or_create_document,
    compute_file_checksum,
    get_version_by_checksum,
    create_document_version,
    get_or_create_source,
)
from src.db.repositories.sections   import build_section_map
from src.db.repositories.chunks     import bulk_upsert_chunks, validate_embedding_dim
from src.db.repositories.financials import seed_financials_from_chunks

import numpy as np


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: Ingest a PDF into PostgreSQL + pgvector knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument("--pdf", type=Path, required=True,
                        help="Path to the PDF file to ingest")
    parser.add_argument("--company", type=str, default="OrionVault Systems",
                        help="Company name (default: OrionVault Systems)")
    parser.add_argument("--document-type", type=str, default="annual_review",
                        choices=["annual_review", "risk_memo", "earnings_call",
                                 "technical_brief", "policy", "incident_report",
                                 "case_study", "benchmark_corpus"],
                        help="Document type (default: annual_review)")
    parser.add_argument("--title", type=str, default=None,
                        help="Document title (default: derived from filename)")
    parser.add_argument("--version-label", type=str, default="v1",
                        help="Version label (e.g., v1, v2, 2024, FY2026)")
    parser.add_argument("--effective-date", type=str, default=None,
                        help="Effective date YYYY-MM-DD (when this version became applicable)")
    parser.add_argument("--published-date", type=str, default=None,
                        help="Published date YYYY-MM-DD (when the document was released)")
    parser.add_argument("--is-current", action="store_true",
                        help="Mark this as the current (default) version")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Target tokens per chunk (default: 500)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and embed without writing to database")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")
    return parser.parse_args()


# ── Main ingestion flow ───────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    bar = "=" * 68

    print(f"\n{bar}")
    print("  DueDiligence-AI  |  Phase 4 — PostgreSQL Ingestion Pipeline")
    print(f"  PDF     : {args.pdf.name}")
    print(f"  Company : {args.company}")
    print(f"  Type    : {args.document_type}")
    print(f"  Version : {args.version_label}")
    print(f"  Current : {args.is_current}")
    print(f"  Dry run : {args.dry_run}")
    print(bar)

    t_start = time.perf_counter()

    # ── Step 0: Validate PDF exists ───────────────────────────────────────────
    if not args.pdf.exists():
        print(f"\n[ERROR] PDF not found: {args.pdf}")
        sys.exit(1)

    # ── Step 1: Database connectivity check ───────────────────────────────────
    if not args.dry_run:
        print("\n[STEP 1/8]  Checking database connectivity...")
        if not ping_db():
            print("[ERROR] Cannot connect to database. Check DATABASE_URL in .env")
            sys.exit(1)
        if not check_pgvector():
            print("[ERROR] pgvector extension not available. Enable it in Supabase dashboard.")
            sys.exit(1)
        print("  [OK] Database reachable, pgvector available.")
    else:
        print("\n[STEP 1/8]  [DRY RUN] Skipping DB connectivity check.")

    # ── Step 2: Compute checksum ──────────────────────────────────────────────
    print(f"\n[STEP 2/8]  Computing file checksum...")
    checksum = compute_file_checksum(args.pdf)
    print(f"  [OK] SHA-256: {checksum[:16]}...{checksum[-8:]}")

    # ── Step 3: Parse PDF ─────────────────────────────────────────────────────
    print(f"\n[STEP 3/8]  Parsing PDF with Phase 2 structure-aware parser...")
    doc = parse_pdf(args.pdf)
    print(f"  [OK] {doc.total_pages} pages parsed. Title: '{doc.title}'")

    # Determine document title
    title = args.title or doc.title or args.pdf.stem

    # ── Step 4: Build structured chunks ──────────────────────────────────────
    print(f"\n[STEP 4/8]  Building structure-aware chunks (target={args.chunk_size} tokens)...")
    raw_chunks = build_structured_chunks(doc, target_tokens=args.chunk_size)
    valid_chunks, summary = validate_chunks(raw_chunks)
    print_validation_summary(summary)

    if not valid_chunks:
        print("[ERROR] No valid chunks after validation.")
        sys.exit(1)

    chunk_dicts = [c.to_dict() for c in valid_chunks]
    print(f"  [OK] {len(chunk_dicts)} valid chunks.")

    # ── Step 5: Generate embeddings ───────────────────────────────────────────
    print(f"\n[STEP 5/8]  Generating embeddings...")
    _actual_dim = get_embedding_dim()
    if _actual_dim != EMBEDDING_DIM:
        print(
            f"[ERROR] Embedding model produces {_actual_dim}-dim vectors, "
            f"but EMBEDDING_DIM={EMBEDDING_DIM}. Update EMBEDDING_DIM in .env."
        )
        sys.exit(1)

    texts = [c["text"] for c in chunk_dicts]
    embeddings = embed_texts(texts)
    print(f"  [OK] Embeddings shape: {embeddings.shape}  (dim={embeddings.shape[1]})")

    if args.dry_run:
        print(f"\n[DRY RUN] Would insert {len(chunk_dicts)} chunks into PostgreSQL.")
        print(f"[DRY RUN] Embedding model: {_get_model_name()}")
        print(f"[DRY RUN] Done (no DB writes).")
        return

    # ── Step 6: Database operations ───────────────────────────────────────────
    print(f"\n[STEP 6/8]  Writing to database...")

    with get_db() as db:
        # 6a. Company
        company = get_or_create_company(db, args.company)
        print(f"  Company: id={company.id} slug='{company.slug}'")

        # 6b. Document
        document = get_or_create_document(
            db,
            company_id=company.id,
            title=title,
            document_type=args.document_type,
        )
        print(f"  Document: id={document.id} title='{document.title}'")

        # 6c. Check for duplicate ingestion
        existing_version = get_version_by_checksum(db, document.id, checksum)
        if existing_version:
            print(
                f"\n  [SKIP] This exact file has already been ingested.\n"
                f"  Existing version: id={existing_version.id} "
                f"label='{existing_version.version_label}' "
                f"effective={existing_version.effective_date}\n"
                f"  To ingest a new version, modify the file first (different checksum)."
            )
            # Record the skipped run
            _record_ingestion_run(
                db=db,
                pdf_file=str(args.pdf),
                strategy="structure_aware_pg",
                chunks_created=0,
                chunks_skipped=len(chunk_dicts),
                duration=time.perf_counter() - t_start,
                status="skipped_duplicate",
                notes=f"Duplicate checksum {checksum[:16]}... → version id={existing_version.id}",
            )
            print(f"\n  [OK] Ingestion run recorded (status=skipped_duplicate).")
            return

        # 6d. Create source record
        source = get_or_create_source(
            db,
            source_name=args.pdf.name,
            source_uri=str(args.pdf.resolve()),
            source_type="local_file",
            checksum=checksum,
        )

        # 6e. Parse dates
        eff_date  = _parse_date(args.effective_date)  if args.effective_date  else None
        pub_date  = _parse_date(args.published_date)  if args.published_date  else None

        # 6f. Create document version
        doc_version = create_document_version(
            db=db,
            document_id=document.id,
            version_label=args.version_label,
            source_filename=args.pdf.name,
            checksum=checksum,
            effective_date=eff_date,
            published_date=pub_date,
            source_uri=str(args.pdf.resolve()),
            is_current=args.is_current,
            source_id=source.id,
        )
        print(f"  DocumentVersion: id={doc_version.id} label='{doc_version.version_label}' "
              f"is_current={doc_version.is_current}")

        # 6g. Build section map
        print(f"\n[STEP 7/8]  Persisting section hierarchy...")
        section_map = build_section_map(db, doc_version.id, chunk_dicts)
        print(f"  [OK] {len(section_map)} unique sections persisted.")

        # 6h. Upsert chunks + embeddings
        print(f"\n[STEP 8/8]  Upserting {len(chunk_dicts)} chunks with embeddings...")
        model_name = _get_model_name()
        created, skipped = bulk_upsert_chunks(
            db=db,
            document_version_id=doc_version.id,
            chunk_dicts=chunk_dicts,
            embeddings=embeddings,
            section_map=section_map,
            embedding_model=model_name,
        )
        print(f"  [OK] Created: {created}  Skipped (dedup): {skipped}")

        # 6i. Seed financial metrics
        print(f"\n  Extracting and seeding financial metrics...")
        fm_created, fm_updated = seed_financials_from_chunks(
            db=db,
            company_id=company.id,
            chunk_dicts=chunk_dicts,
            document_version_id=doc_version.id,
        )
        print(f"  [OK] Financial metrics: {fm_created} new, {fm_updated} updated.")

        # 6j. Record ingestion run
        status = "new_version" if args.version_label != "v1" else "success"
        _record_ingestion_run(
            db=db,
            pdf_file=str(args.pdf),
            strategy="structure_aware_pg",
            chunks_created=created,
            chunks_skipped=skipped,
            duration=time.perf_counter() - t_start,
            status=status,
            notes=f"version_id={doc_version.id} label={args.version_label}",
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    duration = time.perf_counter() - t_start
    print(f"\n{bar}")
    print("  Ingestion complete!")
    print(f"  PDF           : {args.pdf.name}")
    print(f"  Company       : {args.company}")
    print(f"  Document type : {args.document_type}")
    print(f"  Version       : {args.version_label} (is_current={args.is_current})")
    print(f"  Chunks        : {created} created, {skipped} skipped")
    print(f"  Financials    : {fm_created} new, {fm_updated} updated")
    print(f"  Duration      : {duration:.1f}s")
    print(bar)
    print("\nNext steps:")
    print("  Inspect DB     : python scripts/inspect_db.py")
    print("  Inspect query  : python scripts/inspect_query.py 'key risks'")
    print("  Compare phases : python scripts/compare_phases.py")
    print(bar + "\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> date:
    """Parse YYYY-MM-DD string to date object."""
    from datetime import datetime
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _get_model_name() -> str:
    """Return the current embedding model name."""
    import os
    return os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")


def _record_ingestion_run(
    db,
    pdf_file: str,
    strategy: str,
    chunks_created: int,
    chunks_skipped: int,
    duration: float,
    status: str,
    notes: str = "",
) -> None:
    """Write an audit record to ingestion_runs."""
    run = IngestionRun(
        strategy=strategy,
        pdf_file=pdf_file,
        chunks_created=chunks_created,
        chunks_skipped=chunks_skipped,
        duration_seconds=round(duration, 2),
        status=status,
        notes=notes,
    )
    db.add(run)
    db.flush()


if __name__ == "__main__":
    main()
