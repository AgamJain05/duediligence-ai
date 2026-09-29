"""
src/db/repositories/companies.py  ─  Company persistence
══════════════════════════════════════════════════════════
Phase 4: CRUD operations for the companies table.

get_or_create_company() is the idempotency entry point for ingestion.
The slug uniqueness constraint means the same company name always maps
to the same row, preventing duplicates even if ingested multiple times.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from src.db.models import Company


def _slugify(name: str) -> str:
    """
    Generate a URL-safe slug from a company name.

    Uses python-slugify if available, falls back to a simple implementation.
    Example: "OrionVault Systems" → "orionvault-systems"
    """
    try:
        from slugify import slugify
        return slugify(name)
    except ImportError:
        # Minimal fallback
        import re
        s = name.lower()
        s = re.sub(r"[^a-z0-9]+", "-", s)
        return s.strip("-")


def get_or_create_company(db: Session, name: str) -> Company:
    """
    Return existing company by slug, or create a new one.

    This is idempotent: calling it with the same name twice returns the
    same Company row — no duplicates.

    Args:
        db:   Active SQLAlchemy session
        name: Full company name (e.g., "OrionVault Systems")

    Returns:
        Company ORM instance (persisted)
    """
    slug = _slugify(name)
    company = db.query(Company).filter_by(slug=slug).first()
    if company:
        return company

    company = Company(name=name, slug=slug)
    db.add(company)
    db.flush()  # Get the generated id without full commit
    return company


def get_company_by_slug(db: Session, slug: str) -> Optional[Company]:
    """Return company by slug, or None if not found."""
    return db.query(Company).filter_by(slug=slug).first()


def get_company_by_name(db: Session, name: str) -> Optional[Company]:
    """Return company by exact name match."""
    return db.query(Company).filter_by(name=name).first()


def list_companies(db: Session) -> list[Company]:
    """Return all companies ordered by name."""
    return db.query(Company).order_by(Company.name).all()
