"""
tests/db/test_versioning.py  ─  Version management tests
══════════════════════════════════════════════════════════
Tests the two key version query functions:
  - get_current_document_version
  - get_document_version_at_date

Also verifies the documented precedence rules for temporal queries.
"""

from __future__ import annotations

from datetime import date

import pytest

from tests.db.conftest import requires_db


def _setup_versioned_doc(db_session):
    """Helper: Create company + document + v1 (2024) + v2 (2026, current)."""
    from src.db.repositories.companies import get_or_create_company
    from src.db.repositories.documents import (
        get_or_create_document, create_document_version
    )

    co  = get_or_create_company(db_session, "VersionQueryCo")
    doc = get_or_create_document(db_session, co.id, "Annual Review", "annual_review")

    v1 = create_document_version(
        db_session, doc.id, "v1", "dossier_2024.pdf", "cs_v1_2024",
        effective_date=date(2024, 1, 1),
        published_date=date(2024, 3, 15),
        is_current=False,
    )
    v2 = create_document_version(
        db_session, doc.id, "v2", "corpus_2026.pdf", "cs_v2_2026",
        effective_date=date(2026, 1, 1),
        published_date=date(2026, 3, 1),
        is_current=True,
    )
    return co, doc, v1, v2


@requires_db
class TestCurrentVersionLookup:
    """get_current_document_version always returns the is_current=True version."""

    def test_returns_current_version(self, db_session):
        from src.db.repositories.documents import get_current_document_version
        _, doc, v1, v2 = _setup_versioned_doc(db_session)

        current = get_current_document_version(db_session, doc.id)
        assert current is not None
        assert current.id == v2.id
        assert current.version_label == "v2"
        assert current.is_current is True

    def test_returns_none_if_no_current(self, db_session):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import (
            get_or_create_document, create_document_version, get_current_document_version
        )

        co  = get_or_create_company(db_session, "NoCurrentCo")
        doc = get_or_create_document(db_session, co.id, "No Current Doc", "policy")
        _   = create_document_version(db_session, doc.id, "v1", "f.pdf", "no_current_cs",
                                       is_current=False)

        current = get_current_document_version(db_session, doc.id)
        assert current is None, "Should return None if no version is current"


@requires_db
class TestHistoricalVersionLookup:
    """get_document_version_at_date returns the version applicable on a date."""

    def test_returns_v1_for_2024_date(self, db_session):
        from src.db.repositories.documents import get_document_version_at_date
        _, doc, v1, v2 = _setup_versioned_doc(db_session)

        # Query at mid-2024 → v1 is the only effective version
        version = get_document_version_at_date(db_session, doc.id, date(2024, 6, 1))
        assert version is not None
        assert version.id == v1.id
        assert version.version_label == "v1"

    def test_returns_v2_for_2026_date(self, db_session):
        from src.db.repositories.documents import get_document_version_at_date
        _, doc, v1, v2 = _setup_versioned_doc(db_session)

        version = get_document_version_at_date(db_session, doc.id, date(2026, 6, 1))
        assert version is not None
        assert version.id == v2.id
        assert version.version_label == "v2"

    def test_returns_none_before_any_version(self, db_session):
        from src.db.repositories.documents import get_document_version_at_date
        _, doc, v1, v2 = _setup_versioned_doc(db_session)

        # Query before v1 effective date → no applicable version
        version = get_document_version_at_date(db_session, doc.id, date(2023, 1, 1))
        assert version is None, (
            "Should return None when no version was effective on the query date"
        )

    def test_tie_broken_by_published_date(self, db_session):
        """When two versions share the same effective_date, published_date breaks the tie."""
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import (
            get_or_create_document, create_document_version, get_document_version_at_date
        )

        co  = get_or_create_company(db_session, "TieBreakCo")
        doc = get_or_create_document(db_session, co.id, "Tie Doc", "policy")

        # Two versions with same effective_date but different published_date
        va = create_document_version(db_session, doc.id, "va", "fa.pdf", "tba_cs",
                                      effective_date=date(2025, 1, 1),
                                      published_date=date(2025, 1, 15))
        vb = create_document_version(db_session, doc.id, "vb", "fb.pdf", "tbb_cs",
                                      effective_date=date(2025, 1, 1),
                                      published_date=date(2025, 3, 1))

        # Query at the shared effective date → should return the later published version
        version = get_document_version_at_date(db_session, doc.id, date(2025, 6, 1))
        assert version is not None
        assert version.id == vb.id, (
            "When effective_date is tied, the version with latest published_date should win"
        )

    def test_not_affected_by_is_current(self, db_session):
        """Temporal query should NOT use is_current — it uses effective_date."""
        from src.db.repositories.documents import get_document_version_at_date
        _, doc, v1, v2 = _setup_versioned_doc(db_session)

        # v1 is NOT current (is_current=False), but it should be returned for 2024 date
        v1_result = get_document_version_at_date(db_session, doc.id, date(2024, 6, 1))
        assert v1_result is not None
        assert v1_result.is_current is False, (
            "Temporal query should return v1 even though it's not is_current"
        )
