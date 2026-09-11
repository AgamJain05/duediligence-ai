"""
src/ingestion/__init__.py
Phase 2 ingestion package.

Exposes the top-level entry points so callers don't need to know which
sub-module provides each function.

Usage:
    from src.ingestion import parse_pdf, build_structured_chunks, validate_chunks
"""

from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunking import build_structured_chunks
from src.ingestion.validation import validate_chunks

__all__ = ["parse_pdf", "build_structured_chunks", "validate_chunks"]
