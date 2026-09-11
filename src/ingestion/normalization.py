"""
src/ingestion/normalization.py  ─  Text normalization
═══════════════════════════════════════════════════════
Phase 2: Clean up raw PDF extraction artifacts without altering meaning.

WHY WE NEED NORMALIZATION
──────────────────────────
PDF text extraction is notoriously messy. Common problems:

1. EXCESSIVE BLANK LINES
   PDF to text converters often insert many blank lines between sections.
   "Financial Performance\\n\\n\\n\\n\\nRevenue grew..." → wastes tokens.

2. REPEATED WHITESPACE
   Words can be spaced with multiple spaces: "Revenue  grew  by  22%".
   This happens when PDF columns get merged or when kerning is wide.

3. BROKEN LINE WRAPPING
   Long sentences sometimes get split mid-word:
   "The company's sup-
   ply chain depends on..."
   → Should be: "The company's supply chain depends on..."

4. REPEATED HEADERS/FOOTERS
   PDFs often stamp the company name or page number on every page.
   These flood the chunks with redundant content:
   "OrionVault Systems — Confidential"  (appears on every page)

5. LIGATURE/ENCODING ARTIFACTS
   Some PDFs produce "ﬁnancial" (ﬁ is a ligature character) instead of
   "financial". We normalize common ligatures.

DESIGN PRINCIPLE
────────────────
DO NOT alter meaning. Every actual word from the source survives.
We only remove structural noise. The raw_text is always preserved
so the transformation can be inspected and audited.

WHAT WE DON'T DO
────────────────
- No stemming or lemmatization
- No stopword removal
- No spelling correction
- No synonym expansion
Those belong to later phases or not at all.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple


# ── Constants ─────────────────────────────────────────────────────────────────
# A line is considered a "repeated header/footer candidate" if it appears
# on at least this fraction of all pages.
HEADER_FOOTER_THRESHOLD = 0.6   # appears on ≥60% of pages → strip it

# Maximum consecutive blank lines to allow after collapsing
MAX_CONSECUTIVE_BLANKS = 1

# Ligature map: common PDF ligatures → plain ASCII equivalents
_LIGATURE_MAP = {
    "\ufb01": "fi",   # ﬁ  fi ligature
    "\ufb02": "fl",   # ﬂ  fl ligature
    "\ufb00": "ff",   # ﬀ  ff ligature
    "\ufb03": "ffi",  # ﬃ  ffi ligature
    "\ufb04": "ffl",  # ﬄ  ffl ligature
    "\u2019": "'",    # '  right single quote → apostrophe
    "\u2018": "'",    # '  left single quote
    "\u201c": '"',    # "  left double quote
    "\u201d": '"',    # "  right double quote
    "\u2013": "-",    # –  en dash
    "\u2014": "--",   # —  em dash (preserve as double hyphen)
    "\u00a0": " ",    # non-breaking space → regular space
}


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Receives : raw text string from PDF extraction
    Returns  : cleaned text string (wording unchanged)

    Applies all normalization steps in order:
      1. Fix ligatures and special characters
      2. Fix broken hyphenated line wraps
      3. Collapse repeated whitespace within lines
      4. Collapse excessive blank lines
    """
    if not text:
        return text

    text = _fix_ligatures(text)
    text = _fix_broken_hyphens(text)
    text = _collapse_whitespace_in_lines(text)
    text = _collapse_blank_lines(text)
    return text.strip()


def detect_repeated_lines(pages_raw_text: List[str]) -> set[str]:
    """
    Receives : list of raw page texts (one per page)
    Returns  : set of lines that appear repeatedly across pages
               (likely headers/footers — candidates for removal)

    Algorithm:
        1. For each page, collect the first 2 and last 2 non-empty lines.
        2. Count how many pages each line appears on.
        3. Lines appearing on ≥60% of pages are header/footer candidates.

    WHY ONLY FIRST/LAST 2 LINES?
    Headers and footers almost always appear at the top or bottom of
    a page. We avoid scanning the entire page to prevent accidentally
    flagging legitimate repeated content (e.g., section headings that
    appear on multiple pages).
    """
    if not pages_raw_text:
        return set()

    n_pages = len(pages_raw_text)
    line_page_count: Counter[str] = Counter()

    for page_text in pages_raw_text:
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        # Collect candidates: first 2 and last 2 non-empty lines
        candidates = []
        if len(lines) >= 2:
            candidates.extend(lines[:2])
            candidates.extend(lines[-2:])
        elif lines:
            candidates.extend(lines)

        # Deduplicate per page before counting
        for line in set(candidates):
            if len(line) > 3:   # ignore very short lines (page numbers, etc.)
                line_page_count[line] += 1

    threshold_count = n_pages * HEADER_FOOTER_THRESHOLD
    repeated = {
        line
        for line, count in line_page_count.items()
        if count >= threshold_count
    }
    return repeated


def strip_repeated_lines(text: str, repeated_lines: set[str]) -> str:
    """
    Receives : page text and set of lines to strip
    Returns  : text with repeated header/footer lines removed

    We match stripped versions of each line so minor spacing
    differences don't defeat the match.
    """
    if not repeated_lines:
        return text

    result_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped not in repeated_lines:
            result_lines.append(line)

    return "\n".join(result_lines)


def normalize_pages(
    pages_raw_text: List[str],
    strip_headers_footers: bool = True,
) -> List[Tuple[str, str]]:
    """
    Receives : list of raw page text strings
    Returns  : list of (raw_text, normalized_text) tuples

    This is the full normalization pipeline:
    1. Detect repeated lines across all pages (headers/footers)
    2. For each page: strip repeated lines, then apply normalize_text()

    Returning both raw and normalized preserves the original for inspection.
    """
    repeated: set[str] = set()
    if strip_headers_footers:
        repeated = detect_repeated_lines(pages_raw_text)
        if repeated:
            print(
                f"  [normalization] Detected {len(repeated)} repeated "
                f"header/footer line(s) — will strip from all pages."
            )

    result: List[Tuple[str, str]] = []
    for raw in pages_raw_text:
        stripped = strip_repeated_lines(raw, repeated) if repeated else raw
        normalized = normalize_text(stripped)
        result.append((raw, normalized))

    return result


# ── Private helpers ───────────────────────────────────────────────────────────

def _fix_ligatures(text: str) -> str:
    """Replace ligature characters with their ASCII equivalents."""
    for ligature, replacement in _LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)
    return text


def _fix_broken_hyphens(text: str) -> str:
    """
    Fix words broken across lines with a trailing hyphen.

    Pattern: "sup-\n  ply" → "supply"
    The regex matches a hyphen at end of line (possibly with trailing spaces),
    followed by optional whitespace at the start of the next line, then
    a continuation word.

    We only fix soft hyphens (hyphen + newline), NOT intentional hyphens
    inside compound words like "well-known".
    """
    # Pattern: word chars + hyphen at end of line + newline + optional indent
    # + lowercase continuation (lowercase indicates it's a continuation, not
    # a new capitalized word / proper noun)
    pattern = re.compile(r"(\w+)-\n\s*([a-z]\w*)")
    return pattern.sub(r"\1\2", text)


def _collapse_whitespace_in_lines(text: str) -> str:
    """
    Collapse multiple consecutive spaces within each line into one.
    Preserves newlines (line structure is still meaningful at this stage).
    """
    lines = text.split("\n")
    cleaned = [re.sub(r"[ \t]{2,}", " ", line) for line in lines]
    return "\n".join(cleaned)


def _collapse_blank_lines(text: str) -> str:
    """
    Collapse runs of more than MAX_CONSECUTIVE_BLANKS blank lines.
    This reduces wasted tokens without removing structural whitespace.
    """
    # Match 2 or more consecutive newline-only lines
    pattern = re.compile(r"\n{3,}")
    return pattern.sub("\n\n", text)
