"""
src/ingestion/structure.py  ─  Heuristic structure detection
═════════════════════════════════════════════════════════════
Phase 2: Detect headings, paragraphs, tables, and lists from text.

DESIGN PRINCIPLE: DETERMINISTIC HEURISTICS
────────────────────────────────────────────
We use hand-crafted rules — NOT machine learning. Why?

1. EXPLAINABILITY: Every decision has a clear reason you can trace.
   "This is a heading because it's < 80 chars, Title Case, no period."
   There is no black box.

2. DEBUGGABILITY: If a section is mislabeled, you can read the code
   and understand why, then fix the heuristic.

3. SUFFICIENCY: For a due-diligence PDF with a predictable structure,
   simple rules work well enough. A classifier would need training data
   we don't have.

4. LEARNING: The point of Phase 2 is to understand what structure IS,
   not to maximize F1 score on a benchmark.

CONFIDENCE SCORES
─────────────────
Every block gets a confidence score between 0.0 and 1.0.
High confidence (≥ 0.85): strong signal (e.g., pdfplumber font size info)
Medium confidence (0.5–0.85): text heuristics (Title Case, short line)
Low confidence (< 0.5): weak signals only → block_type = "unknown"

HEADING HEURISTICS (in order of strength)
──────────────────────────────────────────
1. Font size clearly larger than body → confidence 0.95
2. ALL CAPS, standalone line, < 80 chars → confidence 0.90
3. Title Case, < 60 chars, no terminal period → confidence 0.80
4. Numbered pattern (1., 2.1, A., I.) → confidence 0.75
5. Known section keyword at start → confidence 0.70
6. Short standalone line (< 40 chars) → confidence 0.55

TABLE HEURISTICS
────────────────
1. pdfplumber native table detection → confidence 0.95
2. Line contains ≥2 '|' characters → confidence 0.85
3. Line is tab-separated with ≥2 tabs → confidence 0.80
4. Multiple lines follow the pipe pattern → confidence 0.90

LIST HEURISTICS
───────────────
1. Line starts with bullet (•, -, *, –) → confidence 0.90
2. Line starts with numbered list item (1., 2., a., b.) → confidence 0.85
3. Multiple adjacent lines follow same pattern → higher confidence

PARAGRAPH DEFAULT
─────────────────
Any block that doesn't meet a stronger signal is a paragraph.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.ingestion.models import (
    TextBlock,
    BLOCK_TYPE_HEADING,
    BLOCK_TYPE_PARAGRAPH,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_LIST,
    BLOCK_TYPE_UNKNOWN,
)


# ── Heading detection parameters ──────────────────────────────────────────────
MAX_HEADING_LENGTH   = 100   # lines longer than this are unlikely to be headings
MIN_HEADING_LENGTH   = 2     # single-character lines are likely noise

# Known section keywords found in due-diligence documents
# These are strong signals that a short line is a heading
KNOWN_SECTION_KEYWORDS = {
    "executive summary", "company overview", "business model",
    "products", "product overview", "technology", "platform",
    "financial", "financial performance", "revenue", "profitability",
    "margins", "operating", "customers", "customer", "geography",
    "geographic", "market", "strategy", "strategic", "growth",
    "risks", "risk factors", "risk", "management", "team",
    "leadership", "history", "timeline", "milestones", "appendix",
    "supply chain", "competition", "competitive", "outlook",
    "conclusion", "summary", "recommendations", "due diligence",
    "legal", "regulatory", "compliance", "governance", "board",
    "shareholders", "ownership", "capital", "funding", "valuation",
    "ebitda", "arr", "mrr", "churn", "nrr", "pipeline",
    "healthcare", "financial services", "enterprise",
    "sentinel platform", "sentinel copilot", "orionvault",
}

# Numbered heading pattern: "1.", "2.1", "A.", "I.", "1.2.3"
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:\d+\.(?:\d+\.)*\d*|[A-Z]\.|[IVX]+\.)\s+\S"
)

# Bullet list pattern
_BULLET_RE = re.compile(r"^[\u2022\u2013\u2014\-\*]\s+\S")   # •, –, —, -, *

# Numbered list item pattern: "1.", "2.", "a.", "(1)", "(a)"
_LIST_ITEM_RE = re.compile(r"^(?:\d+\.|[a-z]\.|[A-Z]\.|\(\d+\)|\([a-z]\))\s+\S")

# Table pipe pattern: at least 2 pipe characters on the line
_PIPE_TABLE_RE = re.compile(r"\|.*\|")   # has at least two | chars

# Tab-separated table pattern: at least 2 tabs
_TAB_TABLE_RE = re.compile(r"\t.*\t")


# ── Public API ────────────────────────────────────────────────────────────────

def detect_blocks(
    text: str,
    page_number: int,
    font_size_info: Optional[List[Tuple[str, float]]] = None,
) -> List[TextBlock]:
    """
    Receives : normalized page text
               page_number (1-indexed)
               font_size_info — optional list of (text_fragment, font_size) pairs
                               from pdfplumber character data
    Returns  : list of TextBlock with detected types

    Algorithm:
        1. Split text into logical segments (double-newline = new segment)
        2. For each segment, classify using heuristics
        3. Return ordered list of TextBlock objects

    WHY SPLIT ON DOUBLE NEWLINES?
    Single newlines inside a PDF paragraph are often just line-wrap
    artifacts. A double (or more) newline separates genuinely different
    content blocks — headings, paragraphs, tables.
    """
    if not text.strip():
        return []

    # Compute average font size if info is available (for heading detection)
    avg_font_size: Optional[float] = None
    if font_size_info:
        sizes = [size for _, size in font_size_info if size > 0]
        if sizes:
            avg_font_size = sum(sizes) / len(sizes)

    # Split on blank lines to get candidate blocks
    raw_segments = re.split(r"\n{2,}", text.strip())
    # Also preserve table blocks detected via pipes (may be within a segment)
    blocks: List[TextBlock] = []

    for segment in raw_segments:
        segment = segment.strip()
        if not segment:
            continue

        sub_blocks = _classify_segment(
            segment, page_number, font_size_info, avg_font_size
        )
        blocks.extend(sub_blocks)

    return blocks


def classify_line(
    line: str,
    avg_font_size: Optional[float] = None,
    line_font_size: Optional[float] = None,
) -> Tuple[str, float]:
    """
    Receives : a single line of text and optional font size info
    Returns  : (block_type, confidence) tuple

    This is exposed for testing individual lines without needing a full page.
    """
    line = line.strip()
    if not line:
        return BLOCK_TYPE_UNKNOWN, 0.0

    # ── Table detection ───────────────────────────────────────────────────────
    if _PIPE_TABLE_RE.search(line) and line.count("|") >= 2:
        return BLOCK_TYPE_TABLE, 0.85
    if _TAB_TABLE_RE.search(line):
        return BLOCK_TYPE_TABLE, 0.80

    # ── List detection ────────────────────────────────────────────────────────
    if _BULLET_RE.match(line):
        return BLOCK_TYPE_LIST, 0.90
    if _LIST_ITEM_RE.match(line):
        return BLOCK_TYPE_LIST, 0.85

    # ── Heading detection ─────────────────────────────────────────────────────
    # Font size signal (strongest when available)
    if avg_font_size and line_font_size and line_font_size > avg_font_size * 1.15:
        if len(line) <= MAX_HEADING_LENGTH:
            return BLOCK_TYPE_HEADING, 0.95

    # Short line checks
    if len(line) > MAX_HEADING_LENGTH or len(line) < MIN_HEADING_LENGTH:
        return BLOCK_TYPE_PARAGRAPH, 0.70

    line_lower = line.lower()

    # ALL CAPS (but not all numbers/punctuation)
    has_alpha = any(c.isalpha() for c in line)
    if has_alpha and line.upper() == line and len(line) <= 80:
        return BLOCK_TYPE_HEADING, 0.90

    # Known section keyword match
    if line_lower in KNOWN_SECTION_KEYWORDS:
        return BLOCK_TYPE_HEADING, 0.88
    # Starts with a known keyword (but NOT if it ends with a period — that's a sentence)
    for keyword in KNOWN_SECTION_KEYWORDS:
        if (line_lower.startswith(keyword) and len(line) <= 80
                and not line.rstrip().endswith(".")):
            return BLOCK_TYPE_HEADING, 0.80

    # Numbered heading: "1.", "2.1", "A.", "I."
    if _NUMBERED_HEADING_RE.match(line):
        return BLOCK_TYPE_HEADING, 0.75

    # Title Case: each major word is capitalized, no trailing period
    words = line.split()
    if len(words) >= 2:
        is_title_case = all(
            w[0].isupper() or w.lower() in {"and", "or", "of", "the", "a", "in", "for", "to", "at", "by"}
            for w in words
            if w and w[0].isalpha()
        )
        no_period = not line.rstrip().endswith(".")
        if is_title_case and no_period and len(line) <= 60:
            return BLOCK_TYPE_HEADING, 0.75

    # Short standalone line with no period — weak heading signal
    if len(line) <= 40 and not line.rstrip().endswith(".") and not line.rstrip().endswith(","):
        return BLOCK_TYPE_HEADING, 0.55

    return BLOCK_TYPE_PARAGRAPH, 0.80


# ── Private helpers ───────────────────────────────────────────────────────────

def _classify_segment(
    segment: str,
    page_number: int,
    font_size_info: Optional[List[Tuple[str, float]]],
    avg_font_size: Optional[float],
) -> List[TextBlock]:
    """
    Classify a single text segment (between double newlines).

    A segment may contain:
    - A single heading line
    - Multiple paragraph lines
    - A table block (multiple pipe-separated lines)
    - A list (multiple bullet/numbered lines)
    - Mixed content → split further
    """
    lines = segment.split("\n")
    lines = [ln for ln in lines if ln.strip()]   # remove empty lines within segment

    if not lines:
        return []

    # ── Attempt table detection at segment level ───────────────────────────────
    # If multiple lines have pipe characters, this whole segment is a table
    pipe_lines = [ln for ln in lines if "|" in ln and ln.count("|") >= 2]
    if len(pipe_lines) >= 2:
        return [TextBlock(
            block_type=BLOCK_TYPE_TABLE,
            text=segment,
            raw_text=segment,
            page_number=page_number,
            confidence=0.88,
        )]

    # ── Attempt list detection at segment level ────────────────────────────────
    bullet_lines = [ln for ln in lines if _BULLET_RE.match(ln.strip()) or _LIST_ITEM_RE.match(ln.strip())]
    if len(bullet_lines) >= 2 or (len(bullet_lines) == 1 and len(lines) == 1):
        # Decide confidence based on how consistently list-like the segment is
        list_ratio = len(bullet_lines) / len(lines)
        confidence = 0.90 if list_ratio >= 0.8 else 0.70
        return [TextBlock(
            block_type=BLOCK_TYPE_LIST,
            text=segment,
            raw_text=segment,
            page_number=page_number,
            confidence=confidence,
        )]

    # ── Single line: classify directly ────────────────────────────────────────
    if len(lines) == 1:
        line = lines[0].strip()
        block_type, confidence = classify_line(line, avg_font_size)
        return [TextBlock(
            block_type=block_type,
            text=line,
            raw_text=line,
            page_number=page_number,
            confidence=confidence,
        )]

    # ── Multi-line segment: check if first line is a heading ──────────────────
    first_line = lines[0].strip()
    first_type, first_conf = classify_line(first_line, avg_font_size)

    blocks: List[TextBlock] = []

    if first_type == BLOCK_TYPE_HEADING and first_conf >= 0.65:
        # Emit the heading as its own block
        blocks.append(TextBlock(
            block_type=BLOCK_TYPE_HEADING,
            text=first_line,
            raw_text=first_line,
            page_number=page_number,
            confidence=first_conf,
        ))
        # Remaining lines become a paragraph
        rest = "\n".join(lines[1:]).strip()
        if rest:
            blocks.append(TextBlock(
                block_type=BLOCK_TYPE_PARAGRAPH,
                text=rest,
                raw_text=rest,
                page_number=page_number,
                confidence=0.80,
            ))
    else:
        # Whole segment is a paragraph
        blocks.append(TextBlock(
            block_type=BLOCK_TYPE_PARAGRAPH,
            text=segment,
            raw_text=segment,
            page_number=page_number,
            confidence=0.80,
        ))

    return blocks


def build_section_hierarchy(blocks: List[TextBlock]) -> List[Tuple[TextBlock, List[str]]]:
    """
    Receives : flat list of TextBlock objects (from all pages, in order)
    Returns  : list of (block, section_path) pairs

    Builds a running section path as we traverse blocks.
    Each heading updates the path; all subsequent blocks inherit it.

    Example output:
        (heading "Risk Factors",   ["Risk Factors"])
        (heading "Supply Chain",   ["Risk Factors", "Supply Chain"])
        (paragraph "...",          ["Risk Factors", "Supply Chain"])
        (heading "Customer Conc.", ["Risk Factors", "Customer Concentration"])

    WHY A STACK?
    We maintain a stack of headings. When we see a new heading that seems
    to be at the same or higher level as the current one, we pop and push.
    Since we don't have reliable level information from the text alone,
    we use a simple heuristic: shorter headings are usually higher-level.
    This is approximate but good enough for Phase 2.
    """
    result: List[Tuple[TextBlock, List[str]]] = []
    section_stack: List[str] = []   # stack of heading texts

    for block in blocks:
        if block.block_type == BLOCK_TYPE_HEADING and block.confidence >= 0.55:
            heading_text = block.text.strip()

            # Heuristic level estimation:
            # ALL CAPS → likely top-level (level 1)
            # Short Title Case → level 2
            # Longer → level 3
            estimated_level = _estimate_heading_level(heading_text)

            # Pop stack to match the estimated level
            # (level 1 = position 0, level 2 = position 1, etc.)
            while len(section_stack) >= estimated_level:
                section_stack.pop()

            section_stack.append(heading_text)

        current_path = list(section_stack)
        result.append((block, current_path))

    return result


def _estimate_heading_level(heading: str) -> int:
    """
    Estimate the heading level (1 = top-level) based on text signals.
    Returns 1, 2, or 3.

    Signals:
    - ALL CAPS and/or very short (≤ 30 chars) → level 1
    - Title Case, 30–70 chars → level 2
    - Longer or starts with number → level 3
    """
    heading = heading.strip()

    # ALL CAPS → very likely top-level
    if heading.upper() == heading and any(c.isalpha() for c in heading):
        return 1

    # Known top-level keywords
    heading_lower = heading.lower()
    top_level_keys = {
        "executive summary", "company overview", "products", "financial",
        "financial performance", "customers", "geography", "strategy",
        "risks", "risk factors", "management", "history", "technology",
        "appendix", "supply chain", "competition", "outlook",
    }
    if heading_lower in top_level_keys:
        return 1

    # Short line (likely a sub-section title)
    if len(heading) <= 35:
        return 2

    # Numbered pattern suggests sub-heading
    if _NUMBERED_HEADING_RE.match(heading):
        return 3

    return 2
