"""
chunking.py  ─  Fixed-size text chunking with overlap
══════════════════════════════════════════════════════
Phase 1: Simple token-count-based chunking only.
- No semantic boundary detection
- No structure-aware splitting (headings, tables, etc.)
- Uses tiktoken for accurate token counting

Pipeline position:
    [page dicts]  →  chunk_documents()  →  [chunk dicts]  →  embeddings.py

WHY OVERLAP EXISTS
──────────────────
When we split a page into fixed-size windows, meaningful information can
straddle two adjacent chunks.  Example: a sentence about "supply-chain
concentration risk" might begin at the last 30 tokens of chunk N and end
at the first 30 tokens of chunk N+1.

Without overlap:
    Chunk N  → contains "...supply-chain concentration"
    Chunk N+1 → contains "risk is material because..."
    A query about "supply chain risk" might not retrieve either chunk
    with high confidence, because neither contains the complete thought.

With overlap (50 tokens repeated):
    Chunk N   → contains "...supply-chain concentration risk is material"
    Chunk N+1 → contains "supply-chain concentration risk is material because..."
    Now both chunks carry the full phrase, so at least one is likely to be
    retrieved for a relevant query.

Trade-off: overlap increases the total number of chunks (and therefore
the index size and embedding cost), but improves recall.  50 tokens on a
500-token chunk = 10% overhead — acceptable for Phase 1 for that.
"""

from __future__ import annotations

import hashlib
from typing import List, Dict

import tiktoken


# ── Defaults (can be overridden per call) ─────────────────────────────────────
CHUNK_SIZE    = 500   # target tokens per chunk
CHUNK_OVERLAP = 50    # tokens shared between consecutive chunks

# cl100k_base is the tokenizer used by GPT-3.5/GPT-4.  We use it here
# as a reasonable approximation for token counting.  BGE embeddings use
# a WordPiece tokenizer internally, but the token counts will be similar
# enough for our fixed-size chunking strategy.
TOKENIZER_ENCODING = "cl100k_base"

# ── Type alias ────────────────────────────────────────────────────────────────
ChunkDict = Dict[str, object]   # {chunk_id, source, page_num, text}


def _get_tokenizer() -> tiktoken.Encoding:
    """Return the tiktoken encoding.  Cached by tiktoken internally."""
    return tiktoken.get_encoding(TOKENIZER_ENCODING)


# Expose for tests without importing tiktoken directly
get_tokenizer = _get_tokenizer


def _make_chunk_id(source: str, page_num: int, token_start: int) -> str:
    """
    Receives : source filename, page number, token start position
    Returns  : 12-character hex string (deterministic, collision-resistant)

    Why deterministic? Re-running ingestion on the same PDFs produces the
    same chunk IDs, which makes debugging and logging reproducible.
    """
    raw = f"{source}::p{page_num}::s{token_start}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_page(
    page: Dict,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[ChunkDict]:
    """
    Receives : a single page dict  {source, page_num, text}
               chunk_size  — max tokens per chunk (default 500)
               overlap     — tokens repeated between chunks (default 50)
    Returns  : list of chunk dicts  [{chunk_id, source, page_num, text}, ...]

    Why it exists: LLMs and embedding models have input-length limits.
    Chunking converts long pages into bite-sized pieces that fit within
    those limits.  Smaller, focused chunks also improve retrieval precision
    because each chunk covers a narrower topic.

    Algorithm:
        tokens = tokenize(page.text)
        start  = 0
        while start < len(tokens):
            end = start + chunk_size
            yield decode(tokens[start:end])
            start += chunk_size - overlap   ← overlap step
    """
    enc = _get_tokenizer()
    tokens = enc.encode(page["text"])

    if not tokens:
        return []   # guard: empty page after stripping

    chunks: List[ChunkDict] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text   = enc.decode(chunk_tokens)

        chunk_id = _make_chunk_id(page["source"], page["page_num"], start)

        chunks.append(
            {
                "chunk_id": chunk_id,
                "source":   page["source"],
                "page_num": page["page_num"],
                "text":     chunk_text,
            }
        )

        if end >= len(tokens):
            break   # we've consumed the entire page

        # Advance by (chunk_size - overlap) so the NEXT chunk
        # re-reads the last `overlap` tokens of THIS chunk.
        start += chunk_size - overlap

    return chunks


def chunk_documents(
    pages: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[ChunkDict]:
    """
    Receives : list of page dicts (output of ingestion.py)
               chunk_size, overlap — forwarded to chunk_page()
    Returns  : flat list of all chunk dicts across all pages and documents

    Why it exists: Applies chunking to the entire corpus in one call.
    The calling code (scripts/ingest.py) doesn't need to loop manually.
    """
    all_chunks: List[ChunkDict] = []

    for page in pages:
        text = str(page.get("text", "")).strip()
        if not text:
            continue   # skip pages that produced no text

        page_chunks = chunk_page(page, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(page_chunks)

    return all_chunks
