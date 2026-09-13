"""
src/retrieval/keyword.py  -  BM25 lexical keyword retrieval
============================================================
Phase 3: Local in-memory BM25 index over chunk texts.

WHAT IS BM25?
--------------
BM25 (Best Match 25) is a ranking function that scores documents based on
the presence and frequency of query terms. It's an improvement over TF-IDF
that adds two key refinements:

  1. TERM SATURATION: extra occurrences of a term add diminishing returns.
     A document with "revenue" appearing 10x is not 10x better than one
     with "revenue" appearing once.

  2. DOCUMENT LENGTH NORMALIZATION: longer documents are penalized so they
     don't naturally score higher just because they contain more words.

THE BM25 FORMULA
-----------------
For query Q with terms q1, q2, ..., qn and document D:

  BM25(Q, D) = sum over qi of:
    IDF(qi) * [ f(qi, D) * (k1 + 1) ]
              /---------------------------------
              [ f(qi, D) + k1 * (1 - b + b * |D| / avgdl) ]

Where:
  f(qi, D)  = term frequency of qi in D
  |D|       = document length in tokens
  avgdl     = average document length across corpus
  k1        = 1.5 (term saturation constant; rank_bm25 default)
  b         = 0.75 (length normalization constant; rank_bm25 default)
  IDF(qi)   = log( (N - df + 0.5) / (df + 0.5) + 1 )
              where N = corpus size, df = number of docs containing qi

WHY BM25 SUCCEEDS WHERE SEMANTIC SEARCH FAILS
----------------------------------------------
Query: "FY2024 revenue 154"

BM25 rewards: documents literally containing "FY2024", "revenue", "154".
Semantic search rewards: documents semantically about "financial performance".

The chunk with the exact figure "Revenue (USD m) | 121 | 154 | 186" will
score higher in BM25 because "154" and "FY2024" are exact token matches.
Semantic search might rank a general financial discussion higher.

WHY BM25 FAILS WHERE SEMANTIC SEARCH SUCCEEDS
-----------------------------------------------
Query: "How exposed is the company to dependence on external manufacturers?"

BM25 searches for: "exposed", "dependence", "external", "manufacturers".
None of these words appear verbatim in the risk section, which says:
"The platform depends on third-party cloud infrastructure."

BM25 score will be near 0. Semantic search finds the match because
"depends on third-party" ~ "dependence on external".

TOKENIZATION
-------------
We use simple whitespace tokenization + lowercase. This is intentional:

  1. UNDERSTANDABLE: you can predict what tokens get matched
  2. SUFFICIENT: for this corpus, simple tokenization is adequate
  3. DEBUGGABLE: you can trace exactly why a document scored high/low

We do NOT use stemming or stopword removal in this implementation to keep
the behavior fully transparent and predictable for learning purposes.

INDEX STORAGE
--------------
The BM25 index is built in memory on construction from the chunk list.
It does NOT persist to disk. Re-construction takes < 10ms for small corpora.
This is intentional for Phase 3 (learning focus, not optimization).
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

from src.retrieval.models import RetrievalResult


def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25 indexing and query expansion.

    Strategy: lowercase, split on non-alphanumeric characters.
    This means "FY2024", "2024", and "fy2024" are different tokens —
    which is correct for exact-match retrieval.

    We intentionally do NOT remove stopwords so you can query for
    things like "what is" and get a predictable result.

    Examples:
        "Revenue (USD m) | 121 | 154" -> ["revenue", "usd", "m", "121", "154"]
        "FY2024 operating margin 14%" -> ["fy2024", "operating", "margin", "14"]
    """
    # Split on non-alphanumeric (keeps numbers, letters; drops punctuation)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return tokens


class KeywordRetriever:
    """
    BM25-based lexical retrieval over the Phase-2 chunk store.

    The index is built on construction from a list of chunk dicts.
    No disk persistence — fast rebuild on startup.

    Usage:
        retriever = KeywordRetriever(chunks)
        results = retriever.search("FY2024 operating margin", top_k=20)

    Each result has keyword_score (BM25, unnormalized) and keyword_rank set.
    """

    def __init__(self, chunks: List[Dict]) -> None:
        """
        Build the BM25 index from chunk texts.

        chunks: list of chunk dicts (from chunks_structure_aware.json)

        WHY BUILD AT CONSTRUCTION?
        BM25 needs corpus-level statistics (avgdl, IDF) which require
        seeing all documents. We compute these once at build time so
        each query is just a lookup + scoring pass.
        """
        from rank_bm25 import BM25Okapi   # lazy import, same pattern as embeddings.py

        self._chunks = chunks

        t0 = time.perf_counter()

        # Tokenize all chunk texts for BM25 indexing
        # Each element is a list of tokens for one chunk
        tokenized_corpus = [_tokenize(c.get("text", "")) for c in chunks]

        # BM25Okapi uses:
        #   k1=1.5 (term frequency saturation)
        #   b=0.75 (document length normalization)
        # These are the canonical defaults from the BM25 paper.
        self._bm25 = BM25Okapi(tokenized_corpus)

        self.build_latency_ms = (time.perf_counter() - t0) * 1000
        self.last_search_ms:  float = 0.0

        print(
            f"[KeywordRetriever] BM25 index built: {len(chunks)} chunks "
            f"in {self.build_latency_ms:.1f}ms"
        )

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[RetrievalResult]:
        """
        Receives : query string
                   top_k — candidate pool size
        Returns  : list[RetrievalResult] sorted by keyword_score descending
                   chunks with BM25 score > 0 only (no results if no term overlap)

        BM25 scoring is NOT comparable to cosine similarity scores.
        BM25 scores are unnormalized sums of IDF-weighted term frequencies.
        Typical range: 0 to ~20, depending on corpus and query length.

        Results with score=0 are filtered out — they mean zero token overlap
        with the query. Returning zero-score chunks would be noise.
        """
        if not query.strip():
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        t0 = time.perf_counter()

        # BM25.get_scores() returns a score for EVERY document in the corpus
        # Shape: (num_chunks,) float64 array
        scores = self._bm25.get_scores(query_tokens)

        self.last_search_ms = (time.perf_counter() - t0) * 1000

        # Sort by score descending, keep top_k, filter out zero scores
        # enumerate to keep the original chunk index
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        scored = [(i, s) for i, s in scored if s > 0.0]   # filter zero-overlap
        scored = scored[:top_k]

        results: List[RetrievalResult] = []
        for rank, (chunk_idx, score) in enumerate(scored, start=1):
            chunk = self._chunks[chunk_idx]
            results.append(RetrievalResult(
                chunk_id      = chunk.get("chunk_id", ""),
                text          = chunk.get("text", ""),
                source        = chunk.get("source_filename") or chunk.get("source", ""),
                page_start    = chunk.get("page_start") or chunk.get("page_num", 0),
                page_end      = chunk.get("page_end", chunk.get("page_start") or chunk.get("page_num", 0)),
                section       = chunk.get("section", ""),
                section_path  = chunk.get("section_path") or [],
                content_type  = chunk.get("content_type", "text"),
                metadata      = chunk,
                keyword_score = float(score),
                keyword_rank  = rank,
            ))

        return results

    def get_chunk_scores(self, query: str) -> Dict[str, float]:
        """
        Return a dict mapping chunk_id -> BM25 score for all chunks.
        Useful for inspecting which chunks scored > 0 and why.
        """
        if not query.strip():
            return {}
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        return {
            self._chunks[i].get("chunk_id", str(i)): float(s)
            for i, s in enumerate(scores)
        }

    def explain(self, query: str, chunk_id: str) -> Optional[str]:
        """
        Return a human-readable explanation of why a chunk scored as it did.
        Shows which query tokens matched and their approximate IDF contribution.
        """
        chunk = next((c for c in self._chunks if c.get("chunk_id") == chunk_id), None)
        if not chunk:
            return None

        query_tokens = _tokenize(query)
        chunk_tokens = set(_tokenize(chunk.get("text", "")))
        matched = [t for t in query_tokens if t in chunk_tokens]
        missed  = [t for t in query_tokens if t not in chunk_tokens]

        lines = [
            f"BM25 explanation for chunk {chunk_id}:",
            f"  Query tokens    : {query_tokens}",
            f"  Matched tokens  : {matched}",
            f"  Missed tokens   : {missed}",
            f"  Overlap         : {len(matched)}/{len(query_tokens)} tokens",
        ]
        return "\n".join(lines)
