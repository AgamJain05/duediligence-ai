"""
vector_store.py  ─  FAISS index management
═══════════════════════════════════════════
Phase 1: Exact brute-force cosine similarity search.

SIMILARITY MATH
───────────────
We use  IndexFlatIP  (inner product / dot product).

Given that embed_texts() and embed_query() both return L2-normalized vectors:

    ||v|| = 1  for every vector v

    dot(v1, v2) = ||v1|| * ||v2|| * cos(θ) = cos(θ)

So: inner product == cosine similarity when vectors are unit-normalized.
Score range: [-1.0, +1.0]  where  +1.0 = identical direction (most similar).

Why not IndexFlatL2?  L2 distance is inversely related to cosine similarity
for normalized vectors, but the mapping is non-linear and scores are less
interpretable.  IndexFlatIP with normalized vectors gives us cosine scores
directly, which are intuitive ("0.85 similarity").

Pipeline position:
    [embeddings]  →  build_index()  →  [FAISS index]
    [FAISS index] →  save_index()   →  [data/index/faiss.index]  (disk)
    [disk]        →  load_index()   →  [FAISS index]             (memory)
    [FAISS index] →  search()       →  [(idx, score), ...]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_INDEX_DIR = Path("data/index")
INDEX_FILENAME    = "faiss.index"


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Receives : float32 numpy array of shape (N, dim), L2-normalized
    Returns  : a populated FAISS IndexFlatIP ready for search

    Why it exists: FAISS must ingest all vectors before it can search.
    We call this once during ingestion, not at query time.

    Phase 1 uses IndexFlatIP (exact search, no approximation).
    For small corpora (< ~100k vectors) this is perfectly fast.
    Approximate methods (HNSW, IVF) are a Phase 2+ optimization.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"embeddings must be a non-empty 2D array, got shape {embeddings.shape}"
        )

    dim = embeddings.shape[1]

    # IndexFlatIP: stores vectors as-is and computes exact inner products
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)   # vectors are copied into the FAISS index

    print(
        f"[vector_store] Index built: {index.ntotal} vectors, dim={dim}"
    )
    return index


def save_index(
    index: faiss.IndexFlatIP,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> None:
    """
    Receives : a populated FAISS index and directory path
    Returns  : nothing  (side-effect: writes faiss.index to disk)

    Why it exists: Embedding the whole corpus takes minutes.  We persist
    the index so app.py can load it in milliseconds on every startup
    without touching the embedding model at all for stored vectors.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / INDEX_FILENAME
    faiss.write_index(index, str(index_path))
    print(f"[vector_store] Index saved -> {index_path}")


def load_index(index_dir: str | Path = DEFAULT_INDEX_DIR) -> faiss.IndexFlatIP:
    """
    Receives : directory path containing a saved faiss.index file
    Returns  : a loaded FAISS index ready for search

    Why it exists: Restores the persisted index at query time.
    Raises a clear FileNotFoundError with instructions if index is missing,
    so the user knows to run ingest.py first.
    """
    index_path = Path(index_dir) / INDEX_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(
            f"\n[vector_store] FAISS index not found at: {index_path}\n"
            "  → Run ingestion first:  python scripts/ingest.py\n"
        )

    index = faiss.read_index(str(index_path))
    print(f"[vector_store] Index loaded <- {index_path} ({index.ntotal} vectors)")
    return index


def search(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Receives : FAISS index, query embedding of shape (1, dim), number of results
    Returns  : list of (chunk_index, cosine_score) tuples
               sorted by score descending (most similar first)
               chunk_index is the integer offset into the chunks list

    Why it exists: Core retrieval.  FAISS returns integer indices into the
    array that was originally passed to build_index().  The caller (retriever.py)
    maps those integers back to the actual chunk text and metadata.

    Score note: cosine similarity ∈ [-1, 1].  For well-matched documents
    in English you typically see scores between 0.5 and 0.95.
    """
    # Ensure shape is (1, dim) — FAISS always needs a 2D query matrix
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Don't ask for more results than exist in the index
    top_k = min(top_k, index.ntotal)

    if top_k == 0:
        return []

    # index.search returns:
    #   scores  shape (n_queries, top_k) — inner product scores (= cosine here)
    #   indices shape (n_queries, top_k) — integer offsets (-1 = padding)
    scores, indices = index.search(query_embedding, top_k)

    results: List[Tuple[int, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue   # FAISS pads with -1 when top_k > index.ntotal
        results.append((int(idx), float(score)))

    return results   # already sorted descending by FAISS
