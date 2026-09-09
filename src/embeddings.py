"""
embeddings.py  ─  Text embedding generation
════════════════════════════════════════════
Phase 1: Local sentence-transformer embeddings — 100% free, no API key needed.

Model: BAAI/bge-base-en-v1.5
  • 768-dimensional dense embeddings
  • Trained specifically for retrieval (asymmetric query ↔ passage search)
  • Competitive on MTEB English retrieval benchmarks
  • Runs on CPU or GPU; ~438 MB download on first use
  • Override via EMBEDDING_MODEL env var

Pipeline position:
    [chunk texts]  →  embed_texts()  →  [numpy float32 matrix]  →  vector_store.py
    [query string] →  embed_query()  →  [numpy float32 vector]  →  retriever.py

SIMILARITY CONVENTION
─────────────────────
Both embed_texts() and embed_query() return L2-NORMALIZED vectors.
This means:
    inner_product(v1, v2)  ==  cosine_similarity(v1, v2)

FAISS IndexFlatIP (inner product) therefore gives us cosine similarity
without any extra computation.  Score range: [-1.0, +1.0].
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

# NOTE: SentenceTransformer is imported lazily inside _get_model() so that
# unit tests which mock embed_query() can import this module without
# sentence-transformers being installed.  The real import happens on first use.

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# BGE models recommend a fixed instruction prefix when encoding QUERIES
# (not passages). This prefix helps the model distinguish search intent
# from a passage that happens to be phrased as a question.
# Reference: https://huggingface.co/BAAI/bge-base-en-v1.5
BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

# ── Module-level model cache ───────────────────────────────────────────────────
# We load the model once per Python process.  Loading ~438 MB from disk
# and initializing weights takes several seconds — we never want to do
# that twice in the same run.
_model = None


def _get_model():
    """
    Receives : nothing  (reads EMBEDDING_MODEL env var or uses default)
    Returns  : a loaded and cached SentenceTransformer model

    Why the cache matters: ingestion embeds thousands of chunks in one batch;
    the query loop calls embed_query() for every user question.  Both share
    this single loaded model.

    Lazy import: SentenceTransformer is only imported here (not at module top)
    so test files that mock embed_query() can import this module without
    requiring sentence-transformers to be installed.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # lazy import
        model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        print(f"[embeddings] Loading model '{model_name}' (first use)...")
        _model = SentenceTransformer(model_name)
        # get_embedding_dimension() is the new name in sentence-transformers 6.x
        # (replaces get_sentence_embedding_dimension() which is now deprecated)
        try:
            dim = _model.get_embedding_dimension()
        except AttributeError:
            dim = _model.get_sentence_embedding_dimension()  # fallback for older versions
        print(f"[embeddings] Model ready - embedding dimension: {dim}")
    return _model


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Receives : list of text strings (chunk texts from chunking.py)
               batch_size — how many texts to encode at once (memory trade-off)
    Returns  : float32 numpy array of shape  (N, embedding_dim)
               where every row is an L2-normalized embedding vector

    Why it exists: Converts raw text into a numerical representation that
    captures semantic meaning.  These vectors are stored in FAISS so that
    at query time we can find the most semantically similar chunks.

    Why L2-normalize? Because FAISS IndexFlatIP computes inner products.
    For unit-norm vectors: dot(a, b) == cosine_similarity(a, b).
    Normalizing here means we never have to handle separate normalization
    logic in the vector store.
    """
    model = _get_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=(len(texts) > 16),
        normalize_embeddings=True,   # ← L2 normalize → dot == cosine
        convert_to_numpy=True,
    )

    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Receives : a single query string from the user
    Returns  : float32 numpy array of shape  (1, embedding_dim)
               L2-normalized, ready for FAISS inner-product search

    Why it exists: At query time we need a vector that represents the
    user's question in the same embedding space as the stored chunk vectors.
    The slight asymmetry (query instruction prefix for BGE) can improve
    retrieval quality, especially for short questions.

    Why shape (1, dim) not (dim,)?
    FAISS index.search() expects a 2D matrix.  Returning (1, dim) means
    the caller never needs to reshape — just pass it straight to search().
    """
    model = _get_model()

    # BGE query instruction — improves retrieval for short questions.
    # The passages (chunks) are encoded WITHOUT this prefix.
    prefixed_query = BGE_QUERY_INSTRUCTION + query

    embedding = model.encode(
        [prefixed_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embedding.astype(np.float32)   # shape: (1, dim)


def get_embedding_dim() -> int:
    """Return the dimension of the current embedding model.  Used in tests."""
    model = _get_model()
    try:
        return model.get_embedding_dimension()
    except AttributeError:
        return model.get_sentence_embedding_dimension()
