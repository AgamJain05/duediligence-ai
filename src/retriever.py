"""
retriever.py  ─  Query-time retrieval
═══════════════════════════════════════
Orchestrates the "R" in RAG:
    query string
      → embed_query()          [embeddings.py]
      → FAISS search()         [vector_store.py]
      → map indices → chunks   [chunks.json]
      → return rich result dicts

Phase 1: Dense vector retrieval only.
No hybrid search, no BM25, no reranking, no metadata filtering.

Pipeline position:
    [query]  →  retrieve()  →  [list of result dicts]  →  generator.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import faiss

from src.embeddings import embed_query
from src.vector_store import load_index, search

# ── Paths (must match what ingest.py writes) ──────────────────────────────────
DEFAULT_INDEX_DIR = Path("data/index")
CHUNKS_FILENAME   = "chunks.json"

# ── Type alias ────────────────────────────────────────────────────────────────
# A "result" is a chunk dict augmented with a similarity score.
ResultDict = Dict[str, object]   # {chunk_id, source, page_num, text, score}


def load_chunks(index_dir: str | Path = DEFAULT_INDEX_DIR) -> List[Dict]:
    """
    Receives : directory containing chunks.json (written by ingest.py)
    Returns  : list of chunk dicts in the SAME ORDER as the FAISS index

    Why order matters: FAISS returns integer indices (0, 1, 2, …) that
    are offsets into the array that was passed to build_index().  chunks.json
    preserves that same ordering so chunks[i] is always the chunk whose
    embedding is at FAISS index position i.

    Raises FileNotFoundError with clear instructions if the file is missing.
    """
    chunks_path = Path(index_dir) / CHUNKS_FILENAME

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"\n[retriever] Chunk store not found at: {chunks_path}\n"
            "  → Run ingestion first:  python scripts/ingest.py\n"
        )

    with open(chunks_path, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)

    print(f"[retriever] Loaded {len(chunks)} chunks <- {chunks_path}")
    return chunks


def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[Dict],
    top_k: int = 5,
) -> List[ResultDict]:
    """
    Receives : query string  (raw user question, no preprocessing)
               index         (loaded FAISS index from vector_store.py)
               chunks        (list from load_chunks(), same ordering as index)
               top_k         (how many chunks to retrieve)
    Returns  : list of result dicts sorted by similarity score descending:
               [{chunk_id, source, page_num, text, score}, ...]

    Why it exists: This IS the retrieval step of RAG.
    Given a question, find the most semantically similar document chunks.
    These chunks are the "context" injected into the LLM prompt.

    Phase 1 flow (traceable, no magic):
        1. embed_query(query)          → (1, 768) float32 vector
        2. search(index, vector, top_k)→ [(faiss_idx, score), ...]
        3. chunks[faiss_idx]           → {chunk_id, source, page_num, text}
        4. attach score                → result dict
    """
    # Step 1: Embed the question
    query_vec = embed_query(query)   # shape (1, 768)

    # Step 2: Search the FAISS index for nearest neighbors
    raw_results = search(index, query_vec, top_k=top_k)

    if not raw_results:
        print("[retriever] WARNING: FAISS search returned no results.")
        return []

    # Step 3 & 4: Map integer indices → chunk metadata, attach score
    results: List[ResultDict] = []
    for faiss_idx, score in raw_results:
        if faiss_idx >= len(chunks):
            # Safety: mismatch between index and chunk store
            print(
                f"[retriever] WARNING: FAISS returned index {faiss_idx} "
                f"but chunk store only has {len(chunks)} entries. Skipping."
            )
            continue

        chunk = chunks[faiss_idx]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source":   chunk["source"],
                "page_num": chunk["page_num"],
                "text":     chunk["text"],
                "score":    score,
            }
        )

    return results
