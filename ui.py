"""
ui.py - DueDiligenceAI Streamlit interface

Launch:
    streamlit run ui.py

Theme lives in .streamlit/config.toml (not in CSS) so native widgets,
the sidebar and custom elements all stay consistent.
All retrieval/generation logic stays in src/ui_backend.py.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_backend import (
    VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED,
    POSTGRES_HNSW, POSTGRES_FILTERED, POSTGRES_FTS, POSTGRES_EXACT, POSTGRES_HYBRID,
    MODE_LABELS,
    check_faiss_index, check_database, check_llm,
    get_available_modes, build_faiss_retriever, run_query,
)

# ── Fill these in. Anything left empty is simply not shown. ──────────────────
GITHUB_URL = ""           # e.g. "https://github.com/<you>/DueDiligenceAI"
CORPUS_DESCRIPTION = ""   # e.g. "Synthetic annual reviews and risk dossiers for <Company>, 2024-2026."
EVAL_SUMMARY = ""         # Only real, reproducible numbers. e.g. "Recall@5 on 40 hand-labelled questions: X"

EXAMPLES = [
    "What were the company's financial highlights?",
    "What are its major business and operational risks?",
    "Compare management's statements with the formal risk disclosures.",
    "What is the current data residency requirement?",
    "What changed between the 2024 and 2026 risk documents?",
    "What evidence supports the company's growth strategy?",
]

PG_MODES = {POSTGRES_HNSW, POSTGRES_FILTERED, POSTGRES_FTS, POSTGRES_EXACT, POSTGRES_HYBRID}
PG_ONLY_MODES = {POSTGRES_HNSW, POSTGRES_FILTERED, POSTGRES_FTS, POSTGRES_EXACT}

st.set_page_config(
    page_title="DueDiligenceAI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-agnostic CSS: neutral translucent greys so it holds up in light or dark.
st.markdown(
    """
<style>
.block-container { padding-top: 2.2rem; max-width: 1180px; }
footer { visibility: hidden; }
.dd-title { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; line-height: 1.2; }
.dd-sub   { opacity: .72; margin: .35rem 0 .6rem 0; font-size: 1rem; }
.passage {
    white-space: pre-wrap; word-wrap: break-word;
    border-left: 3px solid rgba(79,143,247,.8);
    background: rgba(128,128,128,.09);
    padding: .8rem 1rem; border-radius: 4px;
    font-size: .92rem; line-height: 1.65;
}
.pill {
    display: inline-block; padding: 1px 9px; margin: 0 6px 6px 0;
    border: 1px solid rgba(128,128,128,.45); border-radius: 999px;
    font-size: .74rem; font-weight: 600; opacity: .9;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state and callbacks ──────────────────────────────────────────────
# Widget state must be changed through session_state[key] (ideally in a callback).
# Passing value= to a keyed text_area is ignored after the first render, which is
# why examples/history/clear did not update the box in the original version.
def _init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("current_result", None)
    st.session_state.setdefault("current_query", "")
    st.session_state.setdefault("query_text", "")


def _use_example(q: str) -> None:
    st.session_state.query_text = q


def _clear() -> None:
    st.session_state.query_text = ""
    st.session_state.current_result = None
    st.session_state.current_query = ""


def _load_history(i: int) -> None:
    entry = st.session_state.history[i]
    st.session_state.current_result = entry["result"]
    st.session_state.current_query = entry["query"]
    st.session_state.query_text = entry["query"]


_init_state()


# ── Backend checks ───────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _backend_status():
    faiss_ok, faiss_msg = check_faiss_index()
    db_ok, db_msg = check_database()
    llm_ok, llm_msg = check_llm()
    return faiss_ok, faiss_msg, db_ok, db_msg, llm_ok, llm_msg


faiss_ok, faiss_msg, db_ok, db_msg, llm_ok, llm_msg = _backend_status()


@st.cache_resource(show_spinner="Loading retrieval index...")
def _get_faiss_retriever():
    if not faiss_ok:
        return None, None
    try:
        return build_faiss_retriever(mode=HYBRID_RERANKED), None
    except Exception as exc:  # surfaced below, outside the cache
        return None, str(exc)


faiss_retriever, faiss_load_error = _get_faiss_retriever()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _dot(ok: bool, label: str, optional: bool = False) -> str:
    color = "green" if ok else ("orange" if optional else "red")
    return f":{color}[●] {label}"


def _safe_md(text: str) -> str:
    """Escape $ so dollar amounts are not parsed as LaTeX (breaks financial answers)."""
    return (text or "").replace("$", r"\$")


def _fmt(value, spec: str) -> str | None:
    return format(value, spec) if value is not None else None


def _export_markdown(query: str, result: dict) -> str:
    lines = [f"# Research brief\n", f"**Question:** {query}\n",
             f"**Retrieval mode:** {MODE_LABELS.get(result['mode'], result['mode'])}\n",
             "## Answer\n", result.get("answer") or "_No answer generated._", "\n## Sources\n"]
    for i, c in enumerate(result.get("retrieved_chunks", []), start=1):
        doc = c.get("document") or c.get("source", "Unknown")
        lines.append(f"{i}. {doc}, p.{c.get('page_num', '?')}"
                     + (f", {c['section']}" if c.get("section") else ""))
    return "\n".join(lines)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="dd-title">DueDiligenceAI</p>'
    '<p class="dd-sub">Company research with answers you can trace back to the source passage.</p>',
    unsafe_allow_html=True,
)
st.caption("   ·   ".join([
    _dot(faiss_ok, "Vector index"),
    _dot(db_ok, "PostgreSQL", optional=True),
    _dot(llm_ok, "Language model"),
]))

if not llm_ok:
    st.warning(f"Language model unavailable. {llm_msg}")
if faiss_load_error:
    st.error(f"The retrieval index failed to load: {faiss_load_error}")
if not faiss_ok and not db_ok:
    st.error(f"No knowledge base is available. {faiss_msg}")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Research settings")

    available_modes = get_available_modes(faiss_ok, db_ok)
    if not available_modes:
        st.error("No retrieval backend available.")
        available_modes = [HYBRID_RERANKED]

    mode_display = [MODE_LABELS.get(m, m) for m in available_modes]
    default_idx = available_modes.index(HYBRID_RERANKED) if HYBRID_RERANKED in available_modes else 0

    selected_label = st.selectbox(
        "Retrieval mode",
        options=mode_display,
        index=default_idx,
        help=(
            "**Vector**: dense semantic search (FAISS, cosine similarity).\n\n"
            "**Keyword**: BM25 exact-term matching.\n\n"
            "**Hybrid**: Reciprocal Rank Fusion of both.\n\n"
            "**Hybrid + reranking**: a cross-encoder re-scores the fused candidates.\n\n"
            "**PostgreSQL modes**: require DATABASE_URL."
        ),
    )
    selected_mode = available_modes[mode_display.index(selected_label)]

    top_k = st.slider(
        "Passages sent to the model", min_value=1, max_value=20, value=5, step=1,
        help="Top-K chunks passed to the LLM as context.",
    )

    with st.expander("Filters"):
        content_type = st.selectbox(
            "Content type", ["All", "text", "table", "list", "mixed"],
            help="Use 'table' for financial figures.",
        )
        section_filter = st.text_input(
            "Section contains", placeholder="Risk, Financial, Strategy",
            help="Case-insensitive match against section headings.",
        )
        if selected_mode in {POSTGRES_FILTERED, POSTGRES_HNSW, POSTGRES_HYBRID}:
            version_filter = st.selectbox(
                "Document version", ["All versions", "Current only", "Historical only"],
            )
            doc_type_filter = st.text_input(
                "Document type", placeholder="annual_review, dossier",
                help="Exact match on the document_type column.",
            )
        else:
            version_filter, doc_type_filter = "All versions", ""

    show_diagnostics = st.checkbox(
        "Show retrieval diagnostics", value=False,
        help="Per-chunk scores, candidate counts and latency by stage.",
    )

    st.divider()
    st.subheader("Recent questions")
    if not st.session_state.history:
        st.caption("Nothing yet this session.")
    else:
        recent = list(enumerate(st.session_state.history))[::-1][:8]
        for i, entry in recent:
            label = entry["query"][:44] + ("..." if len(entry["query"]) > 44 else "")
            st.button(label, key=f"hist_{i}", use_container_width=True,
                      on_click=_load_history, args=(i,), help=entry["query"])

    st.divider()
    st.caption(f"Model: `{os.getenv('OPENROUTER_MODEL', 'google/gemma-4-31b-it')}`")
    if GITHUB_URL:
        st.markdown(f"[Source code]({GITHUB_URL})")


# ── Filters dict ─────────────────────────────────────────────────────────────
active_filters: dict = {}
if content_type != "All":
    active_filters["content_type"] = content_type
if section_filter.strip():
    active_filters["section"] = section_filter.strip()
if version_filter == "Current only":
    active_filters["is_current"] = True
elif version_filter == "Historical only":
    active_filters["is_current"] = False
if doc_type_filter.strip():
    active_filters["document_type"] = doc_type_filter.strip()


# ── Rendering ────────────────────────────────────────────────────────────────
def render_result(result: dict) -> None:
    chunks = result.get("retrieved_chunks", []) or []
    latency = result.get("latency", {}) or {}

    if result.get("error"):
        st.error(f"The pipeline returned an error: {result['error']}")

    docs = {c.get("document") or c.get("source") for c in chunks}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Retrieval mode", MODE_LABELS.get(result["mode"], result["mode"]))
    m2.metric("Passages used", len(chunks))
    m3.metric("Documents cited", len(docs))
    m4.metric("Total latency", f"{sum(latency.values()) / 1000:.1f} s" if latency else "n/a")

    st.subheader("Answer")
    with st.container(border=True):
        st.markdown(_safe_md(result.get("answer") or "_No answer was generated._"))
    st.download_button(
        "Download brief (.md)",
        data=_export_markdown(st.session_state.current_query, result),
        file_name="research_brief.md",
        mime="text/markdown",
    )

    if chunks:
        st.subheader("Sources")
        st.caption(f"{len(chunks)} passages from {len(docs)} documents. Numbers match the citations in the answer.")
        for i, c in enumerate(chunks, start=1):
            doc_label = c.get("document") or c.get("source", "Unknown source")
            pages = f"p.{c['page_num']}"
            if c.get("page_end") and c["page_end"] != c["page_num"]:
                pages += f"-{c['page_end']}"
            section = f", {c['section']}" if c.get("section") else ""

            with st.expander(f"[{i}] {doc_label} ({pages}{section})", expanded=(i == 1)):
                meta = []
                if c.get("source"):
                    meta.append(f"File `{c['source']}`")
                if c.get("document_type"):
                    meta.append(f"Type `{c['document_type']}`")
                if c.get("version_label") or c.get("effective_date"):
                    v = f"Version `{c.get('version_label', '?')}`"
                    if c.get("effective_date"):
                        v += f" ({c['effective_date']})"
                    meta.append(v)
                if c.get("is_current") is True:
                    meta.append(":green[Current]")
                elif c.get("is_current") is False:
                    meta.append(":orange[Historical]")
                if c.get("content_type"):
                    meta.append(f"Content `{c['content_type']}`")
                if meta:
                    st.caption("   ·   ".join(meta))

                pills = ""
                for key, name, spec in [("rerank_score", "rerank", ".3f"), ("fusion_score", "RRF", ".4f"),
                                        ("semantic_score", "semantic", ".3f"), ("keyword_score", "BM25", ".3f")]:
                    s = _fmt(c.get(key), spec)
                    if s:
                        pills += f'<span class="pill">{name} {s}</span>'
                if pills:
                    st.markdown(pills, unsafe_allow_html=True)

                # Retrieved text is untrusted: always escape before injecting into HTML.
                st.markdown(f'<div class="passage">{html.escape(c.get("text", ""))}</div>',
                            unsafe_allow_html=True)
    elif not result.get("error"):
        st.info("No passages matched. Try removing filters or switching retrieval mode.")

    if show_diagnostics:
        with st.expander("Retrieval diagnostics", expanded=True):
            st.markdown(f"**Mode:** `{MODE_LABELS.get(result['mode'], result['mode'])}`")
            if latency:
                cols = st.columns(len(latency))
                for col, (k, v) in zip(cols, latency.items()):
                    col.metric(k.replace("_", " ").capitalize(), f"{v:.0f} ms")
            if chunks:
                rows = [{
                    "chunk": (c.get("chunk_id") or "")[:16],
                    "source": c.get("source"),
                    "page": c.get("page_num"),
                    "section": (c.get("section") or "")[:30],
                    "semantic": _fmt(c.get("semantic_score"), ".3f") or "-",
                    "bm25": _fmt(c.get("keyword_score"), ".3f") or "-",
                    "rrf": _fmt(c.get("fusion_score"), ".4f") or "-",
                    "rerank": _fmt(c.get("rerank_score"), ".3f") or "-",
                } for c in chunks]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            hr = result.get("hybrid_result")
            if hr and hasattr(hr, "debug"):
                for stage, stage_results in hr.debug.items():
                    if stage_results:
                        st.caption(f"{stage.replace('_', ' ').capitalize()}: {len(stage_results)} candidates")
            if active_filters:
                st.json(active_filters)


def render_design_tab() -> None:
    st.subheader("How it works")
    st.markdown(
        "1. **Retrieve.** The question is matched against indexed document chunks using dense vectors "
        "(FAISS), BM25 keyword search, or both fused with Reciprocal Rank Fusion.\n"
        "2. **Rerank.** In the default mode a cross-encoder re-scores the fused candidates.\n"
        "3. **Generate.** The top passages are sent to the language model, which answers from that context only.\n"
        "4. **Cite.** Every passage shows its file, page, section and version so claims can be checked."
    )
    st.markdown("**Retrieval modes available in this deployment**")
    for m in available_modes:
        st.markdown(f"- {MODE_LABELS.get(m, m)}")
    if CORPUS_DESCRIPTION:
        st.markdown("**Knowledge base**")
        st.write(CORPUS_DESCRIPTION)
    if EVAL_SUMMARY:
        st.markdown("**Evaluation**")
        st.write(EVAL_SUMMARY)
    if GITHUB_URL:
        st.markdown(f"[View the source on GitHub]({GITHUB_URL})")


# ── Main ─────────────────────────────────────────────────────────────────────
tab_research, tab_design = st.tabs(["Research", "How it works"])

with tab_research:
    if CORPUS_DESCRIPTION:
        st.caption(f"Knowledge base: {CORPUS_DESCRIPTION}")

    st.caption("Try an example")
    ex_cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        ex_cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True,
                              on_click=_use_example, args=(ex,))

    st.text_area(
        "Your question", key="query_text", height=100,
        placeholder="Ask about financial performance, strategy, risks or policy changes...",
        label_visibility="collapsed",
    )

    b1, b2, _ = st.columns([1.2, 1, 4])
    run_clicked = b1.button("Run research", type="primary", use_container_width=True)
    b2.button("Clear", use_container_width=True, on_click=_clear)

    if run_clicked:
        query = st.session_state.query_text.strip()
        if not query:
            st.warning("Enter a question first.")
        elif not faiss_ok and selected_mode not in PG_ONLY_MODES:
            st.error(f"The vector index is not available. {faiss_msg}")
        else:
            with st.spinner("Retrieving evidence and drafting the answer..."):
                result = run_query(
                    query=query, mode=selected_mode, top_k=top_k,
                    filters=active_filters or None, faiss_retriever=faiss_retriever,
                )
            st.session_state.history.append({"query": query, "result": result})
            st.session_state.current_result = result
            st.session_state.current_query = query
            st.rerun()

    result = st.session_state.current_result
    if result is None:
        with st.container(border=True):
            st.markdown("**No research yet.** Pick an example or type a question, then select *Run research*.")
            st.caption("Each answer lists the exact passages it was built from.")
    else:
        st.caption(f"Question: {st.session_state.current_query}")
        render_result(result)

with tab_design:
    render_design_tab()