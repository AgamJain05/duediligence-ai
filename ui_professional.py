"""

ui.py  ─  DueDiligenceAI Streamlit Interface

═════════════════════════════════════════════

Launch:

    streamlit run ui.py

This file is the ONLY new file in the project that touches Streamlit.

All retrieval and generation logic lives in the existing src/ modules.

"""

from __future__ import annotations

import os

import sys

from pathlib import Path

import streamlit as st

from dotenv import load_dotenv

# ── Environment and path setup ────────────────────────────────────────────────

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

# ── Adapter and constants ─────────────────────────────────────────────────────

from src.ui_backend import (

    VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED,

    POSTGRES_HNSW, POSTGRES_FILTERED, POSTGRES_FTS, POSTGRES_EXACT, POSTGRES_HYBRID,

    MODE_LABELS,

    check_faiss_index, check_database, check_llm,

    get_available_modes, build_faiss_retriever, run_query,

)

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(

    page_title="DueDiligenceAI",

    page_icon="🔍",

    layout="wide",

    initial_sidebar_state="expanded",

)

# ═══════════════════════════════════════════════════════════════════════════════

#  CSS

# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root { --ink: #172033; --muted: #64748b; --line: #e2e8f0; --surface: #ffffff; --canvas: #f6f8fb; --accent: #2457c5; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--canvas); color: var(--ink); }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3, h4 { letter-spacing: -0.025em; color: var(--ink); }
h1 { font-weight: 700; }
p, li { line-height: 1.7; }
[data-testid="stSidebar"] { background: #101828; border-right: 1px solid #253247; }
[data-testid="stSidebar"] * { color: #e5eaf2; }
[data-testid="stSidebar"] hr { border-color: #344054; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #aab6c8; }
.dd-header {
  background: linear-gradient(115deg, #101828 0%, #18365e 100%);
  border: 1px solid #263f60; border-radius: 16px;
  padding: 30px 34px; margin: 0 0 28px 0;
  box-shadow: 0 8px 24px rgba(16,24,40,.08);
}
.dd-header h1 { color: #fff; font-size: 2.05rem; font-weight: 700; margin: 0; letter-spacing: -.04em; }
.dd-header .sub { color: #c4d5ed; font-size: .98rem; margin-top: 8px; max-width: 760px; line-height: 1.6; }
.dd-badge { display:inline-flex; align-items:center; gap:6px; padding:5px 10px; border-radius:6px; font-size:.75rem; font-weight:600; margin:14px 6px 0 0; }
.dd-badge.ok { background:#e8f7ef; color:#17643a; border:1px solid #b9e5ca; }
.dd-badge.err { background:#fff0ef; color:#a12b25; border:1px solid #f3c3bf; }
.dd-badge.warn { background:#fff7e6; color:#8a5a00; border:1px solid #f2d28c; }
.answer-card { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:24px 28px; margin-top:12px; line-height:1.8; color:#263449; box-shadow:0 2px 8px rgba(16,24,40,.03); overflow-wrap:anywhere; }
.source-card { background:#fff; border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:10px; padding:14px 18px; margin-bottom:10px; font-size:.9rem; }
.source-card .src-title { color:#1d4ed8; font-weight:600; }
.source-card .src-meta { color:var(--muted); font-size:.8rem; margin:4px 0 8px; }
.source-card .src-text { color:#334155; line-height:1.7; }
.score-pill,.mode-badge { display:inline-block; padding:4px 9px; border-radius:5px; font-size:.73rem; font-weight:600; background:#edf3ff; color:#244fa5; border:1px solid #d5e2ff; margin:2px 4px 2px 0; }
.empty-state { text-align:center; background:#fff; border:1px dashed #cbd5e1; border-radius:16px; padding:52px 24px; color:#64748b; margin-top:18px; }
.empty-state .icon { font-size:2.4rem; }
.empty-state h3 { color:#263449; font-weight:600; margin-top:12px; }
div.stButton > button { border-radius:8px; font-weight:600; min-height:2.6rem; transition:all .15s ease; }
div.stButton > button[kind="primary"] { background:#2457c5; border:1px solid #2457c5; }
div.stButton > button[kind="primary"]:hover { background:#1946a3; border-color:#1946a3; }
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input, [data-testid="stSelectbox"] { border-radius:8px; }
[data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:10px; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px; }
@media (max-width: 768px) {
 .block-container { padding: 1rem 1rem 2rem; }
 .dd-header { padding:22px; }
 .dd-header h1 { font-size:1.65rem; }
 .answer-card { padding:18px; }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════

#  Session state initialisation

# ═══════════════════════════════════════════════════════════════════════════════

def _init_state():

    defaults = {

        "history": [],          # list of {query, result} dicts

        "current_query": "",

        "current_result": None,

        "selected_history_idx": None,

    }

    for k, v in defaults.items():

        if k not in st.session_state:

            st.session_state[k] = v

_init_state()

# ═══════════════════════════════════════════════════════════════════════════════

#  Backend checks (run once per session)

# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)

def _backend_status():

    faiss_ok, faiss_msg = check_faiss_index()

    db_ok, db_msg       = check_database()

    llm_ok, llm_msg     = check_llm()

    return faiss_ok, faiss_msg, db_ok, db_msg, llm_ok, llm_msg

faiss_ok, faiss_msg, db_ok, db_msg, llm_ok, llm_msg = _backend_status()

# ═══════════════════════════════════════════════════════════════════════════════

#  FAISS retriever (cached — loads once per session)

# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading retrieval index…")

def _get_faiss_retriever():

    if not faiss_ok:

        return None

    try:

        return build_faiss_retriever(mode=HYBRID_RERANKED)

    except Exception as exc:

        st.error(f"Failed to load FAISS index: {exc}")

        return None

faiss_retriever = _get_faiss_retriever()

# ═══════════════════════════════════════════════════════════════════════════════

#  HEADER

# ═══════════════════════════════════════════════════════════════════════════════

badges = ""

if faiss_ok:

    badges += '<span class="dd-badge ok">✓ FAISS Index</span>'

else:

    badges += '<span class="dd-badge err">✗ No FAISS Index</span>'

if db_ok:

    badges += '<span class="dd-badge ok">✓ PostgreSQL</span>'

else:

    badges += '<span class="dd-badge warn">○ PostgreSQL</span>'

if llm_ok:

    badges += '<span class="dd-badge ok">✓ LLM API</span>'

else:

    badges += '<span class="dd-badge err">✗ LLM API</span>'

st.markdown(f"""

<div class="dd-header">

    <h1>🔍 DueDiligenceAI</h1>

    <div class="sub">Research company documents with evidence-backed answers, source references, and configurable retrieval.</div>

    <div style="margin-top:12px">{badges}</div>

</div>

""", unsafe_allow_html=True)

if not llm_ok:

    st.warning(f"⚠️ {llm_msg}", icon="⚠️")

if not faiss_ok and not db_ok:

    st.error("No knowledge base is available. " + faiss_msg, icon="🚨")

# ═══════════════════════════════════════════════════════════════════════════════

#  SIDEBAR — Research settings

# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    st.markdown("## Research settings")

    st.markdown("---")

    # ── Retrieval mode ────────────────────────────────────────────────────────

    available_modes = get_available_modes(faiss_ok, db_ok)

    if not available_modes:

        st.error("No retrieval backend available.")

        available_modes = [HYBRID_RERANKED]

    mode_display  = [MODE_LABELS.get(m, m) for m in available_modes]

    default_idx   = available_modes.index(HYBRID_RERANKED) if HYBRID_RERANKED in available_modes else 0

    selected_label = st.selectbox(

        "Retrieval mode",

        options=mode_display,

        index=default_idx,

        help=(

            "**Vector**: Dense semantic search via FAISS cosine similarity.\n\n"

            "**Keyword**: Exact-term BM25 matching.\n\n"

            "**Hybrid**: Reciprocal Rank Fusion of both.\n\n"

            "**Hybrid + Reranking**: Cross-encoder re-scores the fused candidates.\n\n"

            "**PostgreSQL modes**: Require DATABASE_URL in .env."

        ),

    )

    selected_mode = available_modes[mode_display.index(selected_label)]

    # ── Top-K ─────────────────────────────────────────────────────────────────

    top_k = st.slider(

        "Retrieved results (Top-K)",

        min_value=1, max_value=20, value=5, step=1,

        help="Number of document chunks passed to the LLM as context.",

    )

    st.markdown("---")

    st.markdown("#### Retrieval filters")

    # ── Content type filter ───────────────────────────────────────────────────

    content_type = st.selectbox(

        "Content type",

        ["All", "text", "table", "list", "mixed"],

        index=0,

        help="Filter by chunk content type. Useful for 'table' when asking about financials.",

    )

    # ── Section filter ────────────────────────────────────────────────────────

    section_filter = st.text_input(

        "Section (substring match)",

        value="",

        placeholder="e.g. Risk, Financial, Strategy",

        help="Case-insensitive substring matched against section headings.",

    )

    # ── Version filter (Phase 4 only) ─────────────────────────────────────────

    if selected_mode in {POSTGRES_FILTERED, POSTGRES_HNSW, POSTGRES_HYBRID}:

        st.markdown("#### Document Version")

        version_filter = st.selectbox(

            "Document version",

            ["All versions", "Current only", "Historical only"],

            index=0,

            help="Filter by document version. 'Current only' = is_current=True in the DB.",

        )

        doc_type_filter = st.text_input(

            "Document type",

            value="",

            placeholder="e.g. annual_review, dossier",

            help="Exact match on document_type column.",

        )

    else:

        version_filter = "All versions"

        doc_type_filter = ""

    st.markdown("---")

    # ── Diagnostics toggle ────────────────────────────────────────────────────

    show_diagnostics = st.checkbox(

        "Show retrieval diagnostics",

        value=False,

        help="Reveal intermediate retrieval scores, chunk IDs, and latency breakdown.",

    )

    st.markdown("---")

    st.caption(f"Model: `{os.getenv('OPENROUTER_MODEL', 'google/gemma-4-31b-it')}`")

    st.caption(f"Index: `data/index/`")

# ═══════════════════════════════════════════════════════════════════════════════

#  Assemble filters dict

# ═══════════════════════════════════════════════════════════════════════════════

active_filters: dict = {}

if content_type and content_type != "All":

    # Phase 3 filter key vs Phase 4 filter key

    if selected_mode in {POSTGRES_FILTERED, POSTGRES_HNSW, POSTGRES_FTS, POSTGRES_HYBRID, POSTGRES_EXACT}:

        active_filters["content_type"] = content_type

    else:

        active_filters["content_type"] = content_type

if section_filter.strip():

    active_filters["section"] = section_filter.strip()

if version_filter == "Current only":

    active_filters["is_current"] = True

elif version_filter == "Historical only":

    active_filters["is_current"] = False

if doc_type_filter.strip():

    active_filters["document_type"] = doc_type_filter.strip()

# ═══════════════════════════════════════════════════════════════════════════════

#  MAIN — Query input

# ═══════════════════════════════════════════════════════════════════════════════

col_main, col_hist = st.columns([3, 1], gap="large")

with col_main:

    st.markdown("### Research workspace")

    # ── Example questions ─────────────────────────────────────────────────────

    EXAMPLES = [

        "What were the company's financial highlights?",

        "What are its major business and operational risks?",

        "Compare management's statements with the formal risk disclosures.",

        "What is the current data residency requirement?",

        "What changed between the 2024 and 2026 risk documents?",

        "What evidence supports the company's growth strategy?",

    ]

    st.markdown("<div style='font-size:0.85rem;color:#64748b;margin-bottom:4px'>Example questions:</div>",

                unsafe_allow_html=True)

    ex_cols = st.columns(3)

    for i, ex in enumerate(EXAMPLES):

        if ex_cols[i % 3].button(

            ex[:45] + ("…" if len(ex) > 45 else ""),

            key=f"ex_{i}",

            use_container_width=True,

        ):

            st.session_state.current_query = ex

    # ── Text area ─────────────────────────────────────────────────────────────

    query_input = st.text_area(

        "Your question",

        value=st.session_state.current_query,

        height=100,

        placeholder="Ask a question about the company's financial performance, strategy, products, or risks…",

        label_visibility="collapsed",

        key="query_textarea",

    )

    # ── Action buttons ────────────────────────────────────────────────────────

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])

    run_clicked   = btn_col1.button("Run research", type="primary", use_container_width=True)

    clear_clicked = btn_col2.button("Clear", use_container_width=True)

    if clear_clicked:

        st.session_state.current_query  = ""

        st.session_state.current_result = None

        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════

#  History sidebar column

# ═══════════════════════════════════════════════════════════════════════════════

with col_hist:

    st.markdown("### Recent research")

    if not st.session_state.history:

        st.caption("No queries yet this session.")

    else:

        for idx, entry in enumerate(reversed(st.session_state.history)):

            btn_label = entry["query"][:40] + ("…" if len(entry["query"]) > 40 else "")

            if st.button(btn_label, key=f"hist_{idx}", use_container_width=True):

                st.session_state.current_result = entry["result"]

                st.session_state.current_query  = entry["query"]

                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════

#  Execute query

# ═══════════════════════════════════════════════════════════════════════════════

if run_clicked:

    query = query_input.strip()

    if not query:

        st.warning("Please enter a question before running research.", icon="⚠️")

    elif not faiss_ok and selected_mode not in {POSTGRES_HNSW, POSTGRES_FILTERED, POSTGRES_FTS, POSTGRES_EXACT}:

        st.error("FAISS index not available. " + faiss_msg, icon="🚨")

    else:

        with st.spinner("Retrieving evidence and generating answer…"):

            result = run_query(

                query=query,

                mode=selected_mode,

                top_k=top_k,

                filters=active_filters if active_filters else None,

                faiss_retriever=faiss_retriever,

            )

        # Save to history

        st.session_state.history.append({"query": query, "result": result})

        st.session_state.current_result = result

        st.session_state.current_query  = query

        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════

#  Display result

# ═══════════════════════════════════════════════════════════════════════════════

result = st.session_state.current_result

with col_main:

    if result is None:

        # ── Empty state ───────────────────────────────────────────────────────

        st.markdown("""

<div class="empty-state">

    <div class="icon">🔍</div>

    <h3>Ask a question to begin your research</h3>

    <p style="font-size:0.85rem">Type a question or select an example above, then click <strong>Run Research</strong>.</p>

</div>""", unsafe_allow_html=True)

    else:

        # ── Error banner ──────────────────────────────────────────────────────

        if result.get("error"):

            st.error(f"**Pipeline error:** {result['error']}", icon="🚨")

        # ── Answer ────────────────────────────────────────────────────────────

        mode_label = MODE_LABELS.get(result["mode"], result["mode"])

        st.markdown(

            f'<div style="margin-top:8px">'

            f'<span class="mode-badge">{mode_label}</span>'

            f'&nbsp;&nbsp;<span style="font-size:0.8rem;color:#64748b">'

            f'Model: {result["model_used"]} &nbsp;|&nbsp; '

            f'{len(result["retrieved_chunks"])} chunk(s) retrieved</span>'

            f'</div>',

            unsafe_allow_html=True

        )

        st.markdown("#### Research answer")

        st.markdown(

            f'<div class="answer-card">{result["answer"]}</div>',

            unsafe_allow_html=True,

        )

        # ── Sources & Evidence ────────────────────────────────────────────────

        chunks = result.get("retrieved_chunks", [])

        if chunks:

            st.markdown("---")

            st.markdown(f"#### Sources and evidence &nbsp; <span style='font-size:0.8rem;color:#64748b'>({len(chunks)} retrieved)</span>",

                        unsafe_allow_html=True)

            # Deduplicate by (source, page_num)

            seen_keys: set = set()

            unique_chunks = []

            for c in chunks:

                key = (c["source"], c["page_num"])

                if key not in seen_keys:

                    seen_keys.add(key)

                    unique_chunks.append(c)

            for i, chunk in enumerate(chunks, start=1):

                # Build score pills

                pills = ""

                if chunk.get("rerank_score") is not None:

                    pills += f'<span class="score-pill">rerank {chunk["rerank_score"]:.3f}</span>'

                if chunk.get("fusion_score") is not None:

                    pills += f'<span class="score-pill">RRF {chunk["fusion_score"]:.4f}</span>'

                if chunk.get("semantic_score") is not None:

                    pills += f'<span class="score-pill">sem {chunk["semantic_score"]:.3f}</span>'

                if chunk.get("keyword_score") is not None:

                    pills += f'<span class="score-pill">bm25 {chunk["keyword_score"]:.3f}</span>'

                # Build title

                doc_label = chunk.get("document") or chunk.get("source", "Unknown source")

                page_range = f"p.{chunk['page_num']}"

                if chunk.get("page_end") and chunk["page_end"] != chunk["page_num"]:

                    page_range = f"p.{chunk['page_num']}–{chunk['page_end']}"

                section_str = f" · {chunk['section']}" if chunk.get("section") else ""

                version_str = ""

                if chunk.get("version_label"):

                    version_str = f" · v{chunk['version_label']}"

                if chunk.get("effective_date"):

                    version_str += f" ({chunk['effective_date']})"

                with st.expander(f"[{i}] {doc_label}  —  {page_range}{section_str}", expanded=(i == 1)):

                    # Metadata row

                    meta_parts = []

                    if chunk.get("source"):

                        meta_parts.append(f"📄 `{chunk['source']}`")

                    if chunk.get("document_type"):

                        meta_parts.append(f"Type: `{chunk['document_type']}`")

                    if chunk.get("version_label") or chunk.get("effective_date"):

                        meta_parts.append(f"Version: `{chunk.get('version_label','?')}` {version_str}")

                    if chunk.get("is_current") is True:

                        meta_parts.append("🟢 Current")

                    elif chunk.get("is_current") is False:

                        meta_parts.append("🕰 Historical")

                    if chunk.get("content_type"):

                        meta_parts.append(f"Content: `{chunk['content_type']}`")

                    if meta_parts:

                        st.markdown("  ".join(meta_parts))

                    # Scores

                    if pills:

                        st.markdown(pills, unsafe_allow_html=True)

                    # Passage

                    st.markdown("---")

                    st.markdown(

                        f'<div style="font-size:0.9rem;color:#cbd5e1;line-height:1.7">{chunk["text"]}</div>',

                        unsafe_allow_html=True

                    )

        elif not result.get("error"):

            st.info("No relevant chunks were retrieved for this query.", icon="ℹ️")

        # ── Retrieval Diagnostics ─────────────────────────────────────────────

        if show_diagnostics:

            st.markdown("---")

            with st.expander("🔬 Retrieval Diagnostics", expanded=True):

                mode_label_full = MODE_LABELS.get(result["mode"], result["mode"])

                st.markdown(f"**Retrieval mode:** `{mode_label_full}`")

                # Latency

                latency = result.get("latency", {})

                if latency:

                    st.markdown("**Latency breakdown:**")

                    lat_cols = st.columns(len(latency))

                    for col, (k, v) in zip(lat_cols, latency.items()):

                        col.metric(k.replace("_", " "), f"{v:.0f} ms")

                # Chunk table

                if chunks:

                    st.markdown("**Retrieved chunk IDs and scores:**")

                    import pandas as pd

                    rows = []

                    for c in chunks:

                        rows.append({

                            "chunk_id": c["chunk_id"][:16] + "…",

                            "source": c["source"],

                            "page": c["page_num"],

                            "section": (c.get("section") or "")[:30],

                            "sem": f"{c['semantic_score']:.3f}" if c.get("semantic_score") is not None else "—",

                            "bm25": f"{c['keyword_score']:.3f}" if c.get("keyword_score") is not None else "—",

                            "rrf": f"{c['fusion_score']:.4f}" if c.get("fusion_score") is not None else "—",

                            "rerank": f"{c['rerank_score']:.3f}" if c.get("rerank_score") is not None else "—",

                        })

                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Debug lists from HybridSearchResult

                hr = result.get("hybrid_result")

                if hr and hasattr(hr, "debug"):

                    for stage, stage_results in hr.debug.items():

                        if stage_results:

                            st.markdown(f"**{stage.replace('_', ' ').title()}:** {len(stage_results)} candidates")

                # Active filters

                if active_filters:

                    st.markdown("**Active filters:**")

                    st.json(active_filters)
