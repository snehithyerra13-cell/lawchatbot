"""ConstitutionGPT — Indian Constitution Assistant.

Demonstrates all 10 Core LangGraph Architectural Concepts:
1. 💾 Persistence (Disk durability with SQLite)
2. 📌 Checkpointers (SqliteSaver snapshot tree)
3. 🏪 Stores (Cross-thread memory: bookmarks & preferences via InMemoryStore)
4. 🛡️ Fault tolerance (RetryPolicy with exponential backoff on nodes)
5. ⚡ Event streaming (Live node progression stream)
6. 🌊 Streaming (Token/word-by-word typing output)
7. ⏸️ Interrupts (Human-in-the-Loop review & approval)
8. ⏳ Time travel (Checkpoint snapshot inspection & branching)
9. 🧠 Memory (In-thread short-term messages + cross-thread store profile)
10. 🔀 Subgraphs (Modular Retrieval & Verification Subgraph)
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.graph.workflow import (
    branch_from_checkpoint,
    get_compiled_graph,
    get_state_at_checkpoint,
    get_thread_history_snapshots,
    resume_constitution_chat,
    run_constitution_chat,
    run_constitution_chat_stream,
    time_travel_rerun_with_gemini,
)
from app.persistence.checkpointer import (
    delete_thread_checkpoints,
    discard_thread_interrupted_turn,
    get_sqlite_checkpointer,
)
from app.persistence.conversation_db import (
    generate_title_from_query,
    get_conversation_db,
)
from app.persistence.store import (
    bookmark_article,
    get_global_store,
    get_user_preferences,
    is_article_bookmarked,
    list_bookmarks,
    remove_bookmark,
    sanitize_store_namespace,
    update_user_preferences,
)
from app.retrieval.knowledge_base import get_knowledge_base
from app.auth.auth0_service import Auth0Service, get_auth0_service
from app.memory.mem0_service import Mem0Service, get_mem0_service
from config.settings import get_settings

# --- Page Configuration ---
st.set_page_config(
    page_title="ConstitutionGPT — Indian Constitution Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 14px;
        background-color: #e8f0fe;
        color: #1a73e8;
        font-size: 0.80rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #c2e7ff;
    }
    .badge-gemini {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 14px;
        background-color: #f3e8fd;
        color: #7b1fa2;
        font-size: 0.80rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #e1bee7;
    }
    .badge-dual {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 14px;
        background-color: #e6f4ea;
        color: #137333;
        font-size: 0.80rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #ceead6;
    }
    .badge-pillar {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 10px;
        background-color: #fff8e1;
        color: #b78103;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
        margin-bottom: 4px;
        border: 1px solid #ffe082;
    }
    .chat-header {
        margin-top: -10px;
        margin-bottom: 12px;
    }
    .interrupt-box {
        background-color: #fff3e0;
        border: 1px solid #ffe0b2;
        border-left: 5px solid #ff9800;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .stButton > button {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()
conv_db = get_conversation_db()
kb = get_knowledge_base()
checkpointer = get_sqlite_checkpointer()
global_store = get_global_store()


def stream_text_word_by_word(
    text: str, delay: Optional[float] = None
) -> Generator[str, None, None]:
    """Yield text token by token to produce a visible, smooth ChatGPT-style typewriter streaming effect."""
    if not text:
        return
    import re
    tokens = re.findall(r"\S+|\n", text)
    if delay is None:
        # Dynamic pacing: ~2.5 to 3.5s total typing duration regardless of text length
        delay = max(0.012, min(0.035, 3.0 / max(len(tokens), 1)))
    for idx, token in enumerate(tokens):
        if token == "\n":
            yield "\n"
        else:
            yield token + (" " if idx < len(tokens) - 1 else "")
        time.sleep(delay)


def extract_source_info(msg_or_result: Any) -> Dict[str, Any]:
    """Extract structured source provenance metadata from an AIMessage or turn result dict."""
    resp_type = None
    intent = None
    gemini_diag = None
    if isinstance(msg_or_result, AIMessage):
        kwargs = getattr(msg_or_result, "additional_kwargs", {}) or {}
        exec_time = kwargs.get("execution_time_seconds")
        sources = kwargs.get("sources") or []
        retrieved_docs = kwargs.get("retrieved_documents") or []
        art_num = kwargs.get("article_number")
        topic = kwargs.get("topic")
        resp_type = kwargs.get("response_type")
        intent = kwargs.get("intent")
        rerank_meta = kwargs.get("rerank_metadata")
        gemini_diag = kwargs.get("gemini_diagnostic")
    elif isinstance(msg_or_result, dict):
        exec_time = msg_or_result.get("execution_time_seconds")
        sources = msg_or_result.get("sources") or []
        retrieved_docs = msg_or_result.get("retrieved_documents") or []
        art_num = msg_or_result.get("article_number") or msg_or_result.get("last_discussed_article")
        topic = msg_or_result.get("topic")
        resp_type = msg_or_result.get("response_type")
        intent = msg_or_result.get("intent")
        rerank_meta = msg_or_result.get("rerank_metadata")
        gemini_diag = msg_or_result.get("gemini_diagnostic")
    else:
        return {}

    docs = sources if sources else retrieved_docs
    d0 = docs[0] if docs else {}
    p_start = d0.get("page_start", "")
    p_end = d0.get("page_end", p_start)
    page_str = (
        f"Page {p_start}"
        if not p_end or p_start == p_end
        else f"Pages {p_start}–{p_end}"
    ) if p_start else "Official Gazette"

    return {
        "exec_time": exec_time,
        "source_pdf": d0.get("source_pdf", "constitution.pdf"),
        "page_str": page_str,
        "page_start": p_start,
        "page_end": p_end,
        "article_number": d0.get("article_number") or art_num or "",
        "article_title": d0.get("article_title", ""),
        "part": d0.get("part", "Part III"),
        "part_title": d0.get("part_title", ""),
        "source_hash": d0.get("source_hash", "c2e0f5f7a670d362..."),
        "total_sources": len(docs),
        "docs": docs,
        "response_type": resp_type,
        "intent": intent,
        "rerank_score": d0.get("rerank_score"),
        "rerank_reason": d0.get("rerank_reason"),
        "rerank_metadata": rerank_meta,
        "gemini_diagnostic": gemini_diag,
    }


def is_gemini_synthesis_error(gemini_content: str, gemini_diag: Optional[Dict[str, Any]] = None) -> bool:
    """Determine whether Gemini AI output represents a failure rather than successful generation."""
    if gemini_diag and isinstance(gemini_diag, dict):
        if gemini_diag.get("success") is False:
            return True
        if gemini_diag.get("error_type") and gemini_diag.get("error_type") != "none":
            return True
    if not gemini_content:
        return False
    stripped = gemini_content.strip()
    return (
        stripped.startswith("⚠️")
        or stripped.startswith("❌")
        or stripped.startswith("⚙️")
        or stripped.startswith("⏱️")
        or stripped.startswith("🔒")
        or stripped.startswith("☁️")
        or "Gemini AI synthesis is temporarily unavailable" in stripped
        or "Google Gemini Service Unavailable" in stripped
        or "Gemini AI Rate Limit" in stripped
        or "Gemini AI Quota" in stripped
    )


def render_message_provenance_bar(info: Dict[str, Any]) -> None:
    """Render a clean, non-intrusive provenance line and verified legal citations."""
    if not info:
        return

    art_label = f"Article {info['article_number']}" if info.get('article_number') else "Constitutional Analysis"
    page_str = info.get("page_str", "Official Gazette")
    part_str = info.get("part", "Part III")
    doc_str = info.get("source_pdf", "constitution.pdf")

    time_badge = ""
    if info.get("exec_time") is not None:
        time_badge = f'<span style="font-size: 0.74rem; font-weight: 600; color: #166534; background: #f0fdf4; padding: 2px 8px; border-radius: 6px; border: 1px solid #bbf7d0;">⏱️ {info["exec_time"]:.2f}s</span>'

    rerank_badge = ""
    if info.get("rerank_score") is not None:
        rerank_badge = f'<span style="font-size: 0.72rem; color: #6b21a8; background: #f3e8ff; padding: 2px 7px; border-radius: 6px; font-weight: 600; border: 1px solid #e9d5ff;">🔄 Rerank: {info["rerank_score"]}/100</span>'
    elif info.get("rerank_metadata") and info["rerank_metadata"].get("method") == "none":
        rerank_badge = '<span style="font-size: 0.72rem; color: #0369a1; background: #e0f2fe; padding: 2px 7px; border-radius: 6px; font-weight: 600; border: 1px solid #bae6fd;">⚡ Direct RAG</span>'

    right_items = [b for b in [rerank_badge, time_badge] if b]
    right_html = f'<div style="display: flex; align-items: center; gap: 6px;">{" ".join(right_items)}</div>' if right_items else ""

    bar_html = (
        f'<div style="margin-top: 10px; margin-bottom: 6px; padding: 6px 12px; background: #f8fafc; '
        f'border: 1px solid #e2e8f0; border-radius: 8px; display: flex; align-items: center; '
        f'justify-content: space-between; flex-wrap: wrap; gap: 8px;">'
        f'<div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap; font-size: 0.78rem; color: #334155;">'
        f'<span style="font-weight: 600; color: #0f172a;">📄 {doc_str}</span>'
        f'<span style="color: #94a3b8;">&bull;</span>'
        f'<span style="color: #1a73e8; font-weight: 600;">{art_label}</span>'
        f'<span style="color: #64748b;">({part_str}, {page_str})</span>'
        f'</div>'
        f'{right_html}'
        f'</div>'
    )
    st.markdown(bar_html, unsafe_allow_html=True)

    if info.get("docs") and len(info["docs"]) > 1:
        with st.expander(f"📚 All {len(info['docs'])} Citations from Constitution PDF", expanded=False):
            for idx, d in enumerate(info["docs"]):
                a_num = d.get("article_number", "")
                p_s = d.get("page_start", "")
                p_e = d.get("page_end", p_s)
                pg = f"Page {p_s}" if p_s == p_e or not p_e else f"Pages {p_s}–{p_e}"
                st.caption(f"**#{idx+1}: Article {a_num}** ({pg}, {d.get('part', 'Part III')}) — {d.get('article_title', '')}")


def render_source_sidebar_card(info: Dict[str, Any]) -> None:
    """Legacy alias redirecting to clean provenance bar."""
    render_message_provenance_bar(info)


def render_assistant_metadata_and_sources(msg_or_result: Any) -> None:
    """Render execution time badge and document provenance from constitution.pdf."""
    info = extract_source_info(msg_or_result)
    if not info:
        return

    # If out of scope or uncertain, do not display PDF citation
    if info.get("response_type") in ["out_of_scope", "uncertain"] or info.get("intent") in ["out_of_scope", "uncertain"]:
        if info.get("exec_time") is not None:
            time_badge = f"""
            <div style="margin-top: 10px; margin-bottom: 6px;">
                <span style="display: inline-flex; align-items: center; gap: 4px; padding: 2px 9px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; background-color: #fef3c7; color: #b45309; border: 1px solid #fde68a;">
                    ⏱️ Execution time: {info['exec_time']:.2f}s
                </span>
            </div>
            """
            st.markdown(time_badge, unsafe_allow_html=True)
        return

    # Performance Badge
    time_badge = ""
    if info.get("exec_time") is not None:
        time_badge = f"""
        <span style="display: inline-flex; align-items: center; gap: 4px; padding: 2px 9px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; background-color: #f1f8e9; color: #2e7d32; border: 1px solid #c8e6c9;">
            ⏱️ Execution time: {info['exec_time']:.2f}s
        </span>
        """

    meta_html = f"""
    <div style="margin-top: 10px; margin-bottom: 6px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center;">
        {time_badge}
        <span style="display: inline-flex; align-items: center; gap: 4px; padding: 2px 9px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; background-color: #e8f0fe; color: #1a73e8; border: 1px solid #c2e7ff;">
            📄 Source: {info['source_pdf']} ({info['page_str']})
        </span>
    </div>
    """
    st.markdown(meta_html, unsafe_allow_html=True)


def render_engine_footer(engine_type: str = "pdf") -> None:
    """Render a light font indicator directly below the answer (e.g. 'pdf answer' or 'ai generated')."""
    if engine_type == "pdf":
        st.markdown(
            """<div style="color: #9aa0a6; font-size: 0.77rem; font-style: italic; margin-top: 8px; margin-bottom: 4px; display: flex; align-items: center; gap: 5px;">
                <span>📄 pdf answer</span>
            </div>""",
            unsafe_allow_html=True,
        )
    elif engine_type == "error":
        st.markdown(
            """<div style="color: #b45309; font-size: 0.77rem; font-style: italic; margin-top: 8px; margin-bottom: 4px; display: flex; align-items: center; gap: 5px;">
                <span>⚠️ ai synthesis unavailable • verified rule-based pdf fallback</span>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="color: #9aa0a6; font-size: 0.77rem; font-style: italic; margin-top: 8px; margin-bottom: 4px; display: flex; align-items: center; gap: 5px;">
                <span>✨ ai generated</span>
            </div>""",
            unsafe_allow_html=True,
        )


def execute_query_chatgpt_style(
    query: str,
    thread_id: str,
    selected_engine: str = "both",
    enable_interrupt: bool = False,
    rerank_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute query with ChatGPT-style dynamic reasoning progression and continuous live stopwatch.

    Rather than freezing between nodes, a worker thread streams LangGraph events while the
    main UI thread continuously updates the stopwatch counter at ~16 FPS.
    The time counter ticks continuously without stopping, while steps appear, transition,
    and disappear dynamically. Upon completion, the active step collapses into a sleek
    ChatGPT-style 'Thought for X.Xs' summary badge.
    """
    import queue
    import threading

    t_start = time.perf_counter()
    step_placeholder = st.empty()
    step_history: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {}
    event_queue: queue.Queue = queue.Queue()

    def _worker() -> None:
        user_id = _active_uid()
        try:
            for update in run_constitution_chat_stream(
                query,
                thread_id,
                checkpointer=checkpointer,
                store=global_store,
                enable_interrupt=enable_interrupt,
                selected_engine=selected_engine,
                user_id=user_id,
                rerank_mode=rerank_mode,
            ):
                event_queue.put(update)
        except Exception as e:
            event_queue.put({"type": "error", "error": str(e)})

    worker_thread = threading.Thread(target=_worker, daemon=True)
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        add_script_run_ctx(worker_thread)
    except Exception:
        pass
    worker_thread.start()

    current_step: Dict[str, Any] = {
        "icon": "⚖️",
        "label": "Analyzing Query",
        "summary": "Initializing constitutional legal pipeline...",
        "step_num": 1,
    }

    # Continuous live timer loop ticking smoothly without stopping
    while worker_thread.is_alive() or not event_queue.empty():
        # Drain all available events from the queue
        while not event_queue.empty():
            try:
                item = event_queue.get_nowait()
            except queue.Empty:
                break

            if item["type"] == "node_update":
                icon = item.get("icon", "⚙️")
                label = item.get("label", "Processing")
                summary = item.get("summary", "")
                step_num = len(step_history) + 1
                current_step = {
                    "icon": icon,
                    "label": label,
                    "summary": summary,
                    "step_num": step_num,
                }
                step_history.append({
                    "icon": icon,
                    "label": label,
                    "summary": summary,
                    "elapsed": round(time.perf_counter() - t_start, 2),
                    "step_num": step_num,
                })
            elif item["type"] == "complete":
                result = item.get("result", {})
            elif item["type"] == "error":
                result = {"answer": f"Error: {item.get('error')}", "error": item.get("error")}

        # Exact elapsed time counting continuously
        elapsed = time.perf_counter() - t_start

        step_placeholder.markdown(
            f"""
            <div style="background: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%); border: 1.5px solid #cbd5e1; border-left: 4px solid #1a73e8; border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.25rem;">{current_step['icon']}</span>
                    <div>
                        <div style="font-weight: 700; color: #0f172a; font-size: 0.88rem;">{current_step['label']}</div>
                        <div style="color: #475569; font-size: 0.82rem; margin-top: 1px;">{current_step['summary']}</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-family: monospace; font-size: 0.82rem; font-weight: 700; color: #15803d; background: #dcfce7; padding: 3px 10px; border-radius: 12px; border: 1px solid #86efac; box-shadow: 0 1px 2px rgba(0,0,0,0.04); display: inline-flex; align-items: center; gap: 4px;">
                        ⏱️ {elapsed:.2f}s
                    </span>
                    <span style="font-size: 0.72rem; color: #0284c7; background: #e0f2fe; padding: 3px 8px; border-radius: 10px; font-weight: 700;">
                        STEP {current_step['step_num']}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.06)

    # Ensure worker has fully finished and final result is captured
    worker_thread.join(timeout=1.0)
    while not event_queue.empty():
        try:
            item = event_queue.get_nowait()
            if item["type"] == "complete":
                result = item.get("result", {})
            elif item["type"] == "node_update" and item.get("label"):
                step_num = len(step_history) + 1
                step_history.append({
                    "icon": item.get("icon", "⚙️"),
                    "label": item.get("label", "Processing"),
                    "summary": item.get("summary", ""),
                    "elapsed": round(time.perf_counter() - t_start, 2),
                    "step_num": step_num,
                })
        except queue.Empty:
            break

    total_elapsed = result.get("execution_time_seconds") or round(time.perf_counter() - t_start, 2)

    # Active step disappears and collapses into ChatGPT-style thought summary badge with total counted time
    step_placeholder.markdown(
        f"""
        <div style="display: inline-flex; align-items: center; gap: 8px; padding: 4px 14px; border-radius: 14px; background: #f8fafc; border: 1px solid #e2e8f0; font-size: 0.80rem; color: #475569; font-weight: 600; margin-bottom: 10px;">
            <span>💭</span>
            <span>Thought for {total_elapsed:.2f}s ({len(step_history)} verified steps)</span>
            <span style="font-family: monospace; font-size: 0.75rem; color: #166534; background: #dcfce7; padding: 2px 8px; border-radius: 10px; border: 1px solid #86efac; font-weight: 700;">
                ⏱️ {total_elapsed:.2f}s
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if step_history:
        with st.expander(f"🔍 Inspect thought trace ({len(step_history)} steps)", expanded=False):
            for s in step_history:
                st.markdown(
                    f"`⏱️ {s['elapsed']:.2f}s` &nbsp; {s['icon']} **{s['label']}**: {s['summary']}"
                )

    return result


# --- Initialize Auth0 & Mem0 Services ---
auth_svc = get_auth0_service()
mem0_svc = get_mem0_service()

# Process Auth0 OAuth Code from callback redirect
_params = dict(st.query_params)
if "code" in _params:
    oauth_code = _params.get("code")
    oauth_state = _params.get("state", "")
    # Clear params immediately to prevent re-processing on rerun
    st.query_params.clear()

    if oauth_code and not st.session_state.get("_auth0_code_processed") == oauth_code:
        st.session_state["_auth0_code_processed"] = oauth_code
        with st.spinner("🔐 Authenticating with Auth0..."):
            # Reinitialise service to pick up latest credentials from settings/env
            import app.auth.auth0_service as _a0_mod
            _a0_mod._auth0_instance = None
            fresh_auth_svc = get_auth0_service()
            auth_res = fresh_auth_svc.exchange_code_for_user(oauth_code)

        if auth_res.get("success"):
            st.session_state.auth_user = auth_res["user"]
            st.session_state["_login_done"] = True
            st.toast(f"✅ Welcome {auth_res['user']['name']}! Logged in via Auth0.", icon="🔐")
        else:
            err = auth_res.get("error", "Unknown error")
            st.error(f"❌ Auth0 Login Failed: {err}")
            st.info("💡 Check that your Client Secret is correct in ⚙️ Auth0 & Mem0 Settings.")

    st.rerun()

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# ============================================================
# LOGIN GATE — shown if user is not authenticated
# ============================================================
def render_login_page() -> None:
    """Full-screen compass-themed login / landing page with multi-user isolation."""
    st.markdown(
        """
        <style>
        /* Hide Streamlit default sidebar & header on login page */
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stHeader"]  { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
        footer { visibility: hidden !important; }
        #MainMenu { visibility: hidden !important; }

        /* Full viewport gradient background */
        .login-bg {
            position: fixed; inset: 0;
            background: linear-gradient(135deg, #0a1628 0%, #0d2b5e 40%, #1a1040 100%);
            z-index: -1;
        }

        /* Compass rose decoration */
        .compass-ring {
            width: 240px; height: 240px;
            border-radius: 50%;
            border: 2px solid rgba(212,175,55,0.35);
            position: relative; margin: 0 auto;
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 0 60px rgba(212,175,55,0.10), inset 0 0 40px rgba(212,175,55,0.05);
        }
        .compass-inner {
            width: 180px; height: 180px;
            border-radius: 50%;
            border: 1.5px solid rgba(212,175,55,0.25);
            display: flex; align-items: center; justify-content: center;
        }
        .compass-glyph {
            font-size: 5rem; line-height: 1;
            filter: drop-shadow(0 0 18px rgba(212,175,55,0.6));
        }

        /* Card container */
        .login-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(212,175,55,0.20);
            border-radius: 20px;
            padding: 36px 32px 28px;
            max-width: 500px;
            margin: 0 auto;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 40px rgba(0,0,0,0.45);
        }

        /* Headings */
        .login-title {
            font-size: 2.3rem; font-weight: 800;
            color: #d4af37;
            text-align: center; margin: 18px 0 4px;
            letter-spacing: -0.5px;
        }
        .login-sub {
            font-size: 1rem; font-weight: 400;
            color: rgba(255,255,255,0.70);
            text-align: center; margin-bottom: 4px;
        }
        .login-tagline {
            text-align: center; font-size: 0.78rem;
            color: rgba(212,175,55,0.75);
            font-style: italic; margin-bottom: 22px;
            letter-spacing: 0.4px;
        }

        /* Divider */
        .login-divider {
            border: none; border-top: 1px solid rgba(255,255,255,0.12);
            margin: 18px 0;
        }

        /* Feature chips */
        .feature-chips {
            display: flex; flex-wrap: wrap; gap: 6px;
            justify-content: center; margin: 14px 0 20px;
        }
        .feature-chip {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(212,175,55,0.22);
            border-radius: 18px;
            padding: 3px 10px;
            font-size: 0.72rem;
            color: rgba(255,255,255,0.80);
        }

        .login-section-title {
            color: rgba(255,255,255,0.88);
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .footer-note {
            text-align: center; font-size: 0.70rem;
            color: rgba(255,255,255,0.35); margin-top: 16px;
        }
        </style>
        <div class="login-bg"></div>
        """,
        unsafe_allow_html=True,
    )

    # Centre the card with columns
    _, col_card, _ = st.columns([0.15, 0.70, 0.15])
    with col_card:
        # Compass rose
        st.markdown(
            """
            <div style="margin-top: 25px; margin-bottom: 0;">
              <div class="compass-ring">
                <div class="compass-inner">
                  <span class="compass-glyph">⚖️</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Title block
        st.markdown(
            """
            <div class="login-title">ConstitutionGPT</div>
            <div class="login-sub">Your AI Constitutional Legal Compass</div>
            <div class="login-tagline">Navigate the Constitution of India with isolated, personal storage</div>
            """,
            unsafe_allow_html=True,
        )

        # Feature chips
        st.markdown(
            """
            <div class="feature-chips">
              <span class="feature-chip">🔒 Isolated User Storage</span>
              <span class="feature-chip">🏛️ PDF Rule Engine</span>
              <span class="feature-chip">✨ Gemini AI</span>
              <span class="feature-chip">🧠 Mem0 Memory</span>
              <span class="feature-chip">📌 LangGraph</span>
              <span class="feature-chip">💾 Private Sessions</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='login-divider'>", unsafe_allow_html=True)

        # --- Auth0 Universal Login (Sole Authentication Entry) ---
        _fresh_auth_svc = get_auth0_service()
        if _fresh_auth_svc.is_configured:
            auth_url = _fresh_auth_svc.get_authorize_url()
            st.markdown(
                f"""
                <div style="margin: 20px 0 10px;">
                  <a href="{auth_url}" target="_self"
                     style="display:block; background:linear-gradient(90deg,#1a73e8,#0d47a1);
                            color:white; text-align:center; padding:15px; border-radius:12px;
                            font-weight:700; font-size:1.05rem; text-decoration:none;
                            border:none; box-shadow:0 4px 18px rgba(26,115,232,0.45);
                            transition:opacity 0.2s;">
                    🔐 &nbsp; Sign in with Auth0
                  </a>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='text-align:center; color:rgba(255,255,255,0.50); font-size:0.78rem; margin-top:10px;'>"
                "🔒 Each Auth0 account gets its own isolated conversation history, bookmarks & memories."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background:rgba(255,152,0,0.12); border:1px solid rgba(255,152,0,0.35);
                     border-radius:10px; padding:14px 18px; margin: 15px 0; font-size:0.85rem;
                     color:rgba(255,200,100,0.95); text-align:center;">
                  ⚠️ <strong>Auth0 Credentials Required</strong><br>
                  Please ensure <code>AUTH0_DOMAIN</code> and <code>AUTH0_CLIENT_ID</code> are configured in <code>.env</code>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='login-divider'>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="footer-note">
              🇮🇳 Grounded in the Official Constitution of India PDF &nbsp;|&nbsp;
              Powered by LangGraph + Google Gemini &nbsp;|&nbsp; Auth0 + Mem0
            </div>
            """,
            unsafe_allow_html=True,
        )

# Show the login page if the user has NOT authenticated yet
_user = st.session_state.get("auth_user") or {}
_is_auth_done = bool(
    _user
    and _user.get("auth_provider") == "auth0"
    and _user.get("is_authenticated", False)
)

if not _is_auth_done:
    render_login_page()
    st.stop()   # ← halt rendering; nothing below executes until user logs in

# ------------------------------------------------------------------
# Active user identity helper — used throughout for user-scoped data
# ------------------------------------------------------------------
def _active_uid() -> str:
    """Return the sanitized user_id of the currently logged-in user.

    Always passes through sanitize_store_namespace so that email-based IDs
    (e.g. google-oauth2 accounts returning the raw email as sub) never
    contain periods or other characters forbidden by LangGraph Store.
    """
    user = st.session_state.get("auth_user")
    if not user:
        return "advocate_guest"
    raw_uid = user.get("user_id", "advocate_guest") or "advocate_guest"
    return sanitize_store_namespace(raw_uid)

# Reset thread state when the logged-in user changes (account switch / re-login)
if st.session_state.get("_last_active_uid") != _active_uid():
    st.session_state["_last_active_uid"] = _active_uid()
    st.session_state.pop("active_thread_id", None)   # clears stale thread from prev user

# --- Initialize Session State (per-user) ---
if "active_thread_id" not in st.session_state or not st.session_state.active_thread_id:
    existing_convs = conv_db.list_conversations(user_id=_active_uid())
    if existing_convs:
        st.session_state.active_thread_id = existing_convs[0]["thread_id"]
    else:
        new_id = str(uuid.uuid4())
        conv_db.create_conversation(new_id, "Welcome to ConstitutionGPT",
                                    engine_mode="both", user_id=_active_uid())
        st.session_state.active_thread_id = new_id

if "delete_target_id" not in st.session_state:
    st.session_state.delete_target_id = None

if "gemini_api_key_input" not in st.session_state:
    st.session_state.gemini_api_key_input = settings.gemini_api_key

if "enable_human_review" not in st.session_state:
    st.session_state.enable_human_review = False

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

if "pending_resume" not in st.session_state:
    st.session_state.pending_resume = None


def switch_conversation(thread_id: str) -> None:
    """Switch active conversation thread and trigger rerun."""
    st.session_state.active_thread_id = thread_id
    st.session_state.delete_target_id = None
    st.session_state.pending_interrupt = None
    st.session_state.pending_resume = None
    st.rerun()


def create_new_conversation(engine_mode: str = "both") -> None:
    """Create and activate a new conversation thread with designated engine mode."""
    new_id = str(uuid.uuid4())
    conv_db.create_conversation(new_id, "New Consultation",
                                engine_mode=engine_mode, user_id=_active_uid())
    st.session_state.active_thread_id = new_id
    st.session_state.delete_target_id = None
    st.session_state.pending_interrupt = None
    st.session_state.pending_resume = None
    st.rerun()


def handle_delete_conversation(thread_id: str) -> None:
    """Delete a conversation from the registry and SQLite checkpointer."""
    conv_db.delete_conversation(thread_id)
    delete_thread_checkpoints(thread_id)

    remaining = conv_db.list_conversations(user_id=_active_uid())
    if remaining:
        st.session_state.active_thread_id = remaining[0]["thread_id"]
    else:
        new_id = str(uuid.uuid4())
        conv_db.create_conversation(new_id, "New Consultation",
                                    engine_mode="both", user_id=_active_uid())
        st.session_state.active_thread_id = new_id

    st.session_state.delete_target_id = None
    st.session_state.pending_interrupt = None
    st.toast("Conversation deleted successfully!", icon="🗑️")
    st.rerun()


def get_thread_messages(thread_id: str, user_id: Optional[str] = None) -> List[BaseMessage]:
    """Retrieve persisted messages for the thread from the SQLite checkpointer, validating ownership."""
    if user_id:
        conv = conv_db.get_conversation(thread_id, user_id=user_id)
        if not conv:
            return []
    graph = get_compiled_graph(checkpointer=checkpointer, store=global_store)
    config = {"configurable": {"thread_id": thread_id}}
    state_tuple = graph.get_state(config)
    if state_tuple and state_tuple.values:
        return state_tuple.values.get("messages", [])
    return []


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚖️ ConstitutionGPT")
    st.caption("10 LangGraph Concepts | PDF + Gemini")

    # --- User Profile & Account Controls ---
    current_user = st.session_state.get("auth_user") or {}
    is_auth = current_user.get("is_authenticated", False)
    auth_provider = current_user.get("auth_provider", "guest")

    col_u_pic, col_u_info = st.columns([0.24, 0.76])
    with col_u_pic:
        pic_url = current_user.get("picture") or "https://api.dicebear.com/7.x/initials/svg?seed=Advocate&backgroundColor=1a73e8&textColor=ffffff"
        st.image(pic_url, width=44)
    with col_u_info:
        st.markdown(
            f"<div style='line-height:1.2; font-weight:700; font-size:0.90rem;'>{current_user.get('name', 'Auth0 User')}</div>"
            "<span style='font-size:0.70rem; background:#e8f0fe; color:#1a73e8; padding:2px 6px; border-radius:10px; font-weight:600;'>🔒 Auth0 Verified</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"`{_active_uid()[:24]}`")

    # Account Actions: Switch Auth0 Account & Log Out
    col_sb_lo, col_sb_sw = st.columns(2)
    with col_sb_lo:
        if st.button("🚪 Log Out", key="btn_logout_sidebar", use_container_width=True):
            st.session_state.auth_user = None
            st.session_state.pop("_login_done", None)
            st.session_state.pop("_auth0_code_processed", None)
            st.session_state.pop("active_thread_id", None)
            st.toast("Logged out. Returned to sign in screen.", icon="👋")
            st.rerun()
    with col_sb_sw:
        _sw_svc = get_auth0_service()
        if _sw_svc.is_configured:
            switch_url = _sw_svc.get_authorize_url(prompt="login")
            st.markdown(
                f"<a href='{switch_url}' target='_self' style='"
                "display:block; background:#f1f3f4; color:#1a73e8; text-align:center;"
                "padding:8px 4px; border-radius:8px; font-weight:600; font-size:0.82rem;"
                "text-decoration:none; border:1px solid #dadce0;'>🔄 Switch Account</a>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("**➕ Start New Chat:**")
    col_nc1, col_nc2 = st.columns(2)
    with col_nc1:
        if st.button("🏛️ Rule-Based", key="btn_new_rule_sb", use_container_width=True, help="New Chat with Rule-Based Engine (Verbatim PDF Grounded)"):
            create_new_conversation(engine_mode="rule_based")
    with col_nc2:
        if st.button("✨ Gemini API", key="btn_new_gemini_sb", use_container_width=True, help="New Chat with Gemini AI (Accessible Plain-English Synthesis)"):
            create_new_conversation(engine_mode="gemini")

    if st.button("⚖️ Dual View (Both Engines)", key="btn_new_both_sb", use_container_width=True, help="New Chat with both engines side-by-side"):
        create_new_conversation(engine_mode="both")

    st.markdown("---")
    st.subheader("💬 Conversations")

    conversations = conv_db.list_conversations(user_id=_active_uid())
    if not conversations:
        st.info("No saved conversations.")
    else:
        for conv in conversations:
            t_id = conv["thread_id"]
            title = conv["title"]
            eng = conv.get("engine_mode", "both")
            eng_icon = "🏛️" if eng == "rule_based" else ("✨" if eng == "gemini" else "⚖️")
            is_active = t_id == st.session_state.active_thread_id

            col_btn, col_del = st.columns([0.82, 0.18])
            with col_btn:
                btn_label = f"📌 {eng_icon} {title}" if is_active else f"{eng_icon} {title}"
                if st.button(
                    btn_label,
                    key=f"conv_btn_{t_id}",
                    use_container_width=True,
                    type="secondary" if not is_active else "primary",
                ):
                    switch_conversation(t_id)

            with col_del:
                if st.button("🗑️", key=f"del_btn_{t_id}", help=f"Delete '{title}'"):
                    handle_delete_conversation(t_id)

        if len(conversations) > 1:
            st.markdown("")
            if st.button("🧹 Clear All Chats", key="clear_all_chats_btn", use_container_width=True):
                for c in conversations:
                    conv_db.delete_conversation(c["thread_id"])
                    delete_thread_checkpoints(c["thread_id"])
                new_id = str(uuid.uuid4())
                conv_db.create_conversation(new_id, "New Consultation",
                                            engine_mode="both", user_id=_active_uid())
                st.session_state.active_thread_id = new_id
                st.toast("All conversations cleared!", icon="🧹")
                st.rerun()

    st.markdown("---")
    # Feature 3: LangGraph Store (Cross-Thread Long-Term Memory)
    with st.expander("⭐ LangGraph Store: Bookmarks & Profile", expanded=False):
        st.markdown("**Cross-Thread Bookmarks**")
        bookmarks = list_bookmarks(user_id=_active_uid())
        if not bookmarks:
            st.caption("No bookmarked articles yet. Click ⭐ Bookmark in any answer to save across conversations.")
        else:
            for b in bookmarks:
                art = b.get("article_number", "")
                title = b.get("title", "")
                col_bm_txt, col_bm_del = st.columns([0.82, 0.18])
                with col_bm_txt:
                    st.markdown(f"**Article {art}** — *{title}*")
                with col_bm_del:
                    if st.button("❌", key=f"del_bm_{art}", help=f"Remove Article {art}"):
                        remove_bookmark(art, user_id=_active_uid())
                        st.rerun()

        st.markdown("---")
        st.markdown("**User Research Preferences** (Stored across threads):")
        current_prefs = get_user_preferences(user_id=_active_uid())
        tone_options = ["Comprehensive", "Concise", "Academic", "Plain English"]
        saved_tone = current_prefs.get("tone", "Comprehensive")
        tone_idx = tone_options.index(saved_tone) if saved_tone in tone_options else 0
        tone_choice = st.selectbox(
            "Explanation Tone",
            tone_options,
            index=tone_idx,
        )
        if tone_choice != current_prefs.get("tone"):
            update_user_preferences({"tone": tone_choice}, user_id=_active_uid())
            st.success("Tone preference saved to Store!")

    # Mem0 Intelligent Memory Engine
    with st.expander("🧠 Mem0 Intelligent Memory", expanded=False):
        uid = _active_uid()
        status_info = mem0_svc.get_status(uid)
        is_cloud = status_info["is_cloud"]

        if is_cloud:
            st.markdown("🟢 **Mem0 Cloud Platform Active**")
        else:
            st.markdown("💾 **Mem0 Local SQLite Engine Active**")

        st.caption(f"Active Identity: `{uid}` ({status_info['total_memories']} memories)")

        # Search User Memories
        mem_query = st.text_input("🔍 Search user memories:", key="input_mem_search", placeholder="e.g. privacy, writ, Article 21")
        if mem_query.strip():
            matched_mems = mem0_svc.search_memories(mem_query.strip(), user_id=uid, limit=4)
            st.markdown(f"**Found {len(matched_mems)} relevant memories:**")
            for m in matched_mems:
                txt = m.get("memory") or m.get("text") or str(m)
                st.markdown(f"• *{txt}*")
            st.markdown("---")

        # Display Stored Memories with Delete buttons
        user_mems = mem0_svc.get_all_memories(user_id=uid, limit=20)
        st.markdown(f"**Stored Memories (`{len(user_mems)}`):**")
        if not user_mems:
            st.caption("No memories recorded yet. Ask constitutional questions or add a custom memory below!")
        else:
            for m in user_mems:
                m_id = m.get("id", "")
                m_text = m.get("memory") or m.get("text") or str(m)
                col_m_text, col_m_del = st.columns([0.82, 0.18])
                with col_m_text:
                    st.markdown(f"• {m_text}")
                with col_m_del:
                    if st.button("🗑️", key=f"del_mem_{m_id}", help="Delete this memory"):
                        mem0_svc.delete_memory(m_id)
                        st.toast("Memory deleted from Mem0!", icon="🗑️")
                        st.rerun()

        st.markdown("---")
        # Add Custom Memory
        st.markdown("**➕ Add Custom Memory / Preference:**")
        new_mem_text = st.text_area(
            "Memory statement:",
            key="input_new_memory_text",
            placeholder="e.g. I am an advocate drafting a PIL on Article 21 right to privacy.",
            height=70,
        )
        if st.button("💾 Save to Mem0", key="btn_save_mem0", use_container_width=True):
            if new_mem_text.strip():
                mem0_svc.add_memory(
                    content=new_mem_text.strip(),
                    user_id=uid,
                    metadata={"source": "manual_ui"},
                )
                st.toast("Memory saved successfully to Mem0!", icon="🧠")
                st.rerun()

        if user_mems:
            st.markdown("")
            if st.button("🧹 Clear All Memories for User", key="btn_clear_all_mem0", use_container_width=True):
                mem0_svc.clear_user_memories(uid)
                st.toast("All memories cleared for this user!", icon="🧹")
                st.rerun()

    # Feature 8: Time Travel & Checkpoint Explorer
    with st.expander("⏳ Time Travel & Checkpoint Explorer", expanded=False):
        active_t = st.session_state.active_thread_id
        snapshots = get_thread_history_snapshots(
            active_t, checkpointer=checkpointer, store=global_store
        )
        if not snapshots:
            st.caption("No checkpoints recorded yet for this thread.")
        else:
            st.markdown(f"**Snapshots in SQLite:** `{len(snapshots)}`")
            snapshot_options = [
                f"Step {s['step_index']}: {s['intent'] or 'start'} ({s['checkpoint_id'][:8]}...)"
                for s in snapshots
            ]
            selected_idx = st.selectbox(
                "Select Checkpoint to Inspect:",
                range(len(snapshot_options)),
                format_func=lambda i: snapshot_options[i],
            )
            selected_snap = snapshots[selected_idx]
            st.markdown(f"**Checkpoint ID:** `{selected_snap['checkpoint_id']}`")
            if selected_snap.get("parent_checkpoint_id"):
                st.caption(f"Parent: `{selected_snap['parent_checkpoint_id'][:8]}...`")
            st.markdown(f"**Next Node:** `{selected_snap['next'] or 'END'}`")
            if selected_snap["question"]:
                st.markdown(f"**Question:** *{selected_snap['question']}*")
            if selected_snap["last_discussed_article"]:
                st.markdown(f"**Last Article:** `Article {selected_snap['last_discussed_article']}`")

            # Time Travel Branching Action
            branch_q = st.text_input(
                "Branch / Rewind with new question:",
                key="branch_q_input",
                placeholder="e.g. Compare it with Article 19",
            )
            col_tt_b1, col_tt_b2 = st.columns(2)
            with col_tt_b1:
                if st.button("⏪ Branch Conversation", use_container_width=True, help="Fork conversation from this checkpoint with the new question above"):
                    if branch_q.strip():
                        with st.spinner("Time Traveling: branching from checkpoint..."):
                            branch_from_checkpoint(
                                active_t,
                                selected_snap["checkpoint_id"],
                                branch_q.strip(),
                                checkpointer=checkpointer,
                                store=global_store,
                            )
                        st.rerun()

            with col_tt_b2:
                if st.button("⏳ Rerun with Gemini", key="btn_sb_tt_gemini", use_container_width=True, help="Rewind to selected checkpoint and rerun using Gemini AI Key"):
                    with st.spinner("Time Traveling: rerunning checkpoint with Gemini AI..."):
                        time_travel_rerun_with_gemini(
                            thread_id=active_t,
                            checkpoint_id=selected_snap["checkpoint_id"],
                            question=branch_q.strip() or selected_snap.get("question") or None,
                            checkpointer=checkpointer,
                            store=global_store,
                            user_id=_active_uid(),
                            rerank_mode="rerank",
                        )
                    st.toast(f"⏳ Time Travel Rerun from checkpoint {selected_snap['checkpoint_id'][:8]}!", icon="✨")
                    st.rerun()

            with st.expander("Inspect Raw Checkpoint State", expanded=False):
                vals_copy = dict(selected_snap["values"])
                if "messages" in vals_copy:
                    vals_copy["messages"] = [
                        f"{m.__class__.__name__}: {str(m.content)[:90]}..."
                        for m in vals_copy["messages"]
                    ]
                st.json(vals_copy)

    # Feature 7: Human-in-the-loop toggle (optional inspection mode)
    st.session_state.enable_human_review = st.checkbox(
        "🛡️ HITL Interrupt Mode (Pause before answering)",
        value=st.session_state.enable_human_review,
        help="When enabled, LangGraph pauses execution before finalization to ask permission. Kept OFF by default so answers generate directly without asking below again.",
    )

    # Feature: RAG Retrieval & Reranker Strategy (with Human-in-the-Loop decision)
    with st.expander("🎯 RAG Retrieval & Reranker Strategy", expanded=True):
        st.markdown("**Second-Stage Retrieval Reranker**")
        rerank_options = [
            "🧑‍⚖️ Human-in-the-Loop (Ask me to Rerank or Not)",
            "🔄 Always Rerank (Gemini AI Semantic Scoring)",
            "⚡ Direct Retrieval (No Reranking)",
        ]
        if "rag_rerank_choice" not in st.session_state:
            st.session_state.rag_rerank_choice = "🧑‍⚖️ Human-in-the-Loop (Ask me to Rerank or Not)"

        current_rerank_idx = (
            rerank_options.index(st.session_state.rag_rerank_choice)
            if st.session_state.rag_rerank_choice in rerank_options
            else 0
        )
        chosen_rerank = st.radio(
            "Reranking Mode:",
            rerank_options,
            index=current_rerank_idx,
            help="Choose whether candidate constitutional articles are reranked via Gemini or whether human in the loop decides per turn.",
        )
        st.session_state.rag_rerank_choice = chosen_rerank

        if chosen_rerank == "🧑‍⚖️ Human-in-the-Loop (Ask me to Rerank or Not)":
            st.caption("ℹ️ Execution will pause before answering so you can inspect candidate articles and decide whether to rerank.")
        elif chosen_rerank == "🔄 Always Rerank (Gemini AI Semantic Scoring)":
            st.caption("ℹ️ Candidate articles from database are automatically scored (0-100) and reordered by Gemini.")
        else:
            st.caption("ℹ️ Pure first-stage direct database retrieval without second-stage reranking.")

    with st.expander("🔑 Gemini API Settings", expanded=False):
        api_key_val = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key_input,
            type="password",
            help="Your Google Gemini API key.",
        )
        if api_key_val != st.session_state.gemini_api_key_input:
            st.session_state.gemini_api_key_input = api_key_val
            settings.gemini_api_key = api_key_val
            st.success("API Key updated!")

        st.caption("Active Models: `gemini-3.5-flash-lite`, `gemini-3.8-flash`")

    with st.expander("⚙️ Auth0 & Mem0 Settings", expanded=False):
        st.markdown("**Auth0 Identity Configuration**")
        a0_domain = st.text_input("Auth0 Domain", value=settings.auth0_domain, placeholder="e.g. dev-xyz.us.auth0.com")
        a0_client_id = st.text_input("Auth0 Client ID", value=settings.auth0_client_id, placeholder="Auth0 App Client ID")
        a0_client_secret = st.text_input("Auth0 Client Secret", value=settings.auth0_client_secret, type="password", placeholder="Auth0 App Client Secret")
        a0_cb_url = st.text_input("Auth0 Callback URL", value=settings.auth0_callback_url)

        st.markdown("---")
        st.markdown("**Mem0 Memory Configuration**")
        m0_key = st.text_input("Mem0 API Key", value=settings.mem0_api_key, type="password", placeholder="m0-... (from app.mem0.ai)")
        st.caption("Leave Mem0 API key blank to run the built-in zero-config SQLite memory store.")

        if st.button("💾 Save Auth0 & Mem0 Settings", key="btn_save_auth_mem_settings", use_container_width=True):
            settings.auth0_domain = a0_domain.strip()
            settings.auth0_client_id = a0_client_id.strip()
            settings.auth0_client_secret = a0_client_secret.strip()
            settings.auth0_callback_url = a0_cb_url.strip()
            settings.mem0_api_key = m0_key.strip()

            # Persist credentials to .env so they survive server restarts
            env_path = Path(__file__).parent / ".env"
            try:
                env_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
            except Exception:
                env_lines = []

            updates = {
                "AUTH0_DOMAIN": a0_domain.strip(),
                "AUTH0_CLIENT_ID": a0_client_id.strip(),
                "AUTH0_CLIENT_SECRET": a0_client_secret.strip(),
                "AUTH0_CALLBACK_URL": a0_cb_url.strip(),
                "MEM0_API_KEY": m0_key.strip(),
            }
            new_lines = []
            handled_keys = set()
            for line in env_lines:
                stripped = line.strip()
                if stripped.startswith("#") or "=" not in stripped:
                    new_lines.append(line)
                    continue
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}")
                    handled_keys.add(key)
                else:
                    new_lines.append(line)
            # Add any keys not already present
            for key, val in updates.items():
                if key not in handled_keys:
                    new_lines.append(f"{key}={val}")
            try:
                env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            except Exception as e:
                st.warning(f"Could not write .env file: {e}")

            # Reset singleton instances so new creds apply immediately
            import app.auth.auth0_service as a0_mod
            import app.memory.mem0_service as m0_mod
            a0_mod._auth0_instance = None
            m0_mod._mem0_instance = None
            st.toast("Auth0 & Mem0 settings saved! ✅", icon="✅")
            st.rerun()

    with st.expander("📖 Step-by-Step Guide: Auth0 & Mem0 Setup", expanded=False):
        st.markdown(
            """
            ### 🔐 Auth0 Setup Steps
            1. **Sign up / Log in** to [auth0.com](https://auth0.com).
            2. Go to **Applications** > **Applications** > **Create Application**.
            3. Choose **Regular Web Applications**.
            4. In the **Settings** tab, configure:
               - **Allowed Callback URLs**: `http://localhost:8501`
               - **Allowed Logout URLs**: `http://localhost:8501`
               - **Allowed Web Origins**: `http://localhost:8501`
            5. Click **Save Changes** at the bottom.
            6. Copy your **Domain**, **Client ID**, and **Client Secret** into the settings above or into `.env`.

            ---
            ### 🧠 Mem0 Setup Steps
            1. **Sign up** at [app.mem0.ai](https://app.mem0.ai).
            2. Navigate to **Settings** > **API Keys** > **Create API Key**.
            3. Copy your key (`m0-...`) and paste it into the Mem0 API Key input above or into `.env`.
            4. *Local Storage*: If no key is set, ConstitutionGPT automatically uses its built-in local SQLite Mem0 engine.

            ---
            ### 🎯 How It Personalizes Consultation
            - Your role (e.g. advocate, student) and constitutional focus are remembered across sessions.
            - When asking about rights or case law, Mem0 injects your prior context directly into Gemini AI.
            """
        )

    with st.expander("🔧 System & Architecture Status", expanded=False):
        status = kb.get_status()
        st.markdown(f"**Knowledge Base:** {'✅ Loaded' if status['is_loaded'] else '❌ Not Loaded'}")
        st.markdown(f"**Articles Indexed:** `{status['total_articles']}`")
        st.markdown(f"**Source Document:** `{status['source_pdf']}`")
        st.markdown(f"**Document Pages:** `{status['total_pages']}`")
        st.markdown(f"**Integrity (SHA-256):** `{status['file_hash'][:16]}...`")
        st.markdown(f"**Persistence:** `SQLite ({settings.checkpoints_db_path.name})`")
        st.markdown(f"**Store:** `InMemoryStore ({len(list_bookmarks(user_id=_active_uid()))} bookmarks)`")
        st.markdown(f"**Fault Tolerance:** `RetryPolicy (3 attempts, backoff 2.0x)`")
        st.markdown(f"**Subgraph:** `RetrievalAndVerificationSubgraph (Active)`")


# ==========================================
# MAIN INTERFACE & POP-UP DIALOGS
# ==========================================

def _instant_dismiss_dialog_js() -> None:
    """Inject instant DOM hiding script so the modal vanishes in 0ms on click."""
    components.html(
        """<script>
        try {
            const pd = window.parent.document;
            pd.querySelectorAll('[data-testid="stDialog"], [role="dialog"], [data-testid="stModal"]').forEach(el => { el.style.display = 'none'; });
            pd.querySelectorAll('[data-testid="stDialogBackdrop"]').forEach(el => { el.style.display = 'none'; });
        } catch(e) {}
        </script>""",
        height=0,
        width=0,
    )


@st.dialog("🎯 Human-in-the-Loop: RAG Ranking Decision", width="large")
def render_hitl_modal(paused_result: Dict[str, Any], active_thread: str, active_engine_mode: str) -> None:
    """Pop-up modal dialog for Human-in-the-Loop decision on ranking & engine approval."""
    raw_q = (
        paused_result.get("raw_question")
        or paused_result.get("prior_user_query")
        or paused_result.get("question", "")
    )
    art_num = paused_result.get("article_number") or paused_result.get("last_discussed_article")
    art_title = paused_result.get("last_discussed_title") or ""
    docs = paused_result.get("retrieved_documents", [])
    if not art_title and docs:
        art_title = docs[0].get("article_title", "")
    part_name = docs[0].get("part", "") if docs else ""
    page_start = docs[0].get("page_start", "") if docs else ""
    cand_list = paused_result.get("candidate_documents") or docs or []

    st.markdown(f"**Question Asked:** *\"{raw_q}\"*")
    if art_num:
        st.markdown(
            f"""<div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px 14px; margin: 8px 0;">
                <div style="color: #166534; font-weight: 700; font-size: 0.90rem;">🏛️ Constitutional Provision Identified</div>
                <div style="color: #15803d; font-size: 0.85rem; margin-top: 2px;"><strong>Article {art_num}</strong> — <em>{art_title}</em> ({part_name}, Page {page_start})</div>
            </div>""",
            unsafe_allow_html=True,
        )

    if len(cand_list) > 1:
        cand_summary = ", ".join([f"Art. {c.get('article_number')}" for c in cand_list[:5]])
        st.caption(f"📚 {len(cand_list)} candidate provisions found in Constitution database: {cand_summary}")

    st.markdown("##### ⚖️ Human Decision: RAG Ranking vs Direct Retrieval")
    st.caption("Click your decision below. The pop-up will automatically close and generate your answer:")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Apply Gemini AI Reranking", key="btn_hitl_apply_rerank", use_container_width=True, type="primary"):
            _instant_dismiss_dialog_js()
            st.session_state.pending_resume = {
                "thread_id": active_thread,
                "engine": active_engine_mode,
                "rerank": "rerank",
            }
            st.session_state.pending_interrupt = None
            st.rerun()
        st.caption("🤖 Score & reorder candidate articles using Gemini AI semantic cross-scoring.")

    with col_btn2:
        if st.button("⚡ Direct Retrieval (Do Not Rank)", key="btn_hitl_no_rerank", use_container_width=True):
            _instant_dismiss_dialog_js()
            st.session_state.pending_resume = {
                "thread_id": active_thread,
                "engine": active_engine_mode,
                "rerank": "no_rerank",
            }
            st.session_state.pending_interrupt = None
            st.rerun()
        st.caption("📄 Fast direct database retrieval directly from local Constitution PDF.")

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    col_foot1, col_foot2 = st.columns([0.7, 0.3])
    with col_foot1:
        mode_text = (
            "✨ With Context (Gemini)" if active_engine_mode == "gemini"
            else ("📜 Without Context (PDF)" if active_engine_mode == "rule_based" else "⚖️ Dual View (Both)")
        )
        st.caption(f"Target Consultation Engine: **{mode_text}**")
    with col_foot2:
        if st.button("❌ Discard", key="btn_hitl_discard", use_container_width=True):
            _instant_dismiss_dialog_js()
            discard_thread_interrupted_turn(active_thread)
            st.session_state.pending_interrupt = None
            st.session_state.pending_resume = None
            st.rerun()


# Modern, clean minimalist header
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 8px; margin-bottom: 12px; border-bottom: 1px solid #e2e8f0;">
        <div>
            <h2 style="margin: 0; font-size: 1.55rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 8px;">
                <span>⚖️ ConstitutionGPT</span>
                <span style="font-size: 0.70rem; font-weight: 600; color: #166534; background: #dcfce7; padding: 2px 8px; border-radius: 12px; border: 1px solid #bbf7d0;">🇮🇳 Official Gazette Grounded</span>
            </h2>
            <div style="font-size: 0.80rem; color: #64748b; margin-top: 2px;">
                Dual Engine: <strong>Verbatim PDF</strong> (Without Context) + <strong>Gemini AI</strong> (With Context) &bull; Auth0 &bull; Mem0
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

active_thread = st.session_state.active_thread_id
conv_info = conv_db.get_conversation(active_thread, user_id=_active_uid())
if not conv_info:
    user_convs = conv_db.list_conversations(user_id=_active_uid())
    if user_convs:
        active_thread = user_convs[0]["thread_id"]
        st.session_state.active_thread_id = active_thread
        conv_info = user_convs[0]
    else:
        new_id = str(uuid.uuid4())
        conv_db.create_conversation(new_id, "Welcome to ConstitutionGPT",
                                    engine_mode="both", user_id=_active_uid())
        st.session_state.active_thread_id = new_id
        active_thread = new_id
        conv_info = conv_db.get_conversation(new_id, user_id=_active_uid())

thread_messages = get_thread_messages(active_thread, user_id=_active_uid())
active_engine_mode = (conv_info.get("engine_mode") if conv_info else "both") or "both"

# Clean Interactive Engine Mode Switcher
col_mode_bar1, col_mode_bar2 = st.columns([0.32, 0.68])
with col_mode_bar1:
    st.markdown(
        """<div style="margin-top: 6px; font-weight: 700; color: #1e293b; font-size: 0.88rem; display: flex; align-items: center; gap: 6px;">
            <span>🎯 Consultation Engine:</span>
        </div>""",
        unsafe_allow_html=True,
    )
with col_mode_bar2:
    mode_options = [
        "⚖️ Dual View (Both Side-by-Side)",
        "✨ With Context (Gemini AI)",
        "📜 Without Context (PDF Verbatim)",
    ]
    curr_mode_idx = (
        1 if active_engine_mode == "gemini"
        else (2 if active_engine_mode == "rule_based" else 0)
    )
    selected_mode_label = st.radio(
        "Active Consultation Engine:",
        mode_options,
        index=curr_mode_idx,
        horizontal=True,
        label_visibility="collapsed",
        key=f"engine_mode_radio_{active_thread}",
    )
    mapped_mode = (
        "gemini" if "With Context" in selected_mode_label
        else ("rule_based" if "Without Context" in selected_mode_label else "both")
    )
    if mapped_mode != active_engine_mode:
        conv_db.update_conversation(active_thread, engine_mode=mapped_mode)
        active_engine_mode = mapped_mode
        st.toast(f"Consultation mode updated to {selected_mode_label}!", icon="🎯")
        st.rerun()

col_rag_bar1, col_rag_bar2 = st.columns([0.32, 0.68])
with col_rag_bar1:
    st.markdown(
        """<div style="margin-top: 6px; font-weight: 700; color: #1e293b; font-size: 0.88rem; display: flex; align-items: center; gap: 6px;">
            <span>🔬 RAG Architecture:</span>
        </div>""",
        unsafe_allow_html=True,
    )
with col_rag_bar2:
    rag_options = [
        "🚀 Advanced RAG (Hybrid RRF + Rerank)",
        "⚡ Normal RAG (Vector Baseline)",
        "⚖️ Compare Pipelines (Side-by-Side)",
    ]
    if "rag_pipeline_mode" not in st.session_state:
        st.session_state.rag_pipeline_mode = "advanced"
    curr_rag_idx = 0 if st.session_state.rag_pipeline_mode == "advanced" else (1 if st.session_state.rag_pipeline_mode == "normal" else 2)
    selected_rag_label = st.radio(
        "RAG Architecture Selection:",
        rag_options,
        index=curr_rag_idx,
        horizontal=True,
        label_visibility="collapsed",
        key=f"rag_mode_radio_{active_thread}",
    )
    if "Advanced RAG" in selected_rag_label:
        st.session_state.rag_pipeline_mode = "advanced"
    elif "Normal RAG" in selected_rag_label:
        st.session_state.rag_pipeline_mode = "normal"
    else:
        st.session_state.rag_pipeline_mode = "compare"

# Trigger Pop-up Modal Dialog if an interrupt is pending
if st.session_state.pending_interrupt:
    render_hitl_modal(
        st.session_state.pending_interrupt.get("result", {}),
        active_thread,
        active_engine_mode,
    )

# If conversation is empty, show sample questions
if not thread_messages and not st.session_state.pending_interrupt:
    st.markdown("<div style='margin-top: 14px; margin-bottom: 8px; font-weight: 600; color: #64748b; font-size: 0.88rem;'>💡 Select a sample question or type your constitutional query below:</div>", unsafe_allow_html=True)

    suggested_questions = [
        "What is Article 21?",
        "Explain Article 14 in simple English",
        "What does Article 32 say?",
        "What are the Fundamental Rights?",
        "Compare Article 14 and Article 21",
        "What is Article 51A?",
    ]

    cols = st.columns(3)
    for idx, sq in enumerate(suggested_questions):
        with cols[idx % 3]:
            if st.button(
                f"⚖️ {sq}", key=f"sugg_{idx}", use_container_width=True
            ):
                st.session_state.pending_query = sq
                st.rerun()

# Render Past Conversation Messages from LangGraph SQLite State
is_last_and_streaming = (
    st.session_state.get("just_streamed_thread") == active_thread
)

for msg_idx, msg in enumerate(thread_messages):
    is_last_message = (msg_idx == len(thread_messages) - 1)
    should_stream = is_last_and_streaming and is_last_message

    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**{msg.content}**")
    elif isinstance(msg, AIMessage):
        choice = msg.additional_kwargs.get("selected_engine", "both")
        rule_content = msg.additional_kwargs.get("rule_answer") or msg.content
        gemini_content = msg.additional_kwargs.get("gemini_answer", "")

        src_info = extract_source_info(msg)
        page_label = src_info.get("page_str", "Page 42")
        src_doc = src_info.get("source_pdf", "constitution.pdf")

        is_out = src_info.get("response_type") in ["out_of_scope", "uncertain"] or src_info.get("intent") in ["out_of_scope", "uncertain"]
        r_badge = "⚖️ Domain Restricted" if is_out else f"📄 {src_doc} • {page_label}"
        r_bg = "#fed7aa" if is_out else "#e8eaed"
        r_color = "#92400e" if is_out else "#1a73e8"

        g_badge = "⚖️ Domain Restricted" if is_out else f"Grounded in {page_label}"
        g_bg = "#fed7aa" if is_out else "#ede7f6"
        g_color = "#92400e" if is_out else "#6a1b9a"

        with st.chat_message("assistant", avatar="⚖️"):
            is_gem_err = is_gemini_synthesis_error(gemini_content, src_info.get("gemini_diagnostic"))
            gem_diag = src_info.get("gemini_diagnostic") or {}
            err_badge = gem_diag.get("error_type") or "Unavailable"
            if gem_diag.get("status_code"):
                err_badge = f"{err_badge} ({gem_diag['status_code']})"

            if is_gem_err:
                gem_header_html = f"""
                <div style="background-color: #fffbeb; padding: 6px 12px; border-radius: 6px; font-weight: 700; margin-bottom: 8px; color: #92400e; display: flex; justify-content: space-between; align-items: center; border-left: 3px solid #f59e0b;">
                    <span>⚠️ Gemini AI (Synthesis Unavailable)</span>
                    <span style="font-size: 0.72rem; background: #fef3c7; color: #b45309; padding: 2px 7px; border-radius: 6px; border: 1px solid #fde68a;">⚠️ {err_badge}</span>
                </div>
                """
                gem_footer_type = "error"
            else:
                gem_header_html = f"""
                <div style="background-color: #faf5ff; padding: 6px 12px; border-radius: 6px; font-weight: 700; margin-bottom: 8px; color: #6b21a8; display: flex; justify-content: space-between; align-items: center; border-left: 3px solid #9333ea;">
                    <span>✨ With Context (Gemini AI Synthesis)</span>
                    <span style="font-size: 0.72rem; background: {g_bg}; color: {g_color}; padding: 2px 7px; border-radius: 6px;">{g_badge}</span>
                </div>
                """
                gem_footer_type = "ai"

            if choice == "both":
                col_rule, col_gemini = st.columns(2)
                with col_rule:
                    st.markdown(
                        f"""
                        <div style="background-color: #f1f5f9; padding: 6px 12px; border-radius: 6px; font-weight: 700; margin-bottom: 8px; color: #1e293b; display: flex; justify-content: space-between; align-items: center; border-left: 3px solid #0284c7;">
                            <span>📜 Without Context (PDF Verbatim)</span>
                            <span style="font-size: 0.72rem; background: {r_bg}; color: {r_color}; padding: 2px 7px; border-radius: 6px;">{r_badge}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if should_stream:
                        st.write_stream(stream_text_word_by_word(rule_content))
                    else:
                        st.markdown(rule_content)
                    render_engine_footer("pdf")

                with col_gemini:
                    st.markdown(gem_header_html, unsafe_allow_html=True)
                    if gemini_content:
                        if should_stream and not is_gem_err:
                            st.write_stream(stream_text_word_by_word(gemini_content))
                        else:
                            st.markdown(gemini_content)
                    else:
                        st.info("Gemini context synthesis is not yet generated for this turn.")
                    render_engine_footer(gem_footer_type)

            elif choice == "gemini":
                st.markdown(gem_header_html, unsafe_allow_html=True)
                if should_stream and not is_gem_err:
                    st.write_stream(stream_text_word_by_word(gemini_content or msg.content))
                else:
                    st.markdown(gemini_content or msg.content)
                render_engine_footer(gem_footer_type)

            else:  # "rule_based"
                st.markdown(
                    f"""
                    <div style="background-color: #f1f5f9; padding: 6px 12px; border-radius: 6px; font-weight: 700; margin-bottom: 8px; color: #1e293b; display: flex; justify-content: space-between; align-items: center; border-left: 3px solid #0284c7;">
                        <span>📜 Rule-Based Engine (Verbatim PDF Grounded)</span>
                        <span style="font-size: 0.72rem; background: {r_bg}; color: {r_color}; padding: 2px 7px; border-radius: 6px;">{r_badge}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if should_stream:
                    st.write_stream(stream_text_word_by_word(rule_content))
                else:
                    st.markdown(rule_content)
                render_engine_footer("pdf")

            # Clean full-width Provenance Bar
            render_message_provenance_bar(src_info)

            # Turn Actions: Bookmark & Time Travel Rerun with Gemini
            col_act_bm, col_act_tt = st.columns([0.48, 0.52])
            turn_art = src_info.get("article_number")
            turn_title = src_info.get("article_title") or (f"Article {turn_art}" if turn_art else "Constitutional Analysis")

            # Extract the prompt question for this turn
            turn_question = ""
            for prev_i in range(msg_idx - 1, -1, -1):
                if isinstance(thread_messages[prev_i], HumanMessage):
                    turn_question = thread_messages[prev_i].content
                    break
            if not turn_question:
                turn_question = msg.additional_kwargs.get("prior_user_query", "")

            with col_act_bm:
                if turn_art:
                    if is_article_bookmarked(turn_art, user_id=_active_uid()):
                        st.caption(f"⭐ *Article {turn_art} Bookmarked*")
                    else:
                        if st.button(f"⭐ Bookmark Art. {turn_art}", key=f"bm_hist_{msg_idx}"):
                            bookmark_article(
                                turn_art,
                                turn_title,
                                notes=turn_question,
                                user_id=_active_uid(),
                            )
                            st.toast(f"Saved Article {turn_art} to Store!", icon="⭐")
                            st.rerun()

            with col_act_tt:
                if turn_question:
                    if st.button("⏳ Rerun with Gemini (Time Travel)", key=f"tt_hist_{msg_idx}", help="LangGraph Time Travel: Rewind conversation to this turn and synthesize answer with Gemini AI"):
                        with st.spinner("⏳ Time Travel: Rewinding checkpoint & rerunning with Gemini AI..."):
                            time_travel_rerun_with_gemini(
                                thread_id=active_thread,
                                question=turn_question,
                                checkpointer=checkpointer,
                                store=global_store,
                                user_id=_active_uid(),
                                rerank_mode="rerank",
                            )
                        st.session_state.just_streamed_thread = active_thread
                        st.session_state.just_streamed_choice = "gemini"
                        conv_db.update_conversation(active_thread, engine_mode="gemini")
                        st.toast("⏳ Time Travel Complete: Rerun with Gemini AI!", icon="✨")
                        st.rerun()

if is_last_and_streaming:
    st.session_state.just_streamed_thread = None

# Handle Human-in-the-Loop decision execution (runs in chat history container after modal closes)
if st.session_state.get("pending_resume"):
    resume_data = st.session_state.pop("pending_resume")
    if resume_data.get("thread_id") == active_thread:
        res_engine = resume_data.get("engine", active_engine_mode)
        res_rerank = resume_data.get("rerank", "rerank")
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("Applying ranking decision and synthesizing constitutional response..."):
                res_output = resume_constitution_chat(
                    active_thread,
                    checkpointer=checkpointer,
                    store=global_store,
                    selected_engine=res_engine,
                    rerank_choice=res_rerank,
                )
        conv_db.update_conversation(active_thread, engine_mode=res_engine)
        st.session_state.just_streamed_thread = active_thread
        st.session_state.just_streamed_choice = res_engine
        st.toast(
            "Gemini AI reranking applied!" if res_rerank == "rerank" else "Direct retrieval applied!",
            icon="✅",
        )
        st.rerun()

# --- User Input Handling with Event Streaming & Word-by-Word Output ---
chat_disabled = bool(st.session_state.pending_interrupt)
chat_placeholder = (
    "Please grant permission on the pending prompt above..."
    if chat_disabled
    else "Ask about an article, right, duty, follow-up, or comparison..."
)
user_query = st.chat_input(chat_placeholder, disabled=chat_disabled)
pending_query = st.session_state.pop("pending_query", None)
active_query = user_query or pending_query

if active_query and active_query.strip():
    clean_query = active_query.strip()

    # Update conversation title from first user query if default
    conv_info = conv_db.get_conversation(active_thread)
    if (
        conv_info
        and conv_info["title"] in ["New Consultation", "Welcome to ConstitutionGPT"]
    ):
        new_title = generate_title_from_query(clean_query)
        conv_db.update_conversation(active_thread, title=new_title)

    # Immediately display user message in chat
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**{clean_query}**")

    # Execute LangGraph turn with ChatGPT-style dynamic step progression
    chosen_rerank_pref = st.session_state.get("rag_rerank_choice", "🧑‍⚖️ Human-in-the-Loop (Ask me to Rerank or Not)")
    is_hitl_rerank = (chosen_rerank_pref == "🧑‍⚖️ Human-in-the-Loop (Ask me to Rerank or Not)")
    eff_enable_interrupt = bool(st.session_state.enable_human_review or is_hitl_rerank)
    eff_rerank_mode = "hitl_decide" if is_hitl_rerank else ("rerank" if "Always Rerank" in chosen_rerank_pref else "no_rerank")

    with st.chat_message("assistant", avatar="⚖️"):
        result = execute_query_chatgpt_style(
            clean_query,
            active_thread,
            selected_engine=active_engine_mode,
            enable_interrupt=eff_enable_interrupt,
            rerank_mode=eff_rerank_mode,
        )

        if result.get("interrupted"):
            st.session_state.pending_interrupt = {
                "thread_id": active_thread,
                "result": result,
            }
            st.rerun()
        else:
            chosen_engine = result.get("selected_engine", active_engine_mode) or active_engine_mode or "both"
            rule_ans = result.get("answer", "")
            gemini_ans = result.get("gemini_answer", "")
            last_art = result.get("last_discussed_article")

            src_info = extract_source_info(result)
            page_label = src_info.get("page_str", "Page 42")
            src_doc = src_info.get("source_pdf", "constitution.pdf")

            is_out = src_info.get("response_type") in ["out_of_scope", "uncertain"] or src_info.get("intent") in ["out_of_scope", "uncertain"]
            r_badge = "⚖️ Domain Restricted" if is_out else f"📄 {src_doc} • {page_label}"
            r_bg = "#fed7aa" if is_out else "#e8eaed"
            r_color = "#92400e" if is_out else "#1a73e8"

            g_badge = "⚖️ Domain Restricted" if is_out else f"Grounded in {page_label}"
            g_bg = "#fed7aa" if is_out else "#ede7f6"
            g_color = "#92400e" if is_out else "#6a1b9a"

            is_gem_err = is_gemini_synthesis_error(gemini_ans, src_info.get("gemini_diagnostic"))
            gem_diag = src_info.get("gemini_diagnostic") or {}
            err_badge = gem_diag.get("error_type") or "Unavailable"
            if gem_diag.get("status_code"):
                err_badge = f"{err_badge} ({gem_diag['status_code']})"

            if is_gem_err:
                gem_header_html = f"""
                <div style="background-color: #fffbeb; padding: 6px 12px; border-radius: 6px; font-weight: 700; margin-bottom: 8px; color: #92400e; display: flex; justify-content: space-between; align-items: center; border-left: 3px solid #f59e0b;">
                    <span>⚠️ Gemini AI (Synthesis Unavailable)</span>
                    <span style="font-size: 0.72rem; background: #fef3c7; color: #b45309; padding: 2px 7px; border-radius: 6px; border: 1px solid #fde68a;">⚠️ {err_badge}</span>
                </div>
                """
                gem_footer_type = "error"
            else:
                gem_header_html = f"""
                <div style="background-color: #faf5ff; padding: 6px 12px; border-radius: 6px; font-weight: 700; margin-bottom: 8px; color: #6b21a8; display: flex; justify-content: space-between; align-items: center; border-left: 3px solid #9333ea;">
                    <span>✨ With Context (Gemini AI Synthesis)</span>
                    <span style="font-size: 0.72rem; background: {g_bg}; color: {g_color}; padding: 2px 7px; border-radius: 6px;">{g_badge}</span>
                </div>
                """
                gem_footer_type = "ai"

            if chosen_engine == "both":
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown(
                        f"""
                        <div style="background-color: #f1f5f9; padding: 6px 12px; border-radius: 6px; font-weight: 700; margin-bottom: 8px; color: #1e293b; display: flex; justify-content: space-between; align-items: center; border-left: 3px solid #0284c7;">
                            <span>📜 Without Context (PDF Verbatim)</span>
                            <span style="font-size: 0.72rem; background: {r_bg}; color: {r_color}; padding: 2px 7px; border-radius: 6px;">{r_badge}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write_stream(stream_text_word_by_word(rule_ans))
                    render_engine_footer("pdf")

                with col_r:
                    st.markdown(gem_header_html, unsafe_allow_html=True)
                    if not is_gem_err:
                        st.write_stream(stream_text_word_by_word(gemini_ans))
                    else:
                        st.markdown(gemini_ans)
                    render_engine_footer(gem_footer_type)

            elif chosen_engine == "gemini":
                st.markdown(gem_header_html, unsafe_allow_html=True)
                if not is_gem_err:
                    st.write_stream(stream_text_word_by_word(gemini_ans))
                else:
                    st.markdown(gemini_ans)
                render_engine_footer(gem_footer_type)

            else:  # "rule_based"
                st.markdown(
                    f"""
                    <div style="background-color: #f1f5f9; padding: 6px 12px; border-radius: 6px; font-weight: 700; margin-bottom: 8px; color: #1e293b; display: flex; justify-content: space-between; align-items: center; border-left: 3px solid #0284c7;">
                        <span>📜 Rule-Based Engine (Verbatim PDF Grounded)</span>
                        <span style="font-size: 0.72rem; background: {r_bg}; color: {r_color}; padding: 2px 7px; border-radius: 6px;">{r_badge}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write_stream(stream_text_word_by_word(rule_ans))
                render_engine_footer("pdf")

            # Clean full-width Provenance Bar
            render_message_provenance_bar(src_info)

            # Feature 3 & 8: Actions for current turn: Bookmark & Time Travel Rerun with Gemini
            col_bm, col_tt_rerun = st.columns([0.48, 0.52])
            with col_bm:
                if last_art:
                    if is_article_bookmarked(last_art, user_id=_active_uid()):
                        st.caption(f"⭐ *Article {last_art} Bookmarked in Store*")
                    else:
                        if st.button(f"⭐ Bookmark Article {last_art}", key=f"bm_btn_curr_{last_art}"):
                            bookmark_article(
                                last_art,
                                result.get("last_discussed_title", f"Article {last_art}"),
                                notes=clean_query,
                                user_id=_active_uid(),
                            )
                            st.toast(f"Saved Article {last_art} to cross-thread Store!", icon="⭐")
                            st.rerun()

            with col_tt_rerun:
                if st.button("⏳ Rerun with Gemini (Time Travel)", key="btn_tt_rerun_current", help="LangGraph Time Travel: Rewind to checkpoint and synthesize answer with Gemini AI"):
                    with st.spinner("⏳ Time Travel: Rewinding checkpoint & synthesizing with Gemini AI..."):
                        time_travel_rerun_with_gemini(
                            thread_id=active_thread,
                            question=clean_query,
                            checkpointer=checkpointer,
                            store=global_store,
                            user_id=_active_uid(),
                            rerank_mode="rerank",
                        )
                        st.session_state.just_streamed_thread = active_thread
                        st.session_state.just_streamed_choice = "gemini"
                        conv_db.update_conversation(active_thread, engine_mode="gemini")
                        st.toast("⏳ Time Travel Complete: Regenerated with Gemini AI!", icon="✨")
                        st.rerun()

    # Update conversation last modified timestamp
    conv_db.update_conversation(active_thread)
