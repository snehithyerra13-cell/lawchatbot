"""LangGraph workflow assembly and execution pipeline for ConstitutionGPT.

Covers all 10 LangGraph architectural pillars:
1. Persistence (State durability on disk)
2. Checkpointers (SQLite Saver managing snapshot trees)
3. Stores (Cross-thread shared memory & bookmarks via InMemoryStore)
4. Fault tolerance (RetryPolicy with exponential backoff on nodes)
5. Event streaming (stream_mode='updates' live node progress)
6. Streaming (Output word-by-word streaming generator)
7. Interrupts (Human-in-the-loop review before state finalization)
8. Time travel (Snapshot history inspection & checkpoint branching)
9. Memory (In-thread short-term messages + cross-thread store profile)
10. Subgraphs (Nested RetrievalAndVerificationSubgraph)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Generator, List, Optional, Sequence

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.graph.nodes import (
    check_retrieval,
    classify_question,
    finalize_response,
    generate_answer,
    handle_general_query,
    handle_uncertainty,
    resolve_context,
    retrieve_constitution,
    rewrite_query,
    synthesize_gemini,
)
from app.graph.routing import (
    route_classification,
    route_context,
    route_retrieval,
)
from app.graph.state import ConstitutionState
from app.persistence.checkpointer import get_sqlite_checkpointer
from app.persistence.store import get_global_store

logger = logging.getLogger(__name__)


def build_constitution_graph() -> StateGraph:
    """Construct the StateGraph defining the constitutional reasoning flow with RetryPolicies."""
    builder = StateGraph(ConstitutionState)

    # Fault-tolerance retry policies
    network_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=0.5,
        backoff_factor=2.0,
    )
    retrieval_retry = RetryPolicy(
        max_attempts=2,
        initial_interval=0.2,
    )

    # 1. Add All Nodes with fault-tolerance policies
    builder.add_node("rewrite_query", rewrite_query, retry_policy=network_retry)
    builder.add_node("classify_question", classify_question)
    builder.add_node("resolve_context", resolve_context)
    builder.add_node("retrieve_constitution", retrieve_constitution, retry_policy=retrieval_retry)
    builder.add_node("check_retrieval", check_retrieval)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("handle_general_query", handle_general_query)
    builder.add_node("handle_uncertainty", handle_uncertainty)
    builder.add_node("synthesize_gemini", synthesize_gemini, retry_policy=network_retry)
    builder.add_node("finalize_response", finalize_response)

    # 2. Add Edges & Conditional Routing
    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "classify_question")

    builder.add_conditional_edges(
        "classify_question",
        route_classification,
        {
            "handle_general_query": "handle_general_query",
            "retrieve_constitution": "retrieve_constitution",
            "resolve_context": "resolve_context",
            "handle_uncertainty": "handle_uncertainty",
        },
    )

    builder.add_conditional_edges(
        "resolve_context",
        route_context,
        {
            "handle_uncertainty": "handle_uncertainty",
            "retrieve_constitution": "retrieve_constitution",
            "finalize_response": "synthesize_gemini",
        },
    )

    builder.add_edge("retrieve_constitution", "check_retrieval")

    builder.add_conditional_edges(
        "check_retrieval",
        route_retrieval,
        {
            "generate_answer": "generate_answer",
            "handle_uncertainty": "handle_uncertainty",
        },
    )

    builder.add_edge("generate_answer", "synthesize_gemini")
    builder.add_edge("handle_general_query", "synthesize_gemini")
    builder.add_edge("handle_uncertainty", "synthesize_gemini")
    builder.add_edge("synthesize_gemini", "finalize_response")
    builder.add_edge("finalize_response", END)

    return builder


_graph_cache: Dict[tuple, Any] = {}


def get_compiled_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[Any] = None,
    interrupt_nodes: Optional[Sequence[str]] = None,
):
    """Retrieve or compile the LangGraph workflow with SQLite checkpointer and Store."""
    saver = checkpointer or get_sqlite_checkpointer()
    st = store or get_global_store()
    cache_key = (id(saver), id(st), tuple(interrupt_nodes or ()))
    if cache_key not in _graph_cache:
        builder = build_constitution_graph()
        _graph_cache[cache_key] = builder.compile(
            checkpointer=saver,
            store=st,
            interrupt_before=list(interrupt_nodes) if interrupt_nodes else None,
        )
    return _graph_cache[cache_key]


def run_constitution_chat(
    question: str,
    thread_id: str,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[Any] = None,
    enable_interrupt: bool = False,
    checkpoint_id: Optional[str] = None,
    selected_engine: Optional[str] = "both",
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a turn in the conversation thread and return updated state.
    
    Supports:
    - Interrupts: Pauses before finalize_response if enable_interrupt is True
    - Time Travel: Can execute or fork from an optional checkpoint_id
    - Persistence: Persists snapshots in SQLite checkpointer
    - Stores: Updates cross-thread knowledge in Store
    - Performance: Tracks precise execution time in seconds
    - Identity & Mem0 Memory: Associates execution turn with user_id
    """
    t_start = time.perf_counter()
    interrupt_nodes = ["finalize_response"] if enable_interrupt else None
    graph = get_compiled_graph(
        checkpointer=checkpointer, store=store, interrupt_nodes=interrupt_nodes
    )

    human_msg = HumanMessage(content=question)
    input_state: Dict[str, Any] = {
        "messages": [human_msg],
        "question": question,
        "selected_engine": selected_engine or "both",
        "user_id": user_id or "default_user",
    }

    config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id

    result = graph.invoke(input_state, config=config)

    duration = round(time.perf_counter() - t_start, 3)
    result["execution_time_seconds"] = duration

    # Attach timing to latest AIMessage
    msgs = result.get("messages", [])
    if msgs and hasattr(msgs[-1], "additional_kwargs"):
        msgs[-1].additional_kwargs["execution_time_seconds"] = duration

    # Check if execution paused at an interrupt breakpoint
    if enable_interrupt:
        state_snap = graph.get_state(config)
        if state_snap and state_snap.next:
            result["interrupted"] = True
            result["next_node"] = list(state_snap.next)

    return result


def run_constitution_chat_stream(
    question: str,
    thread_id: str,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[Any] = None,
    enable_interrupt: bool = False,
    checkpoint_id: Optional[str] = None,
    selected_engine: Optional[str] = "both",
    user_id: Optional[str] = None,
) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
    """Execute a turn yielding LangGraph node events in real-time (Event Streaming).
    
    Yields exact step-by-step progress events:
      {"type": "node_update", "node": node_name, "label": ..., "icon": ..., "summary": ..., "detail": ...}
    Finally, yields the complete result dictionary with execution timing:
      {"type": "complete", "result": final_state}
    """
    t_start = time.perf_counter()
    interrupt_nodes = ["finalize_response"] if enable_interrupt else None
    graph = get_compiled_graph(
        checkpointer=checkpointer, store=store, interrupt_nodes=interrupt_nodes
    )

    human_msg = HumanMessage(content=question)
    input_state: Dict[str, Any] = {
        "messages": [human_msg],
        "question": question,
        "selected_engine": selected_engine or "both",
        "user_id": user_id or "default_user",
    }

    config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id

    node_descriptors = {
        "rewrite_query": {
            "icon": "🧠",
            "label": "LLM Query Analysis & Disambiguation",
            "default_summary": "Analyzing query and mapping to canonical constitutional articles",
        },
        "classify_question": {
            "icon": "⚖️",
            "label": "Intent Classification & Legal Heuristics",
            "default_summary": "Detecting article lookup, comparison, or thematic topic intent",
        },
        "resolve_context": {
            "icon": "📜",
            "label": "Conversation Context & History Grounding",
            "default_summary": "Reconciling prior queries, cross-thread store, and discussed articles",
        },
        "retrieve_constitution": {
            "icon": "📖",
            "label": "Fetching Verbatim Text from constitution.pdf",
            "default_summary": "Querying local Constitution PDF index and extracting clauses",
        },
        "check_retrieval": {
            "icon": "🛡️",
            "label": "Citation & Integrity Verification",
            "default_summary": "Validating document SHA-256 hash and official page references",
        },
        "generate_answer": {
            "icon": "🏛️",
            "label": "Deterministic Rule-Based Synthesis",
            "default_summary": "Structuring verbatim constitutional text and legal principles",
        },
        "handle_general_query": {
            "icon": "👋",
            "label": "Constitutional Guidance & System Assistance",
            "default_summary": "Providing helpful overview of available constitutional capabilities",
        },
        "handle_uncertainty": {
            "icon": "🔍",
            "label": "Clarification & Exploration Guidance",
            "default_summary": "Advising precise query syntax and available articles",
        },
        "synthesize_gemini": {
            "icon": "✨",
            "label": "Gemini AI Constitutional Synthesis",
            "default_summary": "Synthesizing plain English legal principles and case doctrines",
        },
        "finalize_response": {
            "icon": "✅",
            "label": "Human Decision & State Checkpoint",
            "default_summary": "Committing verified answer and provenance to SQLite checkpointer",
        },
    }

    # Stream graph updates
    for chunk in graph.stream(input_state, config=config, stream_mode="updates"):
        if isinstance(chunk, dict):
            for node_name, node_update in chunk.items():
                meta = node_descriptors.get(
                    node_name,
                    {"icon": "⚙️", "label": f"Node: {node_name}", "default_summary": "Processing step"},
                )
                summary = meta["default_summary"]
                detail = ""

                if isinstance(node_update, dict):
                    if node_name == "retrieve_constitution":
                        docs = node_update.get("retrieved_documents", [])
                        if docs:
                            d0 = docs[0]
                            art = d0.get("article_number", "")
                            page = d0.get("page_start", "")
                            part = d0.get("part", "")
                            src_pdf = d0.get("source_pdf", "constitution.pdf")
                            summary = f"Fetched {len(docs)} document(s) from {src_pdf} (Article {art}, Page {page}, {part})"
                            detail = f"Extracted {len(d0.get('constitutional_text', ''))} chars of verbatim text"
                    elif node_name == "rewrite_query":
                        meta_info = node_update.get("query_rewriter_metadata") or {}
                        arts = meta_info.get("articles") or []
                        rewritten = node_update.get("rewritten_question") or node_update.get("question")
                        if arts:
                            summary = f"Canonical target resolved: Article {', '.join(arts)}"
                        elif rewritten:
                            summary = f"Rewritten to: '{rewritten[:60]}...'" if len(rewritten) > 60 else f"Rewritten: '{rewritten}'"
                    elif node_name == "classify_question":
                        intent = node_update.get("intent", "")
                        art_num = node_update.get("article_number")
                        if intent == "out_of_scope":
                            summary = "Non-constitutional query detected; enforcing scope restriction"
                        else:
                            summary = f"Intent identified: `{intent}`" + (f" (Target: Article {art_num})" if art_num else "")
                    elif node_name == "handle_uncertainty":
                        if node_update.get("response_type") == "out_of_scope":
                            meta = {"icon": "🛡️", "label": "Scope Enforcement & Domain Guardrail", "default_summary": "Enforcing scope restriction"}
                            summary = "Out-of-scope query rejected; enforcing strict Constitution-only boundary"
                    elif node_name == "check_retrieval":
                        summary = "Verified against constitution.pdf (SHA-256 integrity checked)"
                    elif node_name == "generate_answer":
                        summary = "Rule-based verbatim answer drafted with official page citations"
                    elif node_name == "synthesize_gemini":
                        summary = "Gemini AI response synthesized with case doctrine & plain English"

                yield {
                    "type": "node_update",
                    "node": node_name,
                    "label": meta["label"],
                    "icon": meta["icon"],
                    "summary": summary,
                    "detail": detail,
                }

    duration = round(time.perf_counter() - t_start, 3)

    # Retrieve final state from checkpointer
    state_snap = graph.get_state(config)
    final_values = state_snap.values if state_snap else {}
    final_values["execution_time_seconds"] = duration

    # Attach timing to latest AIMessage
    msgs = final_values.get("messages", [])
    if msgs and hasattr(msgs[-1], "additional_kwargs"):
        msgs[-1].additional_kwargs["execution_time_seconds"] = duration

    if enable_interrupt and state_snap and state_snap.next:
        final_values["interrupted"] = True
        final_values["next_node"] = list(state_snap.next)

    yield {"type": "complete", "result": final_values}
    return final_values


def resume_constitution_chat(
    thread_id: str,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[Any] = None,
    selected_engine: Optional[str] = "both",
) -> Dict[str, Any]:
    """Resume execution of an interrupted thread with human engine choice (Human-in-the-Loop)."""
    t_start = time.perf_counter()
    graph = get_compiled_graph(
        checkpointer=checkpointer, store=store, interrupt_nodes=["finalize_response"]
    )
    config = {"configurable": {"thread_id": thread_id}}
    if selected_engine:
        graph.update_state(config, {"selected_engine": selected_engine})
    result = graph.invoke(None, config=config)

    duration = round(time.perf_counter() - t_start, 3)
    if isinstance(result, dict):
        result["execution_time_seconds"] = duration
        msgs = result.get("messages", [])
        if msgs and hasattr(msgs[-1], "additional_kwargs"):
            msgs[-1].additional_kwargs["execution_time_seconds"] = duration

    return result


def branch_from_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    new_question: str,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[Any] = None,
) -> Dict[str, Any]:
    """Time Travel: Fork/branch conversation from a specific historical checkpoint."""
    return run_constitution_chat(
        question=new_question,
        thread_id=thread_id,
        checkpointer=checkpointer,
        store=store,
        checkpoint_id=checkpoint_id,
    )


def get_thread_history_snapshots(
    thread_id: str,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Retrieve past LangGraph checkpoints for Time Travel inspection."""
    graph = get_compiled_graph(checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": thread_id}}
    snapshots: List[Dict[str, Any]] = []

    try:
        history_iter = graph.get_state_history(config)
        for idx, snap in enumerate(history_iter):
            cfg = snap.config.get("configurable", {})
            c_id = cfg.get("checkpoint_id", "")
            vals = snap.values or {}
            snapshots.append(
                {
                    "step_index": idx,
                    "checkpoint_id": c_id,
                    "parent_checkpoint_id": cfg.get("parent_checkpoint_id", None),
                    "created_at": snap.created_at if hasattr(snap, "created_at") else None,
                    "next": list(snap.next) if snap.next else [],
                    "question": vals.get("question", ""),
                    "intent": vals.get("intent", ""),
                    "article_number": vals.get("article_number", ""),
                    "response_type": vals.get("response_type", ""),
                    "last_discussed_article": vals.get("last_discussed_article", ""),
                    "total_messages": len(vals.get("messages", [])),
                    "values": vals,
                }
            )
    except Exception as e:
        logger.warning(f"Error fetching state history for {thread_id}: {e}")

    return snapshots


def get_state_at_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[Any] = None,
) -> Dict[str, Any]:
    """Inspect state at a specific past checkpoint (Time Travel)."""
    graph = get_compiled_graph(checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    snap = graph.get_state(config)
    return snap.values if snap else {}
