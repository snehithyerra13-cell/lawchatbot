"""Tests for Human-in-the-Loop (HITL) Engine Decision in LangGraph.

Validates that human can inspect candidate outputs from both the Rule-Based
Engine and Gemini AI, and decide which output to commit to memory (Rule-Based,
Gemini AI, or Both), or discard the turn.
"""

import uuid
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.graph.workflow import (
    get_compiled_graph,
    get_global_store,
    resume_constitution_chat,
    run_constitution_chat,
)
from app.persistence.checkpointer import (
    delete_thread_checkpoints,
    discard_thread_interrupted_turn,
    get_sqlite_checkpointer,
)


def test_hitl_pause_produces_both_candidates():
    """Verify that pausing before finalize_response prepares both candidates."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = f"test-hitl-{uuid.uuid4()}"

    result = run_constitution_chat(
        question="What is Article 21?",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=True,
    )

    assert result.get("interrupted") is True
    assert "finalize_response" in result.get("next_node", [])
    # Candidate 1: Rule-Based output
    assert result.get("answer") is not None
    assert "Article 21" in result.get("answer", "")
    # Candidate 2: Gemini AI output
    assert result.get("gemini_answer") is not None
    assert len(result.get("gemini_answer", "")) > 10


def test_hitl_human_selects_rule_based_only():
    """Verify that human selecting rule_based commits rule output as AIMessage content."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = f"test-hitl-{uuid.uuid4()}"

    res = run_constitution_chat(
        question="What is Article 14?",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=True,
    )
    assert res.get("interrupted") is True
    rule_candidate = res.get("answer")

    final_res = resume_constitution_chat(
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        selected_engine="rule_based",
    )

    assert final_res.get("selected_engine") == "rule_based"
    messages = final_res.get("messages", [])
    assert len(messages) >= 2
    ai_msg = messages[-1]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.additional_kwargs.get("selected_engine") == "rule_based"
    assert ai_msg.content == rule_candidate


def test_hitl_human_selects_gemini_only():
    """Verify that human selecting gemini commits Gemini synthesis as AIMessage content."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = f"test-hitl-{uuid.uuid4()}"

    res = run_constitution_chat(
        question="What is Article 19?",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=True,
    )
    assert res.get("interrupted") is True
    gemini_candidate = res.get("gemini_answer")

    final_res = resume_constitution_chat(
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        selected_engine="gemini",
    )

    assert final_res.get("selected_engine") == "gemini"
    messages = final_res.get("messages", [])
    ai_msg = messages[-1]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.additional_kwargs.get("selected_engine") == "gemini"
    assert ai_msg.content == gemini_candidate


def test_hitl_human_selects_both_dual_view():
    """Verify that human selecting both retains both outputs with dual view flag."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = f"test-hitl-{uuid.uuid4()}"

    res = run_constitution_chat(
        question="What is Article 32?",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=True,
    )
    assert res.get("interrupted") is True

    final_res = resume_constitution_chat(
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        selected_engine="both",
    )

    assert final_res.get("selected_engine") == "both"
    messages = final_res.get("messages", [])
    ai_msg = messages[-1]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.additional_kwargs.get("selected_engine") == "both"
    assert "gemini_answer" in ai_msg.additional_kwargs
    assert "rule_answer" in ai_msg.additional_kwargs


def test_hitl_discard_turn_reverts_state():
    """Verify that discarding an interrupted turn safely cleans up uncompleted state."""
    sqlite_chk = get_sqlite_checkpointer()
    store = get_global_store()
    thread_id = f"test-hitl-discard-{uuid.uuid4()}"

    try:
        # Completed turn 1
        res1 = run_constitution_chat(
            question="Hello",
            thread_id=thread_id,
            checkpointer=sqlite_chk,
            store=store,
            enable_interrupt=False,
        )
        assert len(res1.get("messages", [])) == 2

        # Interrupted turn 2
        res2 = run_constitution_chat(
            question="What is Article 21?",
            thread_id=thread_id,
            checkpointer=sqlite_chk,
            store=store,
            enable_interrupt=True,
        )
        assert res2.get("interrupted") is True

        # Discard the interrupted turn
        discarded = discard_thread_interrupted_turn(thread_id)
        assert discarded is True

        # Check restored state
        graph = get_compiled_graph(checkpointer=sqlite_chk, store=store)
        cfg = {"configurable": {"thread_id": thread_id}}
        state_after = graph.get_state(cfg)
        assert len(state_after.next) == 0
        assert len(state_after.values["messages"]) == 2
    finally:
        delete_thread_checkpoints(thread_id)
