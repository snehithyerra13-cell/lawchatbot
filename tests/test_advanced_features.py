"""Tests for LangGraph Advanced Architecture:
1. Subgraphs (Retrieval & Verification Subgraph)
2. Interrupts (Human-in-the-Loop review)
3. Time Travel (Checkpoint snapshots & state inspection)
4. Memory (Multi-turn state persistence in SQLite)
"""

import sqlite3
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph.subgraphs import retrieval_subgraph
from app.graph.workflow import (
    get_compiled_graph,
    run_constitution_chat,
    resume_constitution_chat,
    get_thread_history_snapshots,
    get_state_at_checkpoint,
)


@pytest.fixture
def memory_checkpointer():
    """Create an isolated SQLite checkpointer in memory for testing."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    yield saver
    conn.close()


def test_retrieval_subgraph_execution():
    """Verify that the modular retrieval subgraph processes queries correctly."""
    sub_state = {
        "question": "What is Article 21?",
        "intent": "article_lookup",
        "article_number": "21",
    }
    result = retrieval_subgraph.invoke(sub_state)

    assert "retrieved_documents" in result
    assert len(result["retrieved_documents"]) >= 1
    doc = result["retrieved_documents"][0]
    assert doc["article_number"] == "21"
    assert "sources" in result
    assert len(result["sources"]) >= 1
    assert result["verification_status"] == "VERIFIED_LOCAL_PDF"
    assert "audit_notes" in result
    assert len(result["audit_notes"]) >= 2


def test_subgraph_sensitive_provision_flagging():
    """Verify that sensitive provisions (Emergency, Amendments) flag requires_human_review."""
    sub_state = {
        "question": "Explain Article 356 and President Rule",
        "intent": "article_lookup",
        "article_number": "356",
    }
    result = retrieval_subgraph.invoke(sub_state)
    assert result["requires_human_review"] is True


def test_langgraph_interrupt_and_resume(memory_checkpointer):
    """Verify Human-in-the-Loop interrupt pauses execution and resumes on user approval."""
    thread_id = "test-hitl-thread"

    # Turn with interrupt enabled
    result_paused = run_constitution_chat(
        question="What is Article 14?",
        thread_id=thread_id,
        checkpointer=memory_checkpointer,
        enable_interrupt=True,
    )

    # Must be paused before finalize_response
    assert result_paused.get("interrupted") is True
    assert "finalize_response" in result_paused.get("next_node", [])

    # Resume the execution
    result_resumed = resume_constitution_chat(
        thread_id=thread_id,
        checkpointer=memory_checkpointer,
    )

    # After resume, final answer must be populated
    assert "answer" in result_resumed
    assert "Article 14" in result_resumed["answer"]
    assert len(result_resumed["messages"]) >= 2


def test_time_travel_checkpoint_snapshots(memory_checkpointer):
    """Verify Time Travel capability: inspect checkpoints from multiple turns."""
    thread_id = "test-time-travel-thread"

    # Turn 1: Article 21
    run_constitution_chat(
        question="What is Article 21?",
        thread_id=thread_id,
        checkpointer=memory_checkpointer,
        enable_interrupt=False,
    )

    # Turn 2: Article 19
    run_constitution_chat(
        question="What is Article 19?",
        thread_id=thread_id,
        checkpointer=memory_checkpointer,
        enable_interrupt=False,
    )

    # Fetch Time Travel history snapshots
    snapshots = get_thread_history_snapshots(
        thread_id=thread_id, checkpointer=memory_checkpointer
    )

    assert len(snapshots) >= 4  # multiple graph transition steps recorded
    c_ids = [s["checkpoint_id"] for s in snapshots if s["checkpoint_id"]]
    assert len(c_ids) >= 2

    # Query state at an earlier checkpoint
    earlier_cid = c_ids[-1]
    earlier_state = get_state_at_checkpoint(
        thread_id=thread_id,
        checkpoint_id=earlier_cid,
        checkpointer=memory_checkpointer,
    )
    assert isinstance(earlier_state, dict)
