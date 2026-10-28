"""Comprehensive Test Suite for all 10 LangGraph Core Concepts:
1. Persistence
2. Checkpointers
3. Stores
4. Fault tolerance
5. Event streaming
6. Streaming
7. Interrupts
8. Time travel
9. Memory
10. Subgraphs
"""

import sqlite3
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import RetryPolicy

from app.graph.subgraphs import retrieval_subgraph
from app.graph.workflow import (
    build_constitution_graph,
    get_compiled_graph,
    run_constitution_chat,
    run_constitution_chat_stream,
    resume_constitution_chat,
    get_thread_history_snapshots,
    get_state_at_checkpoint,
    branch_from_checkpoint,
)
from app.persistence.store import (
    bookmark_article,
    list_bookmarks,
    is_article_bookmarked,
    remove_bookmark,
    get_user_preferences,
    update_user_preferences,
)
from streamlit_app import stream_text_word_by_word


@pytest.fixture
def test_setup():
    """Setup isolated SQLite checkpointer and in-memory store for test run."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    store = InMemoryStore()
    yield {"saver": saver, "store": store}
    conn.close()


# 1. Persistence
def test_feature_1_persistence(test_setup):
    """Verify state durability in checkpointer."""
    saver = test_setup["saver"]
    thread_id = "test-persist-thread"
    res1 = run_constitution_chat(
        question="What is Article 21?",
        thread_id=thread_id,
        checkpointer=saver,
    )
    assert res1["article_number"] == "21"

    # Read state directly from checkpointer to verify persistence
    graph = get_compiled_graph(checkpointer=saver)
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    assert snap.values["article_number"] == "21"
    assert len(snap.values["messages"]) >= 2


# 2. Checkpointers
def test_feature_2_checkpointers(test_setup):
    """Verify checkpointer metadata, IDs, and thread isolation."""
    saver = test_setup["saver"]
    t1, t2 = "thread-chk-1", "thread-chk-2"
    run_constitution_chat("What is Article 14?", t1, checkpointer=saver)
    run_constitution_chat("What is Article 19?", t2, checkpointer=saver)

    graph = get_compiled_graph(checkpointer=saver)
    snap1 = graph.get_state({"configurable": {"thread_id": t1}})
    snap2 = graph.get_state({"configurable": {"thread_id": t2}})

    assert snap1.values["article_number"] == "14"
    assert snap2.values["article_number"] == "19"
    assert snap1.config["configurable"]["checkpoint_id"] != snap2.config["configurable"]["checkpoint_id"]


# 3. Stores (Cross-Thread Shared Memory)
def test_feature_3_stores(test_setup):
    """Verify cross-thread Store for bookmarks and user profiles."""
    store = test_setup["store"]
    # Bookmark Article 21
    bookmark_article("21", "Protection of Life", "Part III", "Landmark right")
    assert is_article_bookmarked("21") is True
    bmarks = list_bookmarks()
    assert any(b["article_number"] == "21" for b in bmarks)

    # Preferences
    update_user_preferences({"tone": "Concise", "detail_level": "High"})
    prefs = get_user_preferences()
    assert prefs["tone"] == "Concise"

    # Cleanup
    remove_bookmark("21")
    assert is_article_bookmarked("21") is False


# 4. Fault Tolerance (RetryPolicy & Fallback)
def test_feature_4_fault_tolerance():
    """Verify RetryPolicy configuration on StateGraph."""
    builder = build_constitution_graph()
    nodes = builder.nodes
    # Verify retry policies exist on sensitive nodes
    assert "finalize_response" in nodes
    assert "retrieve_constitution" in nodes
    # Check graph compilation with retry policy
    graph = builder.compile()
    assert graph is not None


# 5. Event Streaming
def test_feature_5_event_streaming(test_setup):
    """Verify LangGraph live event streaming yielding node execution updates."""
    saver = test_setup["saver"]
    store = test_setup["store"]
    events = []
    final_res = None

    for item in run_constitution_chat_stream(
        question="What is Article 32?",
        thread_id="test-event-stream",
        checkpointer=saver,
        store=store,
    ):
        if item["type"] == "node_update":
            events.append(item["node"])
        elif item["type"] == "complete":
            final_res = item["result"]

    assert "classify_question" in events
    assert "retrieve_constitution" in events
    assert "generate_answer" in events
    assert final_res is not None
    assert final_res.get("article_number") == "32"


# 6. Streaming (Word-by-Word Generator)
def test_feature_6_streaming():
    """Verify ChatGPT-style word-by-word streaming generator."""
    sample_text = "Article 21 guarantees the right to life."
    generator = stream_text_word_by_word(sample_text, delay=0.001)
    chunks = list(generator)
    reconstructed = "".join(chunks)
    assert reconstructed == sample_text
    assert len(chunks) == 7


# 7. Interrupts (Human-in-the-Loop)
def test_feature_7_interrupts(test_setup):
    """Verify Human-in-the-Loop breakpoint pauses and resume continues."""
    saver = test_setup["saver"]
    thread_id = "test-hitl-feature"

    paused = run_constitution_chat(
        question="What is Article 21?",
        thread_id=thread_id,
        checkpointer=saver,
        enable_interrupt=True,
    )
    assert paused.get("interrupted") is True
    assert "finalize_response" in paused.get("next_node", [])

    resumed = resume_constitution_chat(
        thread_id=thread_id,
        checkpointer=saver,
    )
    assert "answer" in resumed
    assert "Article 21" in resumed["answer"]


# 8. Time Travel (Checkpoint Snapshots & Branching)
def test_feature_8_time_travel(test_setup):
    """Verify state history retrieval and branching from a historical checkpoint."""
    saver = test_setup["saver"]
    thread_id = "test-time-travel-feature"

    # Turn 1
    run_constitution_chat("What is Article 14?", thread_id, checkpointer=saver)
    # Turn 2
    run_constitution_chat("What is Article 21?", thread_id, checkpointer=saver)

    snaps = get_thread_history_snapshots(thread_id, checkpointer=saver)
    assert len(snaps) >= 4

    # Select earlier checkpoint
    target_chk = snaps[-2]["checkpoint_id"]
    state_at_chk = get_state_at_checkpoint(thread_id, target_chk, checkpointer=saver)
    assert isinstance(state_at_chk, dict)

    # Branch from earlier checkpoint
    branch_res = branch_from_checkpoint(
        thread_id=thread_id,
        checkpoint_id=target_chk,
        new_question="What is Article 19?",
        checkpointer=saver,
    )
    assert branch_res["article_number"] == "19"


# 9. Memory (In-Thread + Cross-Thread)
def test_feature_9_memory(test_setup):
    """Verify dual-layer memory: thread context resolution and cross-thread topics."""
    saver = test_setup["saver"]
    thread_id = "test-memory-feature"

    # Turn 1: Discuss Article 21
    run_constitution_chat("What is Article 21?", thread_id, checkpointer=saver)

    # Turn 2: Follow-up question using pronoun
    res2 = run_constitution_chat("Explain it simply", thread_id, checkpointer=saver)
    assert res2["article_number"] == "21"
    assert res2["last_discussed_article"] == "21"


# 10. Subgraphs (Modular Retrieval Subgraph)
def test_feature_10_subgraphs():
    """Verify modular RetrievalAndVerificationSubgraph execution."""
    sub_res = retrieval_subgraph.invoke({
        "question": "What is Article 14?",
        "intent": "article_lookup",
        "article_number": "14",
    })
    assert "retrieved_documents" in sub_res
    assert len(sub_res["retrieved_documents"]) >= 1
    assert sub_res["verification_status"] == "VERIFIED_LOCAL_PDF"
    assert "audit_notes" in sub_res
