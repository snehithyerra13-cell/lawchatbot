"""Tests for Enhanced Rule-Based Context Continuity and Conversation Deletion.
"""

import sqlite3
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph.workflow import (
    run_constitution_chat,
    get_compiled_graph,
)
from app.persistence.conversation_db import get_conversation_db
from app.persistence.checkpointer import delete_thread_checkpoints


@pytest.fixture
def test_setup():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    yield saver
    conn.close()


def test_rule_based_context_continuity(test_setup):
    """Verify Rule-Based Engine demonstrates explicit context continuity on follow-ups."""
    saver = test_setup
    thread_id = "test-rule-context-thread"

    # Turn 1: Article 21 lookup
    res1 = run_constitution_chat("What is Article 21?", thread_id, checkpointer=saver)
    assert res1["article_number"] == "21"
    assert "Article 21" in res1["answer"]

    # Turn 2: Follow-up simple explanation
    res2 = run_constitution_chat("Explain it simply", thread_id, checkpointer=saver)
    assert res2["article_number"] == "21"
    # Rule engine must include explicit context banner
    assert "Context Continuity" in res2["answer"]
    assert "Article 21" in res2["answer"]

    # Turn 3: Question about conversation history
    res3 = run_constitution_chat("what article i had asked you", thread_id, checkpointer=saver)
    assert "Article 21" in res3["answer"]
    assert "Conversation Context & History Recap" in res3["answer"]
    assert "Verbatim Provision from Constitution PDF" in res3["answer"]


def test_rule_based_multi_article_history(test_setup):
    """Verify Rule-Based Engine tracks and recaps multiple articles discussed in a thread."""
    saver = test_setup
    thread_id = "test-multi-art-context"

    # Turn 1: Article 14
    run_constitution_chat("What is Article 14?", thread_id, checkpointer=saver)
    # Turn 2: Article 21
    run_constitution_chat("What is Article 21?", thread_id, checkpointer=saver)

    # Turn 3: Multi-article recap inquiry
    res3 = run_constitution_chat("which articles have we discussed?", thread_id, checkpointer=saver)
    assert "Article 21" in res3["answer"]
    assert "Article 14" in res3["answer"]
    assert "Articles Explored in this Consultation" in res3["answer"]


def test_conversation_deletion():
    """Verify that deleting a conversation removes it from registry and SQLite checkpoints."""
    db = get_conversation_db()
    t_id = "test-delete-validation-thread"

    # Create thread in registry
    db.create_conversation(t_id, "Temporary Thread for Deletion")
    found = db.get_conversation(t_id)
    assert found is not None

    # Delete thread
    ok_db = db.delete_conversation(t_id)
    assert ok_db is True
    assert db.get_conversation(t_id) is None

    # Delete checkpoints
    ok_chk = delete_thread_checkpoints(t_id)
    assert ok_chk is True
