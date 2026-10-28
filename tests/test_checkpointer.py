"""Tests for persistent SQLite checkpointer, state recovery, and thread isolation."""

import sqlite3
import tempfile
from pathlib import Path
import uuid
import pytest

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage

from app.graph.workflow import build_constitution_graph, run_constitution_chat


def test_sqlite_checkpointer_persistence():
    """Verify state persists across re-opening SQLite checkpointer."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        # Create checkpointer 1
        conn1 = sqlite3.connect(str(db_path), check_same_thread=False)
        saver1 = SqliteSaver(conn1)
        saver1.setup()

        builder1 = build_constitution_graph()
        graph1 = builder1.compile(checkpointer=saver1)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke turn 1 on graph 1
        graph1.invoke(
            {"messages": [HumanMessage(content="What is Article 21?")], "question": "What is Article 21?"},
            config=config,
        )

        conn1.close()

        # Reopen checkpointer on the same database file (graph 2)
        conn2 = sqlite3.connect(str(db_path), check_same_thread=False)
        saver2 = SqliteSaver(conn2)
        saver2.setup()

        builder2 = build_constitution_graph()
        graph2 = builder2.compile(checkpointer=saver2)

        # Retrieve state from reopened database
        saved_state = graph2.get_state(config)
        assert saved_state is not None
        messages = saved_state.values.get("messages", [])
        assert len(messages) == 2
        assert "Article 21" in messages[-1].content

        conn2.close()
    finally:
        if db_path.exists():
            db_path.unlink()


def test_thread_isolation():
    """Verify that multiple threads maintain completely separate states."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()

        builder = build_constitution_graph()
        graph = builder.compile(checkpointer=saver)

        thread_a = str(uuid.uuid4())
        thread_b = str(uuid.uuid4())

        # Thread A talks about Article 14
        graph.invoke(
            {"messages": [HumanMessage(content="What is Article 14?")], "question": "What is Article 14?"},
            config={"configurable": {"thread_id": thread_a}},
        )

        # Thread B talks about Article 21
        graph.invoke(
            {"messages": [HumanMessage(content="What is Article 21?")], "question": "What is Article 21?"},
            config={"configurable": {"thread_id": thread_b}},
        )

        state_a = graph.get_state({"configurable": {"thread_id": thread_a}})
        state_b = graph.get_state({"configurable": {"thread_id": thread_b}})

        assert "Article 14" in state_a.values["messages"][-1].content
        assert "Article 21" in state_b.values["messages"][-1].content
        assert len(state_a.values["messages"]) == 2
        assert len(state_b.values["messages"]) == 2

        conn.close()
    finally:
        if db_path.exists():
            db_path.unlink()
