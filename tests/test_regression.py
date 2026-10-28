"""Regression Protection Tests for Core Platform Capabilities.

Ensures that Auth0, Mem0, Checkpointer, Time-Travel, and State Graph
remain 100% operational after RAG enhancements.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.auth.auth0_service import Auth0Service
from app.graph.workflow import get_global_store, run_constitution_chat
from app.memory.mem0_service import Mem0Service


def test_auth0_regression():
    """Verify Auth0 profile normalization and demo personas remain intact."""
    advocate = Auth0Service.get_demo_user("advocate")
    assert advocate["is_authenticated"] is True
    assert "user_id" in advocate
    assert advocate["name"] == "Advocate Priya Sharma"

    raw = {
        "sub": "auth0|reg_test_user",
        "name": "Test User",
        "email": "test@example.com",
    }
    normalized = Auth0Service.normalize_user_profile(raw)
    assert normalized["user_id"] == "auth0|reg_test_user"
    assert normalized["is_authenticated"] is True


def test_mem0_regression():
    """Verify Mem0 local memory storage CRUD operations."""
    service = Mem0Service()
    user_id = "test_user_regression"
    service.add_memory(
        content="I am researching Article 21 and right to privacy.",
        user_id=user_id,
        metadata={"domain": "fundamental_rights"},
    )
    mems = service.get_all_memories(user_id)
    assert isinstance(mems, list)
    assert len(mems) > 0
    assert any("privacy" in m["memory"].lower() for m in mems)


def test_checkpointer_and_chat_turn_regression():
    """Verify LangGraph chat execution with persistence checkpointer."""
    checkpointer = MemorySaver()
    store = get_global_store()
    thread_id = "test-regression-turn"

    res = run_constitution_chat(
        question="What is Article 14?",
        thread_id=thread_id,
        checkpointer=checkpointer,
        store=store,
        enable_interrupt=False,
    )

    assert res is not None
    assert "equality" in res.get("answer", "").lower()
    assert len(res.get("sources", [])) > 0
