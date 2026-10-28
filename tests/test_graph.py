"""Tests for LangGraph StateGraph, nodes, routing, and message accumulation."""

import uuid
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from app.graph.nodes import classify_question, resolve_context
from app.graph.routing import route_classification, route_context, route_retrieval
from app.graph.workflow import run_constitution_chat


def test_classify_question_greetings():
    """Verify greeting intent classification."""
    state = {"question": "Hello there, good morning"}
    res = classify_question(state)
    assert res["intent"] == "greeting"
    assert route_classification(res) == "handle_general_query"


def test_classify_question_article_lookup():
    """Verify article lookup classification."""
    state = {"question": "What is Article 21?"}
    res = classify_question(state)
    assert res["intent"] == "article_lookup"
    assert res["article_number"] == "21"
    assert route_classification(res) == "retrieve_constitution"


def test_classify_question_comparison():
    """Verify comparison classification."""
    state = {"question": "Compare Article 14 and Article 21"}
    res = classify_question(state)
    assert res["intent"] == "comparison"
    assert "14" in res["comparison_articles"]
    assert "21" in res["comparison_articles"]
    assert route_classification(res) == "retrieve_constitution"


def test_classify_question_follow_up():
    """Verify follow-up classification."""
    state = {"question": "Explain it simply."}
    res = classify_question(state)
    assert res["intent"] == "follow_up_simple"
    assert route_classification(res) == "resolve_context"


def test_resolve_context_from_messages():
    """Verify extracting previous article from conversation history."""
    messages = [
        HumanMessage(content="What is Article 21?"),
        AIMessage(content="### Article 21: Protection of life and personal liberty..."),
    ]
    state = {"intent": "follow_up_simple", "messages": messages}
    res = resolve_context(state)
    assert res["article_number"] == "21"
    assert res["needs_clarification"] is False


def test_resolve_context_ambiguous():
    """Verify ambiguous context sets needs_clarification."""
    state = {"intent": "follow_up_simple", "messages": []}
    res = resolve_context(state)
    assert res["needs_clarification"] is True
    assert route_context(res) == "handle_uncertainty"


def test_full_graph_turn_execution():
    """Verify executing complete graph turn and message accumulation."""
    thread_id = str(uuid.uuid4())

    # Turn 1
    out1 = run_constitution_chat("What is Article 14?", thread_id)
    assert out1["response_type"] == "article"
    assert "Article 14" in out1["answer"]
    assert len(out1["messages"]) == 2

    # Turn 2: Follow-up resolution
    out2 = run_constitution_chat("Explain it simply.", thread_id)
    assert out2["response_type"] == "article"
    assert "Article 14" in out2["answer"] or "Plain English" in out2["answer"]
    assert len(out2["messages"]) == 4
