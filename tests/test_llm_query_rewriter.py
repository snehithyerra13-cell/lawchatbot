"""Tests for LLM-powered Constitutional Query Rewriter.

Verifies that when a user asks a question, it directly goes to the LLM rewriter first,
which transforms conversational, colloquial, or follow-up queries into canonical constitutional
questions so the Rule-Based engine also understands them, ensuring both LLM and Non-LLM
engines give correct, grounded answers.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.graph.workflow import get_global_store, run_constitution_chat
from app.retrieval.gemini_service import fallback_heuristic_rewrite, rewrite_constitutional_query


def test_fallback_heuristic_rewriter_follow_up():
    """Verify offline fallback correctly resolves history inquiry to prior article."""
    history = [
        HumanMessage(content="What is Article 21?"),
        AIMessage(content="### Article 21: Protection of life and personal liberty..."),
    ]
    res = fallback_heuristic_rewrite("what article i had asked you", history)
    assert "21" in res["articles"]
    assert res["intent"] == "follow_up_history"
    assert "Article 21" in res["rewritten_query"]


def test_fallback_heuristic_rewriter_comparison():
    """Verify comparison query rewrite."""
    res = fallback_heuristic_rewrite("Compare Article 14 and Article 21")
    assert "14" in res["articles"]
    assert "21" in res["articles"]
    assert res["intent"] == "comparison"


def test_llm_rewriter_enables_rule_based_correct_history_answer():
    """Verify that multi-turn history questions produce Article 21 in rule-based answer."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = "test-rewriter-history-thread"

    # Turn 1: Discuss Article 21
    res1 = run_constitution_chat(
        question="What is Article 21?",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=False,
    )
    assert "Article 21" in res1.get("answer", "")

    # Turn 2: User asks 'what article i had asked you'
    res2 = run_constitution_chat(
        question="what article i had asked you",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=False,
    )

    rule_answer = res2.get("answer", "")
    gemini_answer = res2.get("gemini_answer", "")

    # Rule-based engine must correctly identify Article 21 (NOT Article 358 or Emergency!)
    assert "Article 21" in rule_answer
    assert "Protection of life and personal liberty" in rule_answer or "Article 21" in rule_answer
    assert "358" not in rule_answer

    # Gemini AI should also acknowledge Article 21
    if gemini_answer and "unavailable" not in gemini_answer:
        assert "Article 21" in gemini_answer
