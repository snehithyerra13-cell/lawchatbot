"""Tests for Gemini AI integration and multi-turn context retention."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.retrieval.gemini_service import (
    format_conversation_history,
    format_retrieved_evidence,
    generate_gemini_constitutional_response,
)
from app.retrieval.knowledge_base import get_knowledge_base


def test_format_conversation_history():
    """Verify conversation history formatting for Gemini prompt context."""
    history = [
        HumanMessage(content="What is Article 21?"),
        AIMessage(content="Article 21 protects life and personal liberty."),
        HumanMessage(content="Explain it simply."),
        AIMessage(content="It guarantees the right to live with human dignity."),
    ]
    formatted = format_conversation_history(history)
    assert "User: What is Article 21?" in formatted
    assert "Assistant: Article 21 protects life" in formatted
    assert "User: Explain it simply." in formatted


def test_format_retrieved_evidence():
    """Verify formatting of retrieved PDF constitutional clauses."""
    kb = get_knowledge_base()
    art14 = kb.lookup_article("14")
    assert art14 is not None

    evidence = format_retrieved_evidence([art14])
    assert "Article: 14" in evidence
    assert "Equality before law" in evidence
    assert "Page" in evidence


def test_gemini_context_aware_generation():
    """Verify Gemini generates a grounded response with prior context knowledge."""
    kb = get_knowledge_base()
    art21 = kb.lookup_article("21")
    assert art21 is not None

    history = [
        HumanMessage(content="What is Article 21?"),
        AIMessage(content="Article 21 provides protection of life and personal liberty."),
    ]

    # Ask follow-up without mentioning "Article 21"
    response = generate_gemini_constitutional_response(
        question="Explain it in simpler words.",
        retrieved_docs=[art21],
        conversation_history=history,
    )
    assert len(response) > 50
    assert "Article 21" in response or "21" in response or "liberty" in response.lower()
