"""Conditional routing functions for LangGraph workflow."""

from __future__ import annotations

from typing import Literal

from app.graph.state import ConstitutionState


def route_classification(
    state: ConstitutionState,
) -> Literal[
    "handle_general_query",
    "retrieve_constitution",
    "resolve_context",
    "handle_uncertainty",
    "finalize_response",
]:
    """Route after classify_question node."""
    intent = state.get("intent", "uncertain")

    if intent == "cached_response":
        return "finalize_response"

    if intent == "out_of_scope":
        return "handle_uncertainty"

    if intent == "greeting":
        return "handle_general_query"

    if intent in [
        "article_lookup",
        "article_simple",
        "article_example",
        "topic_lookup",
        "comparison",
        "general_search",
    ]:
        return "retrieve_constitution"

    if intent.startswith("follow_up"):
        return "resolve_context"

    return "handle_uncertainty"


def route_context(
    state: ConstitutionState,
) -> Literal[
    "handle_uncertainty", "retrieve_constitution", "finalize_response"
]:
    """Route after resolve_context node."""
    if state.get("needs_clarification", False):
        return "handle_uncertainty"

    # If the user specifically asked for conversation history,
    # resolve_context has already prepared the answer
    if state.get("intent") == "follow_up_history":
        return "finalize_response"

    return "retrieve_constitution"


def route_retrieval(
    state: ConstitutionState,
) -> Literal["generate_answer", "handle_uncertainty"]:
    """Route after check_retrieval node."""
    has_comp = bool(state.get("comparison_data"))
    has_docs = len(state.get("retrieved_documents", [])) > 0
    has_topic = bool(state.get("topic"))

    if has_comp or has_docs or has_topic:
        return "generate_answer"

    return "handle_uncertainty"
