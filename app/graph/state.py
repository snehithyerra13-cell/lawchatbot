"""LangGraph State definition for ConstitutionGPT."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ConstitutionState(TypedDict, total=False):
    """The central state dictionary passed between LangGraph nodes.

    Attributes:
        messages: Conversation message history, accumulated via add_messages reducer.
        question: The latest sanitized user prompt.
        intent: Classified query intent (greeting, article_lookup, topic_lookup, comparison, follow_up, etc.)
        article_number: Primary targeted article number (e.g., '14', '21', '51A', 'PREAMBLE').
        comparison_articles: List of two article numbers when performing side-by-side comparison.
        topic: Specific topic string if a thematic query was matched.
        retrieved_documents: Articles/sections retrieved from the local knowledge base.
        comparison_data: Comparative synthesis data structure if intent is comparison.
        answer: The final formatted constitutional response text.
        sources: Citation metadata (PDF filename, page numbers, status).
        needs_clarification: Flag indicating ambiguous context requiring user guidance.
        response_type: Category identifier for UI rendering ('article', 'topic', 'comparison', 'general', 'uncertain').
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    raw_question: Optional[str]
    rewritten_question: Optional[str]
    query_rewriter_metadata: Optional[Dict[str, Any]]
    intent: str
    article_number: Optional[str]
    last_discussed_article: Optional[str]
    last_discussed_title: Optional[str]
    comparison_articles: Optional[List[str]]
    topic: Optional[str]
    retrieved_documents: List[Dict[str, Any]]
    comparison_data: Optional[Dict[str, Any]]
    answer: str
    gemini_answer: Optional[str]
    sources: List[Dict[str, Any]]
    needs_clarification: bool
    response_type: str
    requires_human_review: Optional[bool]
    interrupted: Optional[bool]
    subgraph_metadata: Optional[Dict[str, Any]]
    user_id: Optional[str]
    mem0_memories: Optional[List[Dict[str, Any]]]
    event_log: Optional[List[str]]
    prior_user_query: Optional[str]
    candidate_documents: Optional[List[Dict[str, Any]]]
    rerank_mode: Optional[str]
    rerank_metadata: Optional[Dict[str, Any]]
    selected_engine: Optional[str]
    execution_time_seconds: Optional[float]
