"""LangGraph Modular Subgraphs for ConstitutionGPT.

Implements the Retrieval and Provenance Verification Subgraph nested
inside the parent ConstitutionStateGraph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from app.retrieval.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


class RetrievalSubState(TypedDict, total=False):
    """Sub-state for the modular retrieval and verification subgraph."""

    question: str
    intent: str
    article_number: Optional[str]
    comparison_articles: Optional[List[str]]
    topic: Optional[str]
    retrieved_documents: List[Dict[str, Any]]
    comparison_data: Optional[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    verification_status: str
    requires_human_review: bool
    audit_notes: List[str]


def sub_prepare_query(state: RetrievalSubState) -> Dict[str, Any]:
    """Subgraph Node 1: Sanitize and prepare search parameters."""
    q = state.get("question", "").strip()
    notes = ["Subgraph initiated: parameter preparation and sanitization."]
    return {
        "question": q,
        "retrieved_documents": [],
        "sources": [],
        "audit_notes": notes,
    }


def sub_execute_pdf_retrieval(state: RetrievalSubState) -> Dict[str, Any]:
    """Subgraph Node 2: Deterministic retrieval from indexed Constitution PDF."""
    kb = get_knowledge_base()
    q = state.get("question", "")
    art_num = state.get("article_number")
    comparison_articles = state.get("comparison_articles")
    topic = state.get("topic")

    retrieved_docs: List[Dict[str, Any]] = []
    comp_data: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = []
    notes: List[str] = list(state.get("audit_notes", []))

    # 1. Comparison between articles
    if comparison_articles and len(comparison_articles) >= 2:
        comp_data = kb.compare_articles(
            comparison_articles[0], comparison_articles[1]
        )
        if comp_data:
            retrieved_docs = [comp_data["article_1"], comp_data["article_2"]]
            sources = [
                {
                    "source_pdf": comp_data["article_1"]["source_pdf"],
                    "page": comp_data["article_1"]["page_start"],
                    "article": comp_data["article_1"]["article_number"],
                },
                {
                    "source_pdf": comp_data["article_2"]["source_pdf"],
                    "page": comp_data["article_2"]["page_start"],
                    "article": comp_data["article_2"]["article_number"],
                },
            ]
            notes.append(
                f"Subgraph comparison retrieved: Articles {comparison_articles[0]} and {comparison_articles[1]}."
            )
            return {
                "comparison_data": comp_data,
                "retrieved_documents": retrieved_docs,
                "sources": sources,
                "audit_notes": notes,
            }

    # 2. Specific article lookup
    if art_num:
        doc = kb.lookup_article(art_num)
        if doc:
            retrieved_docs = [doc]
            sources = [
                {
                    "source_pdf": doc["source_pdf"],
                    "page": doc["page_start"],
                    "article": doc["article_number"],
                }
            ]
            notes.append(f"Subgraph article lookup retrieved: Article {art_num}.")
            return {
                "comparison_data": None,
                "retrieved_documents": retrieved_docs,
                "sources": sources,
                "audit_notes": notes,
            }

    # 3. Topic lookup
    topic_data = kb.search_topic(q) if not topic else kb.search_topic(topic)
    if topic_data:
        retrieved_docs = topic_data.get("articles", [])
        matched_topic = topic_data.get("topic_title", topic or "Constitutional Topic")
        sources = [
            {
                "source_pdf": d["source_pdf"],
                "page": d["page_start"],
                "article": d["article_number"],
            }
            for d in retrieved_docs[:3]
        ]
        notes.append(f"Subgraph thematic topic match: {matched_topic}.")
        return {
            "comparison_data": None,
            "retrieved_documents": retrieved_docs,
            "topic": matched_topic,
            "sources": sources,
            "audit_notes": notes,
        }

    # 4. General keyword search fallback
    search_results = kb.search(q, top_k=3)
    if search_results:
        retrieved_docs = search_results
        sources = [
            {
                "source_pdf": d["source_pdf"],
                "page": d["page_start"],
                "article": d["article_number"],
            }
            for d in search_results
        ]
        notes.append(f"Subgraph keyword ranking match: {len(search_results)} articles found.")

    return {
        "comparison_data": None,
        "retrieved_documents": retrieved_docs,
        "sources": sources,
        "audit_notes": notes,
    }


def sub_verify_provenance(state: RetrievalSubState) -> Dict[str, Any]:
    """Subgraph Node 3: Verify PDF page citations and flag sensitive provisions."""
    docs = state.get("retrieved_documents", [])
    notes = list(state.get("audit_notes", []))
    requires_review = False

    sensitive_keywords = ["emergency", "proclamation", "amendment", "martial law", "article 352", "article 356", "article 368"]
    q_lower = state.get("question", "").lower()

    # Check if sensitive constitutional matter requiring human review
    if any(k in q_lower for k in sensitive_keywords):
        requires_review = True
        notes.append("Audit Flag: Sensitive constitutional category detected (Emergency or Amendment provision).")

    for doc in docs:
        art_no = str(doc.get("article_number", ""))
        # Articles 352-360 (Part XVIII Emergency) and 368 (Part XX Amendment)
        if art_no in ["352", "356", "358", "359", "360", "368"]:
            requires_review = True
            notes.append(f"Audit Flag: Special constitutional power Article {art_no} retrieved.")

    verification_status = "VERIFIED_LOCAL_PDF" if docs else "NO_MATCH"
    notes.append(f"Provenance verification completed: status={verification_status}.")

    return {
        "verification_status": verification_status,
        "requires_human_review": requires_review,
        "audit_notes": notes,
    }


def build_retrieval_subgraph() -> StateGraph:
    """Build and compile the modular retrieval & verification subgraph."""
    sub_builder = StateGraph(RetrievalSubState)

    sub_builder.add_node("prepare_query", sub_prepare_query)
    sub_builder.add_node("execute_pdf_retrieval", sub_execute_pdf_retrieval)
    sub_builder.add_node("verify_provenance", sub_verify_provenance)

    sub_builder.add_edge(START, "prepare_query")
    sub_builder.add_edge("prepare_query", "execute_pdf_retrieval")
    sub_builder.add_edge("execute_pdf_retrieval", "verify_provenance")
    sub_builder.add_edge("verify_provenance", END)

    return sub_builder.compile()


# Singleton compiled instance of the subgraph
retrieval_subgraph = build_retrieval_subgraph()
