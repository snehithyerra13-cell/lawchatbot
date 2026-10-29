"""Context-Aware Follow-Up and Pronoun Resolver for ConstitutionGPT.

Interprets follow-up queries using active conversation history:
- Resolves phrases like "in detail", "make it crisp", "give an example", "what are its exceptions",
  "explain about it", "compare both", "continue", and "the second point".
- Distinguishes between contextual follow-ups, topic pivots (e.g., "What about Article 21?"),
  and explicit subject changes (e.g., "What is the capital of France?").
- Guarantees deterministic, sub-millisecond resolution for standard follow-up patterns
  without wasting LLM tokens or incurring unnecessary API latency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)


@dataclass
class ResolvedQueryContext:
    """Outcome of contextual query resolution."""

    original_query: str
    resolved_query: str
    is_follow_up: bool
    follow_up_type: Optional[str] = None  # "detail", "crisp", "detail_and_crisp", "example", "simplify", "exceptions", "pronoun", "pivot", "compare", "history"
    target_articles: List[str] = field(default_factory=list)
    active_topic: Optional[str] = None
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    confidence: float = 1.0
    audit_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "resolved_query": self.resolved_query,
            "is_follow_up": self.is_follow_up,
            "follow_up_type": self.follow_up_type,
            "target_articles": self.target_articles,
            "active_topic": self.active_topic,
            "needs_clarification": self.needs_clarification,
            "clarification_message": self.clarification_message,
            "confidence": self.confidence,
            "audit_notes": self.audit_notes,
        }


def extract_articles_from_string(text: str) -> List[str]:
    """Extract canonical article numbers (1 to 395 with optional suffix, or PREAMBLE)."""
    if not text:
        return []
    text_up = text.upper()
    found: List[str] = []
    if "PREAMBLE" in text_up:
        found.append("PREAMBLE")

    matches = re.findall(r"\b(?:ARTICLE|ART\.?|A)?\s*(\d+[A-Z]?)\b", text_up)
    for m in matches:
        base_match = re.match(r"^(\d+)", m)
        if base_match:
            val = int(base_match.group(1))
            if 1 <= val <= 395 and m not in found:
                found.append(m)
    return found


def extract_prior_context_from_messages(
    messages: Sequence[BaseMessage],
) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """Scan message history in reverse to extract last article, title, topic, and prior user question."""
    last_article: Optional[str] = None
    last_title: Optional[str] = None
    active_topic: Optional[str] = None
    prior_user_question: Optional[str] = None
    discussed_articles: List[str] = []

    if not messages:
        return None, None, None, []

    # Filter human prompts to identify prior question
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    # The immediate previous user question before current turn (if current turn is already appended)
    if len(human_messages) >= 2:
        prior_user_question = getattr(human_messages[-2], "content", "").strip()
    elif len(human_messages) == 1:
        prior_user_question = getattr(human_messages[0], "content", "").strip()

    # Reverse scan for discussed articles and topics
    for msg in reversed(messages):
        kwargs = getattr(msg, "additional_kwargs", {}) or {}
        art = kwargs.get("article_number")
        if art and not last_article:
            last_article = str(art).strip().upper()
        if kwargs.get("last_discussed_title") and not last_title:
            last_title = str(kwargs["last_discussed_title"]).strip()
        if kwargs.get("topic") and not active_topic:
            active_topic = str(kwargs["topic"]).strip()

        content = getattr(msg, "content", "")
        # Look for explicit Article headers
        m_head = re.search(r"###\s*(?:Simple English Explanation:\s*|Official Constitutional Provision \()?(?:Article\s+)?(\d+[A-Z]?|PREAMBLE)", content, re.IGNORECASE)
        if m_head and not last_article:
            last_article = m_head.group(1).upper()

        arts = extract_articles_from_string(content)
        for a in arts:
            if a not in discussed_articles:
                discussed_articles.append(a)

    if not last_article and discussed_articles:
        last_article = discussed_articles[0]

    return last_article, last_title, active_topic, discussed_articles


def resolve_follow_up_query(
    question: str,
    conversation_history: Sequence[BaseMessage] = (),
    last_discussed_article: Optional[str] = None,
    last_discussed_title: Optional[str] = None,
    active_topic: Optional[str] = None,
) -> ResolvedQueryContext:
    """Analyze the user's latest query in the context of previous conversation turns.
    
    Determines whether it is a follow-up or a standalone query, and constructs the
    fully grounded resolved query.
    """
    clean_q = " ".join(question.strip().split())
    q_lower = clean_q.lower()
    audit: List[str] = []

    # Step 1: Detect explicit new article mentions in the current question
    current_articles = extract_articles_from_string(clean_q)

    # If user explicitly pivots to another article (e.g. "what about Article 21?", "how about Article 32?")
    pivot_patterns = [
        r"^(?:what|how)\s+about\s+(?:article\s+)?(\d+[a-z]?|preamble)\b",
        r"^(?:and|tell\s+me\s+about)\s+(?:article\s+)?(\d+[a-z]?|preamble)\b",
        r"^(?:what\s+does|explain)\s+(?:article\s+)?(\d+[a-z]?|preamble)\s+(?:say|protect|mandate)\b",
    ]
    is_pivot = False
    pivot_art = None
    for p in pivot_patterns:
        m_p = re.search(p, q_lower)
        if m_p:
            is_pivot = True
            pivot_art = m_p.group(1).upper()
            break

    if is_pivot and pivot_art:
        audit.append(f"Explicit topic pivot detected to Article {pivot_art}.")
        resolved = f"Explain Article {pivot_art} of the Constitution of India and its constitutional protections."
        return ResolvedQueryContext(
            original_query=clean_q,
            resolved_query=resolved,
            is_follow_up=True,
            follow_up_type="pivot",
            target_articles=[pivot_art],
            active_topic="Fundamental Rights" if pivot_art in ["14", "19", "21", "21A", "22", "32"] else None,
            confidence=1.0,
            audit_notes=audit,
        )

    # If the user asks a complete, standalone question with its own explicit article (e.g., "Explain Article 21")
    if current_articles:
        audit.append(f"Query contains explicit article references: {current_articles}.")
        return ResolvedQueryContext(
            original_query=clean_q,
            resolved_query=clean_q,
            is_follow_up=False,
            follow_up_type=None,
            target_articles=current_articles,
            active_topic=active_topic,
            confidence=1.0,
            audit_notes=audit,
        )

    # Step 2: Extract previous conversational context from state and message history
    hist_art, hist_title, hist_topic, discussed_arts = extract_prior_context_from_messages(conversation_history)
    effective_article = last_discussed_article or hist_art
    effective_title = last_discussed_title or hist_title or (f"Article {effective_article}" if effective_article else None)
    effective_topic = active_topic or hist_topic

    # Step 3: Check for standalone out-of-domain inquiries that MUST NOT inherit context
    # E.g. "What is the capital of France?", "Write a Python sorting program", "Who is Narendra Modi?"
    explicit_out_of_domain_indicators = [
        "capital of", "weather in", "recipe for", "bake a", "python code", "java code",
        "sorting program", "cricket score", "ipl score", "movie review", "stock market",
        "who is modi", "who is narendra modi", "who is rahul gandhi", "who is kejriwal",
    ]
    for ood in explicit_out_of_domain_indicators:
        if ood in q_lower:
            audit.append(f"Explicit non-constitutional subject change detected ('{ood}'). Will not inherit context.")
            return ResolvedQueryContext(
                original_query=clean_q,
                resolved_query=clean_q,
                is_follow_up=False,
                follow_up_type=None,
                target_articles=[],
                active_topic=None,
                confidence=1.0,
                audit_notes=audit,
            )

    # Step 4: Classify follow-up patterns
    # Check for "in detail and crisp" / combination
    has_detail = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ["in detail", "detailed", "elaborate", "tell me more", "expand", "deep dive", "in depth", "more details"])
    has_crisp = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ["crisp", "make it crisp", "summarize", "in short", "briefly", "concise", "tldr", "short summary", "bullet points"])
    has_example = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ["give an example", "give me an example", "an example", "examples", "practical example", "case example", "scenario", "illustration"])
    has_simplify = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ["simplify", "simple english", "in simple words", "in simpler words", "simple terms", "for a beginner", "layman"])
    has_exceptions = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ["exceptions", "what are its exceptions", "limitations", "reasonable classification", "restrictions", "provisos", "proviso"])
    has_clauses = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ["second point", "first point", "first clause", "second clause", "sub clause", "exact wording", "verbatim text"])
    has_compare = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ["compare both", "difference between both", "compare them", "difference between them"])
    has_pronoun = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ["explain about it", "tell me about it", "what does it mean", "why is it important", "explain this", "what about that", "the same", "explain the same", "about it"])

    # If the query is an isolated follow-up fragment
    is_contextual_follow_up = (
        has_detail
        or has_crisp
        or has_example
        or has_simplify
        or has_exceptions
        or has_clauses
        or has_compare
        or has_pronoun
        or q_lower in ["continue", "next", "more", "explain", "detail", "crisp"]
    )

    if is_contextual_follow_up:
        # Check if we have preceding context to resolve
        if not effective_article and not effective_topic:
            audit.append("Follow-up fragment detected, but no prior conversation context found.")
            return ResolvedQueryContext(
                original_query=clean_q,
                resolved_query=clean_q,
                is_follow_up=True,
                needs_clarification=True,
                clarification_message=(
                    "Could you please specify which constitutional article or topic you would like me to address? "
                    "(For example: *'Explain Article 14 in detail'* or *'Explain Fundamental Rights'*)."
                ),
                confidence=1.0,
                audit_notes=audit,
            )

        target_art_str = f"Article {effective_article}" if effective_article else (effective_topic or "the Constitution of India")
        articles_list = [effective_article] if effective_article else []

        # Combination: "in detail and crisp"
        if has_detail and has_crisp:
            audit.append(f"Combined detail & crisp follow-up resolved for {target_art_str}.")
            resolved = f"Provide a detailed yet crisp and concise explanation of {target_art_str} of the Constitution of India, highlighting its essential legal principles and direct provisions."
            f_type = "detail_and_crisp"

        elif has_detail:
            audit.append(f"Elaboration follow-up resolved for {target_art_str}.")
            resolved = f"Provide a more detailed, in-depth, Constitution-grounded explanation of {target_art_str} of the Constitution of India."
            f_type = "detail"

        elif has_crisp:
            audit.append(f"Crisp summary follow-up resolved for {target_art_str}.")
            resolved = f"Provide a concise, crisp summary of {target_art_str} of the Constitution of India, retaining the key legal meaning and provisions."
            f_type = "crisp"

        elif has_example:
            audit.append(f"Example request follow-up resolved for {target_art_str}.")
            resolved = f"Provide constitutionally accurate practical examples and case scenarios illustrating {target_art_str} of the Constitution of India."
            f_type = "example"

        elif has_simplify:
            audit.append(f"Simplification follow-up resolved for {target_art_str}.")
            resolved = f"Explain {target_art_str} of the Constitution of India in simple plain English for easy understanding."
            f_type = "simplify"

        elif has_exceptions:
            audit.append(f"Exceptions/limitations follow-up resolved for {target_art_str}.")
            resolved = f"What are the constitutional exceptions, reasonable classifications, and permissible limitations under {target_art_str} of the Constitution of India?"
            f_type = "exceptions"

        elif has_clauses:
            audit.append(f"Specific clause/point follow-up resolved for {target_art_str}.")
            resolved = f"Explain the specific clauses, points, and exact provisions of {target_art_str} of the Constitution of India referenced in our discussion."
            f_type = "clauses"

        elif has_compare:
            if len(discussed_arts) >= 2:
                art1, art2 = discussed_arts[0], discussed_arts[1]
                audit.append(f"Comparative follow-up resolved between Article {art1} and Article {art2}.")
                resolved = f"Compare Article {art1} and Article {art2} of the Constitution of India, detailing similarities, differences, and practical applications."
                articles_list = [art1, art2]
            else:
                audit.append(f"Comparative follow-up resolved for {target_art_str}.")
                resolved = f"Compare {target_art_str} with related fundamental rights in the Constitution of India."
            f_type = "compare"

        else:  # Pronoun / general continuation
            audit.append(f"Pronoun/continuation follow-up resolved for {target_art_str}.")
            resolved = f"Explain {target_art_str} of the Constitution of India in detail regarding '{clean_q}'."
            f_type = "pronoun"

        return ResolvedQueryContext(
            original_query=clean_q,
            resolved_query=resolved,
            is_follow_up=True,
            follow_up_type=f_type,
            target_articles=articles_list,
            active_topic=effective_topic,
            confidence=0.98,
            audit_notes=audit,
        )

    # If query is neither a follow-up fragment nor explicit article query, return as-is
    audit.append("Standard or standalone query passed without follow-up transformation.")
    return ResolvedQueryContext(
        original_query=clean_q,
        resolved_query=clean_q,
        is_follow_up=False,
        follow_up_type=None,
        target_articles=[],
        active_topic=active_topic,
        confidence=1.0,
        audit_notes=audit,
    )
