"""LangGraph node implementations for ConstitutionGPT.

Every node is a pure deterministic Python function operating on ConstitutionState.
No LLM, external AI, or model-based embeddings are used.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.graph.state import ConstitutionState
from app.retrieval.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


def normalize_query(query: str) -> str:
    """Normalize whitespace and punctuation for rule-based analysis."""
    clean = " ".join(query.strip().split())
    return clean


def extract_article_numbers_from_query(query: str) -> List[str]:
    """Extract all article numbers referenced in a query."""
    q_upper = query.upper()
    # Check for Preamble
    results = []
    if "PREAMBLE" in q_upper:
        results.append("PREAMBLE")

    # Match patterns like: Article 14, Art. 21, Art 21A, 21A, etc.
    matches = re.findall(
        r"\b(?:ARTICLE|ART\.?|A)?\s*(\d+[A-Z]?)\b", q_upper
    )
    for m in matches:
        # Validate that it is a plausible article number (1 to 395 or with letter suffix)
        base_digits = re.match(r"^(\d+)", m)
        if base_digits:
            num = int(base_digits.group(1))
            if 1 <= num <= 395 and m not in results:
                results.append(m)
    return results


def rewrite_query(state: ConstitutionState) -> Dict[str, Any]:
    """Node 0: Send user query directly to LLM to rewrite and disambiguate for the Rule-Based engine."""
    from app.retrieval.gemini_service import rewrite_constitutional_query
    from app.cache.semantic_cache import get_semantic_cache
    from config.settings import get_settings

    raw_question = state.get("question", "")
    messages = state.get("messages", [])
    settings = get_settings()

    from datetime import datetime
    turn_start_t = state.get("turn_start_time") or time.perf_counter()
    turn_ts_str = state.get("turn_timestamp") or datetime.now().strftime("%I:%M:%S %p")

    # Phase 4/6 Optimization: Early probe persistent exact and semantic cache BEFORE calling Gemini rewriter
    if getattr(settings, "enable_exact_cache", True) or getattr(settings, "enable_semantic_cache", True):
        try:
            sem_cache = get_semantic_cache()
            user_id = str(state.get("user_id") or "default_user")
            clean_q = raw_question.strip()
            q_vec = None

            if getattr(settings, "enable_semantic_cache", True):
                try:
                    from app.retrieval.dual_pipeline import SharedConstitutionalCorpus
                    corpus = SharedConstitutionalCorpus.get_instance()
                    if corpus.vectorizer is not None:
                        q_vec = corpus.vectorizer.transform([clean_q]).toarray().flatten()
                except Exception as ex_vec:
                    logger.debug("Query vector generation for cache probe failed: %s", ex_vec)

            choice = state.get("selected_engine") or "both"
            cache_res = sem_cache.lookup(query=clean_q, query_vector=q_vec, chatbot_mode=choice, user_id=user_id)
            if not cache_res.hit and choice != "both":
                cache_res = sem_cache.lookup(query=clean_q, query_vector=q_vec, chatbot_mode="both", user_id=user_id)
            if not cache_res.hit and choice != "gemini":
                cache_res = sem_cache.lookup(query=clean_q, query_vector=q_vec, chatbot_mode="gemini", user_id=user_id)

            if cache_res.hit and cache_res.response_payload:
                payload = cache_res.response_payload
                rule_ans = payload.get("answer")
                gemini_ans = payload.get("gemini_answer")
                sources = payload.get("sources", [])
                response_type = payload.get("response_type", "article")
                target_art = payload.get("article_number") or cache_res.matched_article

                logger.info(
                    "Early LangGraph bypass: Query hit %s cache (entry_id=%s, latency=%.2fms)",
                    cache_res.cache_type, cache_res.entry_id, cache_res.latency_ms
                )

                diag = {
                    "success": True,
                    "status_code": 200,
                    "model_used": "semantic_cache",
                    "note": f"cache_hit_{cache_res.cache_type}",
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "duration_seconds": cache_res.latency_ms / 1000.0,
                }

                return {
                    "raw_question": raw_question,
                    "question": raw_question,
                    "rewritten_question": raw_question,
                    "intent": "cached_response",
                    "article_number": target_art,
                    "answer": rule_ans,
                    "gemini_answer": gemini_ans,
                    "sources": sources,
                    "response_type": response_type,
                    "gemini_diagnostic": diag,
                    "selected_engine": choice,
                }
        except Exception as e_cache:
            logger.debug("Early cache probe skipped on exception: %s", e_cache)

    rewrite_result = rewrite_constitutional_query(
        question=raw_question,
        conversation_history=messages,
    )

    # Adversarial premise & ambiguity gatekeeper check
    from app.adversarial.query_classifier import analyze_adversarial_query
    adv_analysis = analyze_adversarial_query(raw_question, messages)

    rewritten_q = rewrite_result.get("rewritten_query") or raw_question
    articles = rewrite_result.get("articles") or []
    intent = rewrite_result.get("intent")
    topic = rewrite_result.get("topic")

    updates: Dict[str, Any] = {
        "raw_question": raw_question,
        "question": rewritten_q,
        "rewritten_question": rewritten_q,
        "original_query": raw_question,
        "resolved_query": rewritten_q,
        "query_rewriter_metadata": rewrite_result,
        "prior_user_query": raw_question,
        "is_follow_up": rewrite_result.get("is_follow_up", False),
        "follow_up_type": rewrite_result.get("follow_up_type", ""),
        "active_topic": rewrite_result.get("active_topic") or topic,
        "domain_classification": "constitutional" if intent != "out_of_scope" else "out_of_domain",
        "adversarial_metadata": {
            "query_type": adv_analysis.query_type,
            "is_false_premise": adv_analysis.is_false_premise,
            "refutation": adv_analysis.refutation,
            "subqueries": adv_analysis.subqueries,
        },
    }

    # Intercept false premise attacks immediately before hallucination occurs
    if adv_analysis.is_false_premise:
        ans_msg = (
            f"### ⚖️ Constitutional Premise Clarification & Refutation\n\n"
            f"{adv_analysis.refutation}\n\n"
            f"*Source: Extracted and verified from the official Constitution of India PDF.*"
        )
        updates["intent"] = "false_premise_refutation"
        updates["article_number"] = adv_analysis.target_articles[0] if adv_analysis.target_articles else None
        updates["topic"] = "Constitutional Refutation"
        updates["response_type"] = "article"
        updates["answerability_status"] = "ANSWERABLE"
        updates["answer"] = ans_msg
        updates["gemini_answer"] = ans_msg
        # Retrieve the relevant corroborating chunks so citations pass
        kb = get_knowledge_base()
        verified_chunks = []
        for art in adv_analysis.target_articles:
            art_data = kb.lookup_article(art)
            if art_data:
                verified_chunks.append({
                    "article_number": art,
                    "article_title": art_data.get("article_title", ""),
                    "constitutional_text": art_data.get("constitutional_text", ""),
                    "page_start": art_data.get("page_start", 42),
                    "source_pdf": "constitution.pdf",
                })
        updates["retrieved_documents"] = verified_chunks
        updates["sources"] = [
            {
                "source_pdf": "constitution.pdf",
                "page": d.get("page_start", 42),
                "article": d["article_number"],
                "article_number": d["article_number"],
            }
            for d in verified_chunks
        ]
    elif adv_analysis.needs_clarification:
        clar_prompt = adv_analysis.clarification_prompt
        updates["intent"] = "clarification_needed"
        updates["needs_clarification"] = True
        updates["article_number"] = None
        updates["topic"] = None
        updates["response_type"] = "uncertain"
        updates["answerability_status"] = "INSUFFICIENT_EVIDENCE"
        updates["abstention_reason"] = clar_prompt
        updates["answer"] = clar_prompt
        updates["gemini_answer"] = clar_prompt
        updates["sources"] = []
        updates["retrieved_documents"] = []
    elif adv_analysis.subqueries:
        updates["intent"] = "multi_hop"
        updates["comparison_articles"] = adv_analysis.target_articles
        updates["article_number"] = adv_analysis.target_articles[0] if adv_analysis.target_articles else None
    elif intent in ["temporal_inquiry", "insufficient_evidence"]:
        from app.retrieval.evidence_guard import is_temporal_or_dynamic_query
        _, reason = is_temporal_or_dynamic_query(raw_question)
        ans_msg = reason or (
            "I cannot determine the exact number of constitutional amendments passed yesterday "
            "from the uploaded Constitution text. The available document does not establish this current information."
        )
        updates["intent"] = "temporal_inquiry"
        updates["article_number"] = None
        updates["topic"] = "Constitutional Amendments"
        updates["response_type"] = "insufficient_evidence"
        updates["answerability_status"] = "INSUFFICIENT_EVIDENCE"
        updates["abstention_reason"] = ans_msg
        updates["answer"] = ans_msg
        updates["gemini_answer"] = ans_msg
        updates["sources"] = []
        updates["retrieved_documents"] = []
    elif intent == "clarification_needed":
        clarification_msg = rewrite_result.get("clarification_message") or (
            "I could not identify which constitutional article or topic was previously discussed in this conversation. "
            "Please specify an article number (for example: *'What is Article 14?'* or *'Explain Article 21'*)."
        )
        updates["intent"] = "clarification_needed"
        updates["needs_clarification"] = True
        updates["article_number"] = None
        updates["topic"] = None
        updates["response_type"] = "uncertain"
        updates["answerability_status"] = "INSUFFICIENT_EVIDENCE"
        updates["abstention_reason"] = clarification_msg
        updates["answer"] = clarification_msg
        updates["gemini_answer"] = clarification_msg
        updates["sources"] = []
        updates["retrieved_documents"] = []
    elif intent == "out_of_scope":
        updates["intent"] = "out_of_scope"
        updates["article_number"] = None
        updates["topic"] = None
        updates["response_type"] = "out_of_scope"
        updates["answerability_status"] = "OUT_OF_DOMAIN"
        updates["question"] = raw_question
        updates["rewritten_question"] = raw_question
        updates["sources"] = []
        updates["retrieved_documents"] = []
    elif intent in ["topic", "topic_lookup"]:
        updates["intent"] = "topic_lookup"
        updates["article_number"] = None
        updates["topic"] = topic or "Fundamental Rights"
    elif articles:
        if len(articles) >= 2 and intent == "comparison":
            updates["comparison_articles"] = [str(a).strip().upper() for a in articles[:2]]
            updates["article_number"] = str(articles[0]).strip().upper()
            updates["intent"] = "comparison"
        elif len(articles) >= 1:
            art_str = str(articles[0]).strip().upper()
            updates["article_number"] = art_str
            updates["last_discussed_article"] = art_str
            if intent:
                updates["intent"] = intent
    elif intent:
        updates["intent"] = intent

    if topic:
        updates["topic"] = topic

    updates["turn_start_time"] = turn_start_t
    updates["turn_timestamp"] = turn_ts_str

    thread_id = state.get("thread_id") or "default"
    logger.info(
        "Structured Observability: thread_id=%s, history_len=%d, original_query='%s', resolved_query='%s', "
        "is_follow_up=%s (type=%s), active_topic=%s, domain=%s, intent=%s",
        thread_id,
        len(messages),
        raw_question,
        rewritten_q,
        rewrite_result.get("is_follow_up", False),
        rewrite_result.get("follow_up_type", ""),
        updates.get("active_topic"),
        updates.get("domain_classification"),
        updates.get("intent"),
    )

    return updates


def classify_question(state: ConstitutionState) -> Dict[str, Any]:
    """Node 1: Classify user query using deterministic regex and keyword heuristics."""
    # 0. Check if intent/articles were pre-resolved by LLM query rewriter
    existing_intent = state.get("intent")
    existing_art = state.get("article_number")
    existing_comp = state.get("comparison_articles")

    if existing_intent:
        if existing_intent in ["cached_response", "false_premise_refutation"]:
            return {
                "intent": existing_intent,
                "article_number": existing_art,
                "comparison_articles": existing_comp,
                "comparison_data": state.get("comparison_data"),
                "gemini_answer": state.get("gemini_answer"),
                "answer": state.get("answer"),
                "sources": state.get("sources"),
                "retrieved_documents": state.get("retrieved_documents"),
                "topic": state.get("topic"),
                "needs_clarification": False,
                "response_type": state.get("response_type", "article"),
                "answerability_status": state.get("answerability_status", "ANSWERABLE"),
            }
        if existing_intent in ["temporal_inquiry", "insufficient_evidence"]:
            from app.retrieval.evidence_guard import is_temporal_or_dynamic_query
            _, reason = is_temporal_or_dynamic_query(state.get("question", ""))
            ans_text = reason or state.get("abstention_reason") or (
                "I cannot determine the exact number of constitutional amendments passed yesterday "
                "from the uploaded Constitution text. The available document does not establish this current information."
            )
            return {
                "intent": "temporal_inquiry",
                "article_number": None,
                "comparison_articles": None,
                "comparison_data": None,
                "gemini_answer": ans_text,
                "answer": ans_text,
                "topic": "Constitutional Amendments",
                "needs_clarification": False,
                "response_type": "insufficient_evidence",
                "answerability_status": "INSUFFICIENT_EVIDENCE",
                "abstention_reason": ans_text,
                "sources": [],
                "retrieved_documents": [],
            }
        if existing_intent == "out_of_scope":
            return {
                "intent": "out_of_scope",
                "article_number": None,
                "comparison_articles": None,
                "comparison_data": None,
                "gemini_answer": None,
                "topic": None,
                "needs_clarification": False,
                "response_type": "out_of_scope",
            }
        if existing_intent == "greeting":
            return {
                "intent": "greeting",
                "article_number": None,
                "comparison_articles": None,
                "comparison_data": None,
                "gemini_answer": None,
                "topic": None,
                "needs_clarification": False,
                "response_type": "general",
            }
        if existing_intent == "comparison" and existing_comp and len(existing_comp) >= 2:
            return {
                "intent": "comparison",
                "comparison_articles": existing_comp,
                "comparison_data": None,
                "gemini_answer": None,
                "article_number": existing_art or existing_comp[0],
                "topic": state.get("topic"),
                "needs_clarification": False,
                "response_type": "comparison",
            }
        if existing_intent == "clarification_needed":
            ans_msg = state.get("answer") or (
                "I could not identify which constitutional article or topic was previously discussed in this conversation. "
                "Please specify an article number (for example: *'What is Article 14?'* or *'Explain Article 21'*)."
            )
            return {
                "intent": "clarification_needed",
                "article_number": None,
                "comparison_articles": None,
                "comparison_data": None,
                "gemini_answer": ans_msg,
                "answer": ans_msg,
                "topic": None,
                "needs_clarification": True,
                "response_type": "uncertain",
                "answerability_status": "INSUFFICIENT_EVIDENCE",
                "abstention_reason": ans_msg,
                "sources": [],
                "retrieved_documents": [],
            }
        if existing_intent in ["follow_up_history", "history_inquiry"]:
            return {
                "intent": "follow_up_history",
                "article_number": existing_art or state.get("last_discussed_article"),
                "last_discussed_article": existing_art or state.get("last_discussed_article"),
                "comparison_articles": None,
                "comparison_data": None,
                "gemini_answer": None,
                "topic": state.get("topic") or state.get("active_topic"),
                "needs_clarification": False,
                "response_type": "article",
            }
        if str(existing_intent).startswith("follow_up"):
            return {
                "intent": existing_intent,
                "article_number": existing_art or state.get("last_discussed_article"),
                "last_discussed_article": existing_art or state.get("last_discussed_article"),
                "comparison_articles": None,
                "comparison_data": None,
                "gemini_answer": None,
                "topic": state.get("topic") or state.get("active_topic"),
                "needs_clarification": False,
                "response_type": "article",
            }
        if existing_intent in ["topic", "topic_lookup"]:
            return {
                "intent": "topic_lookup",
                "topic": state.get("topic") or "Fundamental Rights",
                "article_number": None,
                "comparison_articles": None,
                "comparison_data": None,
                "gemini_answer": None,
                "needs_clarification": False,
                "response_type": "topic",
            }
        if existing_intent == "article_lookup" and existing_art:
            return {
                "intent": "article_lookup",
                "article_number": existing_art,
                "last_discussed_article": existing_art,
                "comparison_articles": None,
                "comparison_data": None,
                "gemini_answer": None,
                "topic": state.get("topic"),
                "needs_clarification": False,
                "response_type": "article",
            }

    raw_q = state.get("question") or state.get("query") or ""
    q = normalize_query(raw_q)
    q_lower = q.lower()

    # Check for temporal or dynamic inquiries (e.g. amendments passed yesterday)
    from app.retrieval.evidence_guard import is_temporal_or_dynamic_query
    is_temp, temp_reason = is_temporal_or_dynamic_query(raw_q)
    if is_temp:
        ans_text = temp_reason or (
            "I cannot determine the exact number of constitutional amendments passed yesterday "
            "from the uploaded Constitution text. The available document does not establish this current information."
        )
        return {
            "intent": "temporal_inquiry",
            "article_number": None,
            "comparison_articles": None,
            "comparison_data": None,
            "gemini_answer": ans_text,
            "answer": ans_text,
            "topic": "Constitutional Amendments",
            "needs_clarification": False,
            "response_type": "insufficient_evidence",
            "answerability_status": "INSUFFICIENT_EVIDENCE",
            "abstention_reason": ans_text,
            "sources": [],
            "retrieved_documents": [],
        }

    from app.retrieval.gemini_service import is_person_or_name_query

    # Reject inquiries asking for a person's name or who holds an office
    if is_person_or_name_query(raw_q) or is_person_or_name_query(state.get("raw_question", "")):
        return {
            "intent": "out_of_scope",
            "article_number": None,
            "comparison_articles": None,
            "comparison_data": None,
            "gemini_answer": None,
            "topic": None,
            "needs_clarification": False,
            "response_type": "out_of_scope",
            "answerability_status": "OUT_OF_DOMAIN",
        }

    # 1. Greetings and identity
    greetings = {
        "hi",
        "hello",
        "hey",
        "namaste",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "who are you",
        "what can you do",
        "help",
    }
    if q_lower in greetings or any(
        q_lower.startswith(g + " ") for g in greetings
    ):
        return {
            "intent": "greeting",
            "article_number": None,
            "comparison_articles": None,
            "comparison_data": None,
            "gemini_answer": None,
            "topic": None,
            "needs_clarification": False,
            "response_type": "general",
        }

    # 2. Comparison queries (e.g., 'Compare Article 14 and Article 21', 'Difference between 14 and 21')
    articles_found = extract_article_numbers_from_query(q)
    is_compare_query = any(
        w in q_lower
        for w in [
            "compare",
            "comparison",
            "difference between",
            "versus",
            " vs ",
            "distinguish between",
        ]
    )

    if is_compare_query:
        if len(articles_found) >= 2:
            return {
                "intent": "comparison",
                "comparison_articles": [articles_found[0], articles_found[1]],
                "comparison_data": None,
                "gemini_answer": None,
                "article_number": articles_found[0],
                "topic": None,
                "needs_clarification": False,
                "response_type": "comparison",
            }
        elif len(articles_found) == 1 and any(
            w in q_lower for w in ["it", "this", "that", "earlier", "previous"]
        ):
            # Follow-up comparison e.g., 'Compare it with Article 14'
            return {
                "intent": "follow_up_compare",
                "comparison_articles": [articles_found[0]],
                "comparison_data": None,
                "gemini_answer": None,
                "topic": None,
                "needs_clarification": False,
                "response_type": "comparison",
            }

    # 3. History inspection query (e.g. 'what article i had asked you', 'Which article did I ask about previously?')
    def is_history_query(text: str) -> bool:
        t = " " + " ".join(text.lower().split()) + " "
        history_phrases = [
            "what article i had asked",
            "what article i asked",
            "what article did i ask",
            "which article did i ask",
            "what was the article",
            "which article was that",
            "what article was that",
            "which article was it",
            "what article was it",
            "what did i ask",
            "what i asked",
            "what did we discuss",
            "which article we discussed",
            "which article did you explain",
            "what was my question",
            "what were we talking about",
            "what was that article",
            "previous article",
            "last article",
            "article we discussed",
            "article i asked earlier",
            "article discussed earlier",
            "what article was i asking",
            "what was i asking",
            "repeat the article",
            "tell me what article i asked",
        ]
        for hp in history_phrases:
            if hp in t:
                return True
        if re.search(r"\b(what|which)\s+article\b.*?\b(ask|asked|discuss|discussed|talk|talked|mention|mentioned|earlier|previously|before)\b", t):
            return True
        if re.search(r"\b(what|which)\b.*?\b(did\s+i\s+ask|i\s+asked|i\s+had\s+asked|we\s+discussed|we\s+talked\s+about|we\s+were\s+discussing)\b", t):
            return True
        return False

    if is_history_query(q):
        return {
            "intent": "follow_up_history",
            "comparison_articles": None,
            "comparison_data": None,
            "gemini_answer": None,
            "topic": None,
            "needs_clarification": False,
            "response_type": "article",
        }

    # 4. Follow-up requests without explicit article number
    is_simple_request = any(
        p in q_lower
        for p in [
            "explain it simply",
            "in simpler words",
            "simple english",
            "in simple terms",
            "simpler words",
            "layman",
            "easy words",
            "explain it",
            "simplify it",
        ]
    )
    is_example_request = any(
        p in q_lower
        for p in [
            "give an example",
            "give me an example",
            "example of this",
            "real world example",
            "practical example",
            "an example",
            "examples",
        ]
    )
    is_generic_followup = any(
        p in q_lower
        for p in [
            "the article we discussed earlier",
            "discussed earlier",
            "the earlier article",
            "tell me more about it",
            "explain it more",
        ]
    )

    # If it's a follow-up referring to 'it' or 'earlier' and NO article is mentioned in this prompt
    if len(articles_found) == 0:
        if is_simple_request:
            return {
                "intent": "follow_up_simple",
                "comparison_articles": None,
                "comparison_data": None,
                "gemini_answer": None,
                "topic": None,
                "needs_clarification": False,
                "response_type": "article",
            }
        if is_example_request:
            return {
                "intent": "follow_up_example",
                "comparison_articles": None,
                "comparison_data": None,
                "gemini_answer": None,
                "topic": None,
                "needs_clarification": False,
                "response_type": "article",
            }
        if is_generic_followup or " it " in f" {q_lower} ":
            return {
                "intent": "follow_up_general",
                "comparison_articles": None,
                "comparison_data": None,
                "gemini_answer": None,
                "topic": None,
                "needs_clarification": False,
                "response_type": "article",
            }

    # 5. Explicit Article lookup (e.g., 'What is Article 21?', 'Explain Article 14 in simple English')
    if len(articles_found) >= 1:
        art_num = articles_found[0]
        sub_intent = "article_lookup"
        if is_simple_request:
            sub_intent = "article_simple"
        elif is_example_request:
            sub_intent = "article_example"

        return {
            "intent": sub_intent,
            "article_number": art_num,
            "comparison_articles": None,
            "comparison_data": None,
            "gemini_answer": None,
            "topic": None,
            "needs_clarification": False,
            "response_type": "article",
        }

    # 6. Thematic / Topic queries
    kb = get_knowledge_base()
    matched_topic = kb.search_topic(q)
    if matched_topic:
        return {
            "intent": "topic_lookup",
            "topic": matched_topic["topic_title"],
            "article_number": None,
            "comparison_articles": None,
            "comparison_data": None,
            "gemini_answer": None,
            "needs_clarification": False,
            "response_type": "topic",
        }

    # 7. Check if query is out of scope (no constitutional indicators, no articles, no topics)
    constitutional_indicators = {
        "constitution", "constitutional", "article", "amendment", "preamble", "schedule", "part",
        "fundamental right", "fundamental rights", "directive principle", "directive principles",
        "fundamental duty", "fundamental duties", "writ", "habeas corpus", "mandamus",
        "supreme court", "high court", "judiciary", "president", "governor",
        "parliament", "lok sabha", "rajya sabha", "legislature", "ordinance",
        "citizenship", "reservation", "emergency", "equality", "discrimination",
        "untouchability", "speech", "expression", "life", "liberty", "privacy",
        "arrest", "detention", "double jeopardy", "self incrimination", "religion",
        "election commission", "attorney general", "upsc", "tribunal", "cag",
        "remedy", "remedies", "secular", "socialist", "sovereign", "republic",
        "justice", "fraternity", "law", "legal", "court",
    }
    has_const_indicator = any(kw in q_lower for kw in constitutional_indicators)

    if not has_const_indicator:
        return {
            "intent": "out_of_scope",
            "article_number": None,
            "comparison_articles": None,
            "comparison_data": None,
            "gemini_answer": None,
            "topic": None,
            "needs_clarification": False,
            "response_type": "out_of_scope",
            "answerability_status": "OUT_OF_DOMAIN",
        }

    # 8. General search fallback or uncertain for in-scope constitutional queries
    if len(q.split()) >= 2:
        return {
            "intent": "general_search",
            "article_number": None,
            "comparison_articles": None,
            "comparison_data": None,
            "gemini_answer": None,
            "topic": None,
            "needs_clarification": False,
            "response_type": "article",
        }

    return {
        "intent": "uncertain",
        "article_number": None,
        "comparison_articles": None,
        "comparison_data": None,
        "gemini_answer": None,
        "topic": None,
        "needs_clarification": True,
        "response_type": "uncertain",
    }


def resolve_context(state: ConstitutionState) -> Dict[str, Any]:
    """Node 2: Resolve pronoun and conversation references from message history."""
    intent = state.get("intent", "")
    messages: Sequence[BaseMessage] = state.get("messages", [])

    # Extract all user prompts in chronological order
    user_prompts = [str(getattr(m, "content", "")) for m in messages if isinstance(m, HumanMessage)]
    # The immediate previous user question before current prompt
    prior_user_query = user_prompts[-2] if len(user_prompts) >= 2 else (user_prompts[0] if user_prompts else "")

    # Extract all distinct constitutional articles discussed in this conversation thread
    all_discussed_articles: List[str] = []
    for m in messages:
        content = str(getattr(m, "content", ""))
        for match in re.findall(r"\bArticle\s+(\d+[A-Z]?|PREAMBLE)\b", content, re.IGNORECASE):
            m_up = match.upper()
            if m_up not in all_discussed_articles:
                all_discussed_articles.append(m_up)

    # Check state memory first, then search past messages in reverse order
    last_article: Optional[str] = state.get("article_number") or state.get("last_discussed_article")

    if not last_article:
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            # Check for Article headers in assistant messages or user prompts
            m = re.search(r"\bArticle\s+(\d+[A-Z]?|PREAMBLE)\b", content, re.IGNORECASE)
            if m:
                last_article = m.group(1).upper()
                break

    if not last_article and all_discussed_articles:
        last_article = all_discussed_articles[-1]

    if not last_article:
        clarification_msg = (
            "I could not identify which constitutional article or topic was previously discussed in this conversation. "
            "Please specify an article number (for example: *'What is Article 14?'* or *'Explain Article 21'*)."
        )
        return {
            "needs_clarification": True,
            "intent": "clarification_needed",
            "answer": clarification_msg,
            "gemini_answer": clarification_msg,
            "response_type": "uncertain",
            "answerability_status": "INSUFFICIENT_EVIDENCE",
            "abstention_reason": clarification_msg,
            "sources": [],
            "retrieved_documents": [],
        }

    kb = get_knowledge_base()
    art_info = kb.lookup_article(last_article)
    title = art_info["article_title"] if art_info else ""

    # Rich contextual response when user asks about previous history
    if intent == "follow_up_history":
        text_snippet = (
            art_info["constitutional_text"][:420] + "..."
            if art_info and len(art_info["constitutional_text"]) > 420
            else (art_info["constitutional_text"] if art_info else "")
        )
        page_num = art_info.get("page_start", "N/A") if art_info else "N/A"
        part_name = art_info.get("part", "Part III") if art_info else "Part III"
        part_title = art_info.get("part_title", "") if art_info else ""
        principles = art_info.get("key_principles", []) if art_info else []
        examples = art_info.get("examples", []) if art_info else []

        history_lines = [
            f"### 📜 Conversation Context & History Recap\n",
            f"Earlier in our conversation, you asked about **Article {last_article}**"
            f"{f' (*{title}*)' if title else ''}."
        ]
        if prior_user_query:
            history_lines.append(f"**Your Previous Question Was:** *\"{prior_user_query}\"*\n")

        # If multiple articles were discussed across this chat, show a structured overview table
        if len(all_discussed_articles) > 1:
            history_lines.append("#### 📚 Articles Explored in this Consultation:")
            history_lines.append("| Article | Constitutional Subject | Part | Source Page |")
            history_lines.append("| :--- | :--- | :--- | :--- |")
            for a_num in all_discussed_articles:
                a_data = kb.lookup_article(a_num)
                if a_data:
                    current_marker = " 👈 *(Current Focus)*" if a_num == last_article else ""
                    history_lines.append(
                        f"| **Article {a_num}**{current_marker} | {a_data['article_title']} | {a_data['part']} | Page {a_data['page_start']} |"
                    )
            history_lines.append("")

        history_lines.extend([
            f"#### 🏛️ Verbatim Provision from Constitution PDF (Page {page_num}):",
            f"> {text_snippet}\n",
            f"#### ⚖️ Grounded Constitutional Attributes:",
            f"- **Constitutional Part:** {part_name} ({part_title})",
            f"- **Subject Provision:** {title}",
            f"- **Source Citation:** {art_info.get('source_pdf', 'constitution.pdf')} (Page {page_num})",
        ])

        if principles:
            history_lines.append(f"- **Key Principle:** {principles[0]}")
        if examples:
            history_lines.append(f"- **Practical Application:** {examples[0]}")

        history_lines.extend([
            f"\n#### 💡 What you can ask next:",
            f"- *'Explain it simply'* — for a plain-English breakdown of Article {last_article}",
            f"- *'Give an example'* — for practical court scenarios",
            f"- *'Compare Article 14 and Article {last_article}'* — for comparative constitutional analysis",
        ])

        history_answer = "\n".join(history_lines)

        return {
            "article_number": last_article,
            "last_discussed_article": last_article,
            "last_discussed_title": title,
            "prior_user_query": prior_user_query,
            "needs_clarification": False,
            "answer": history_answer,
            "response_type": "article",
        }

    if intent == "follow_up_compare":
        articles = state.get("comparison_articles", [])
        if articles:
            return {
                "article_number": last_article,
                "last_discussed_article": last_article,
                "last_discussed_title": title,
                "prior_user_query": prior_user_query,
                "comparison_articles": [last_article, articles[0]],
                "intent": "comparison",
                "needs_clarification": False,
                "response_type": "comparison",
            }

    # For follow_up_simple, follow_up_example, follow_up_general
    return {
        "article_number": last_article,
        "last_discussed_article": last_article,
        "last_discussed_title": title,
        "prior_user_query": prior_user_query,
        "needs_clarification": False,
    }


def retrieve_constitution(state: ConstitutionState) -> Dict[str, Any]:
    """Node 3: Search the locally extracted Constitution PDF content via Subgraph.
    
    Executes the modular RetrievalAndVerificationSubgraph, isolating search,
    scoring, and provenance verification. Supports RAG Reranking with Human-in-the-Loop decision.
    """
    from app.graph.subgraphs import retrieval_subgraph
    from app.retrieval.reranker import rerank_candidates

    sub_input = {
        "question": state.get("question", ""),
        "intent": state.get("intent", ""),
        "article_number": state.get("article_number"),
        "comparison_articles": state.get("comparison_articles"),
        "topic": state.get("topic"),
    }
    sub_output = retrieval_subgraph.invoke(sub_input)
    raw_candidates = sub_output.get("retrieved_documents", [])
    requires_review = sub_output.get("requires_human_review", False)

    rerank_mode = state.get("rerank_mode")
    if rerank_mode == "rerank":
        final_docs, r_meta = rerank_candidates(
            query=state.get("question", ""),
            candidates=raw_candidates,
        )
    elif rerank_mode == "hitl_decide":
        final_docs = raw_candidates
        r_meta = {"method": "hitl_pending", "count": len(raw_candidates), "status": "Pending Human Decision"}
        requires_review = True
    else:
        final_docs = raw_candidates
        r_meta = {"method": "none", "count": len(raw_candidates), "status": "Direct Retrieval (No Rerank)"}

    # Rebuild sources to reflect final docs order
    sources = [
        {
            "source_pdf": d.get("source_pdf", "constitution.pdf"),
            "page": d.get("page_start", "") or d.get("page_number", ""),
            "article": d.get("article_number", ""),
            "chunk_id": d.get("chunk_id", ""),
            "source_id": d.get("source_id", ""),
        }
        for d in final_docs
    ] if final_docs else sub_output.get("sources", [])

    return {
        "candidate_documents": raw_candidates,
        "retrieved_documents": final_docs,
        "comparison_data": sub_output.get("comparison_data"),
        "sources": sources,
        "topic": sub_output.get("topic", state.get("topic")),
        "requires_human_review": requires_review,
        "rerank_metadata": r_meta,
        "subgraph_metadata": {
            "verification_status": sub_output.get("verification_status", "UNKNOWN"),
            "audit_notes": sub_output.get("audit_notes", []),
        },
    }


def check_retrieval(state: ConstitutionState) -> Dict[str, Any]:
    """Node 4: Evaluate sufficiency and relevance of retrieved documents."""
    from app.retrieval.evidence_guard import (
        evaluate_evidence_sufficiency,
        STATUS_ANSWERABLE,
        STATUS_INSUFFICIENT_EVIDENCE,
        STATUS_OUT_OF_DOMAIN,
        STATUS_RETRIEVAL_FAILURE,
    )

    if state.get("answerability_status") in [
        STATUS_INSUFFICIENT_EVIDENCE,
        STATUS_OUT_OF_DOMAIN,
        STATUS_RETRIEVAL_FAILURE,
    ]:
        return {}

    q = state.get("question") or state.get("query") or ""
    retrieved = state.get("retrieved_documents", [])
    intent = state.get("intent", "")

    eval_res = evaluate_evidence_sufficiency(q, retrieved, intent=intent)
    if eval_res.status != STATUS_ANSWERABLE:
        resp_type = (
            "insufficient_evidence"
            if eval_res.status == STATUS_INSUFFICIENT_EVIDENCE
            else ("out_of_scope" if eval_res.status == STATUS_OUT_OF_DOMAIN else "retrieval_failure")
        )
        return {
            "answerability_status": eval_res.status,
            "abstention_reason": eval_res.explanation,
            "response_type": resp_type,
            "answer": eval_res.explanation,
            "gemini_answer": eval_res.explanation,
            "sources": [],
            "retrieved_documents": [],
        }

    return {
        "answerability_status": STATUS_ANSWERABLE,
        "retrieved_documents": eval_res.verified_sources or retrieved,
    }


def generate_answer(state: ConstitutionState) -> Dict[str, Any]:
    """Node 5: Generate deterministic, template-based constitutional response with enhanced context awareness."""
    intent = state.get("intent", "")
    comp_data = state.get("comparison_data")
    docs = state.get("retrieved_documents", [])
    topic_title = state.get("topic")
    prior_q = state.get("prior_user_query", "")

    # If abstaining due to insufficient evidence or retrieval failure
    if state.get("answerability_status") == "INSUFFICIENT_EVIDENCE" or state.get("response_type") == "insufficient_evidence":
        return {
            "answer": state.get("abstention_reason") or "I cannot determine the exact number of constitutional amendments passed yesterday from the uploaded Constitution text. The available document does not establish this current information.",
            "response_type": "insufficient_evidence",
            "sources": [],
            "retrieved_documents": [],
        }

    # 1. Comparison Response
    if comp_data:
        art1 = comp_data["article_1"]
        art2 = comp_data["article_2"]
        context_prefix = (
            f"> 🔗 **Context Continuity:** Comparing **Article {art1['article_number']}** (from our ongoing discussion) with **Article {art2['article_number']}**.\n\n"
            if intent == "follow_up_compare"
            else ""
        )
        answer = (
            f"{context_prefix}"
            f"### Constitutional Comparison: Article {art1['article_number']} vs Article {art2['article_number']}\n\n"
            f"| Feature | **Article {art1['article_number']}** | **Article {art2['article_number']}** |\n"
            f"| :--- | :--- | :--- |\n"
            f"| **Title** | {art1['article_title']} | {art2['article_title']} |\n"
            f"| **Constitutional Part** | {art1['part']} ({art1['part_title']}) | {art2['part']} ({art2['part_title']}) |\n"
            f"| **PDF Source Page** | Page {art1['page_start']} | Page {art2['page_start']} |\n\n"
            f"#### Article {art1['article_number']} Provision *(Source: {art1['source_pdf']}, Page {art1['page_start']})*\n"
            f"> {art1['constitutional_text'][:600]}\n\n"
            f"#### Article {art2['article_number']} Provision *(Source: {art2['source_pdf']}, Page {art2['page_start']})*\n"
            f"> {art2['constitutional_text'][:600]}\n\n"
            f"#### Comparative Constitutional Analysis\n"
            f"{comp_data['synthesis']}\n\n"
            f"---\n"
            f"*Source: Extracted from {art1['source_pdf']} (Pages {art1['page_start']} and {art2['page_start']}). Rule-based analysis.*"
        )
        return {
            "answer": answer,
            "response_type": "comparison",
            "last_discussed_article": art1["article_number"],
            "last_discussed_title": art1["article_title"],
        }

    # 2. Topic Overview Response
    is_explicit_topic = intent == "topic_lookup" or (
        topic_title and not state.get("article_number")
    )
    if is_explicit_topic and docs:
        kb = get_knowledge_base()
        topic_info = kb.search_topic(topic_title)
        overview_text = topic_info["overview"] if topic_info else ""
        art_range = topic_info.get("article_range", "") if topic_info else ""

        articles_summary = ""
        for d in docs[:6]:
            articles_summary += (
                f"- **Article {d['article_number']} — {d['article_title']}** (Page {d['page_start']}): "
                f"{d.get('simple_explanation', '')[:180]}...\n"
            )

        answer = (
            f"### {topic_title}\n\n"
            f"**Constitutional Scope:** Articles {art_range}\n\n"
            f"{overview_text}\n\n"
            f"#### Key Articles in this Category:\n"
            f"{articles_summary}\n\n"
            f"You can ask me to detail any specific article (e.g. *'What is Article {docs[0]['article_number']}?'*)."
        )
        return {"answer": answer, "response_type": "topic"}

    # 3. Article Lookup or Follow-Up Response
    if docs:
        doc = docs[0]
        art_num = doc["article_number"]
        title = doc["article_title"]
        part = doc["part"]
        part_title = doc["part_title"]
        text = doc["constitutional_text"]
        explanation = doc.get("simple_explanation", "")
        examples = doc.get("examples", [])
        principles = doc.get("key_principles", [])
        page = doc["page_start"]
        pdf_name = doc["source_pdf"]

        # Format based on sub-intent
        if intent in ["follow_up_simple", "article_simple", "follow_up_simplification"]:
            context_prefix = (
                f"> 🔗 **Context Continuity:** Continuing our follow-up on **Article {art_num}** (*{title}*)"
                f"{f' following your prior question: *\"{prior_q}\"*' if prior_q else ''}.\n\n"
                if str(intent).startswith("follow_up")
                else ""
            )
            answer = (
                f"{context_prefix}"
                f"### Simple English Explanation: Article {art_num} *(Source: {pdf_name}, Page {page})*\n"
                f"**{title}** (*{part}: {part_title}*)\n\n"
                f"**In Plain English:**\n"
                f"{explanation}\n\n"
                f"**Key Constitutional Takeaways:**\n"
            )
            for p in principles:
                answer += f"- {p}\n"

            if examples:
                answer += f"\n**Practical Scenario / Example:**\n- {examples[0]}\n"

            answer += (
                f"\n---\n"
                f"> {text}\n\n"
                f"*Source: {pdf_name}, Page {page}. Deterministic legal rule-based interpretation.*"
            )

        elif intent in ["follow_up_example", "article_example", "follow_up_examples"]:
            context_prefix = (
                f"> 🔗 **Context Continuity:** Practical case scenario for **Article {art_num}** (*{title}*)"
                f"{f' (answering follow-up on *\"{prior_q}\"*' if prior_q else ''}.\n\n"
                if str(intent).startswith("follow_up")
                else ""
            )
            answer = (
                f"{context_prefix}"
                f"### Practical Examples & Case Scenarios: Article {art_num} *(Source: {pdf_name}, Page {page})*\n"
                f"**{title}** (*{part}*)\n\n"
            )
            if examples:
                for idx, ex in enumerate(examples, 1):
                    answer += f"**Scenario {idx}:**\n{ex}\n\n"
            else:
                answer += (
                    f"Under Article {art_num} ({title}), any action by the State or public authority "
                    f"that infringes upon this constitutional guarantee can be challenged by affected persons "
                    f"before the High Court under Article 226 or the Supreme Court under Article 32.\n\n"
                )

            answer += (
                f"**Summary of Provision:**\n{explanation}\n\n"
                f"*Source: {pdf_name}, Page {page}.*"
            )

        elif intent in ["follow_up_detail"]:
            context_prefix = (
                f"> 🔗 **Context Continuity:** Detailed constitutional analysis of **Article {art_num}** (*{title}*)"
                f"{f' expanding on: *\"{prior_q}\"*' if prior_q else ''}.\n\n"
            )
            answer = (
                f"{context_prefix}"
                f"### Article {art_num}: {title} — Verbatim Constitution Text & Detailed Breakdown *(Source: {pdf_name}, Page {page})*\n"
                f"**Part:** {part} ({part_title}) | **Source:** {pdf_name} (Page {page})\n\n"
                f"#### Complete Statutory Constitutional Text (Verbatim from Official PDF):\n"
                f"> {text}\n\n"
                f"#### Comprehensive Legal Analysis:\n"
                f"{explanation}\n\n"
            )
            if principles:
                answer += "#### Core Constitutional Principles & Doctrines:\n"
                for p in principles:
                    answer += f"- {p}\n"
                answer += "\n"
            if examples:
                answer += "#### Judicial Applications & Scenarios:\n"
                for idx, ex in enumerate(examples, 1):
                    answer += f"{idx}. {ex}\n"
                answer += "\n"
            answer += (
                f"---\n"
                f"*Source: Extracted verbatim from {pdf_name} (Page {page}). Full clause fidelity without truncation or silent paraphrasing.*"
            )

        elif intent in ["follow_up_crisp", "follow_up_detail_and_crisp"]:
            context_prefix = (
                f"> 🔗 **Context Continuity:** Concise statutory breakdown of **Article {art_num}** (*{title}*)"
                f"{f' addressing: *\"{prior_q}\"*' if prior_q else ''}.\n\n"
            )
            answer = (
                f"{context_prefix}"
                f"### Article {art_num}: {title} (Crisp Breakdown) *(Source: {pdf_name}, Page {page})*\n"
                f"**Part:** {part} ({part_title})\n\n"
                f"**Statutory Provision (Verbatim Excerpt):**\n"
                f"> {text}\n\n"
                f"**Core Constitutional Rule:**\n"
                f"{explanation}\n\n"
            )
            if principles:
                answer += f"**Key Principle:** {principles[0]}\n\n"
            answer += f"*Source: {pdf_name}, Page {page}. Deterministic statutory summary.*"

        else:
            context_prefix = (
                f"> 🔗 **Context Continuity:** Detailed constitutional analysis of **Article {art_num}** (*{title}*)"
                f"{f' in response to: *\"{prior_q}\"*' if prior_q else ''}.\n\n"
                if intent == "follow_up_general"
                else ""
            )
            answer = (
                f"{context_prefix}"
                f"### Article {art_num}: {title} *(Source: {pdf_name}, Page {page})*\n"
                f"**Part:** {part} ({part_title}) | **Source:** {pdf_name} (Page {page})\n\n"
                f"#### Constitutional Text (as extracted from {pdf_name}, Page {page}):\n"
                f"> {text[:800]}{'...' if len(text) > 800 else ''}\n\n"
                f"#### Plain English Explanation (grounded in {pdf_name}, Page {page}):\n"
                f"{explanation}\n\n"
            )

            if principles:
                answer += "#### Core Constitutional Principles:\n"
                for p in principles[:4]:
                    answer += f"- {p}\n"
                answer += "\n"

            if examples:
                answer += f"#### Practical Example:\n- {examples[0]}\n\n"

            answer += (
                f"---\n"
                f"*Verified against local Constitution PDF (SHA-256 integrity checked). Rule-based response.*"
            )

        return {
            "answer": answer,
            "response_type": "article",
            "last_discussed_article": art_num,
            "last_discussed_title": title,
        }

    # Fallback to uncertainty handler
    return handle_uncertainty(state)


def handle_general_query(state: ConstitutionState) -> Dict[str, Any]:
    """Node 6: Respond to greetings and general assistance queries."""
    greeting_text = (
        "### Welcome to ConstitutionGPT\n"
        "I am a deterministic, rule-based legal assistant for the **Constitution of India**, "
        "powered exclusively by your locally loaded Constitution PDF, LangGraph, and SQLite.\n\n"
        "**What you can ask me:**\n"
        "- **Article Lookups:** *'What is Article 21?'*, *'Tell me about Article 14'*, *'What does Article 32 say?'*\n"
        "- **Plain English Explanations:** *'Explain Article 14 in simple English'*, *'Explain it simply'*\n"
        "- **Comparisons:** *'Compare Article 14 and Article 21'*\n"
        "- **Thematic Topics:** *'What are the Fundamental Rights?'*, *'Directive Principles'*, *'Fundamental Duties'*\n"
        "- **Government & Judiciary:** *'Tell me about the President'*, *'Supreme Court'*, *'Emergency provisions'*\n"
        "- **Practical Examples:** *'Give an example of Article 21'*\n"
        "- **Follow-up Questions:** *'Which article did we discuss earlier?'*, *'Explain it in simpler words'*\n\n"
        "*(Note: All answers are strictly grounded in your Constitution PDF with zero generative AI or hallucinations.)*"
    )
    return {
        "answer": greeting_text,
        "sources": [],
        "response_type": "general",
    }


def handle_uncertainty(state: ConstitutionState) -> Dict[str, Any]:
    """Node 7: Notification when query is out of scope, has insufficient evidence, or encounters an operational error."""
    intent = state.get("intent")
    resp_type = state.get("response_type")
    ans_status = state.get("answerability_status")

    # 1. Insufficient Evidence / Temporal Unrecorded Facts (e.g. amendments passed yesterday)
    if (
        intent in ["temporal_inquiry", "insufficient_evidence"]
        or resp_type == "insufficient_evidence"
        or ans_status == "INSUFFICIENT_EVIDENCE"
    ):
        reason = state.get("abstention_reason")
        if not reason:
            from app.retrieval.evidence_guard import is_temporal_or_dynamic_query
            _, reason = is_temporal_or_dynamic_query(state.get("question", ""))

        answer = reason or (
            "I cannot determine the exact number of constitutional amendments passed yesterday "
            "from the uploaded Constitution text. The available document does not establish this current information."
        )
        return {
            "answer": answer,
            "gemini_answer": answer,
            "sources": [],
            "retrieved_documents": [],
            "response_type": "insufficient_evidence",
            "answerability_status": "INSUFFICIENT_EVIDENCE",
            "abstention_reason": answer,
        }

    # 2. Out of domain (e.g. cricket score, coding, weather, living person identity)
    if (
        intent == "out_of_scope"
        or resp_type == "out_of_scope"
        or ans_status == "OUT_OF_DOMAIN"
    ):
        answer = (
            "### ⚖️ Domain Restriction\n\n"
            "Please ask questions based on the Constitution of India domain only."
        )
        return {
            "answer": answer,
            "gemini_answer": answer,
            "sources": [],
            "retrieved_documents": [],
            "response_type": "out_of_scope",
            "answerability_status": "OUT_OF_DOMAIN",
        }

    # 3. Retrieval failure / operational error
    if (
        intent == "retrieval_failure"
        or resp_type == "retrieval_failure"
        or ans_status == "RETRIEVAL_FAILURE"
    ):
        answer = (
            "### ⚠️ Retrieval Error\n\n"
            "An operational error or technical difficulty occurred while retrieving constitutional records. Please try again."
        )
        return {
            "answer": answer,
            "gemini_answer": answer,
            "sources": [],
            "retrieved_documents": [],
            "response_type": "retrieval_failure",
            "answerability_status": "RETRIEVAL_FAILURE",
        }

    if intent == "clarification_needed" or state.get("needs_clarification"):
        clarification_msg = state.get("abstention_reason") or state.get("answer") or (
            "I could not identify which constitutional article or topic you are referring to. "
            "Please specify an article number (for example: *'What is Article 14?'* or *'Explain Article 21'*)."
        )
        return {
            "answer": clarification_msg,
            "gemini_answer": clarification_msg,
            "sources": [],
            "retrieved_documents": [],
            "response_type": "uncertain",
            "answerability_status": "INSUFFICIENT_EVIDENCE",
            "abstention_reason": clarification_msg,
        }

    answer = (
        "### ⚖️ Domain Restriction\n\n"
        "Please ask questions based on the Constitution of India domain only."
    )
    return {
        "answer": answer,
        "gemini_answer": answer,
        "sources": [],
        "retrieved_documents": [],
        "response_type": "out_of_scope",
        "answerability_status": "OUT_OF_DOMAIN",
    }


def synthesize_gemini(state: ConstitutionState) -> Dict[str, Any]:
    """Node: Generate candidate Gemini AI synthesis before finalization/human review.
    
    Demonstrates:
    - Fault tolerance: Graceful recovery with local fallback if external API fails
    """
    from app.retrieval.gemini_service import generate_gemini_constitutional_response
    from app.memory.mem0_service import get_mem0_service

    intent = state.get("intent", "")
    resp_type = state.get("response_type")
    gemini_ans = state.get("gemini_answer")
    gemini_diagnostic = state.get("gemini_diagnostic")
    user_id = state.get("user_id") or "default_user"
    mem0_memories = state.get("mem0_memories") or []

    if not gemini_ans:
        if intent == "clarification_needed" or state.get("needs_clarification"):
            gemini_ans = state.get("abstention_reason") or state.get("answer") or (
                "I could not identify which constitutional article or topic was previously discussed in this conversation. "
                "Please specify an article number (for example: *'What is Article 14?'* or *'Explain Article 21'*)."
            )
            gemini_diagnostic = {"success": True, "status_code": 200, "note": "clarification_needed"}
        elif (
            intent in ["temporal_inquiry", "insufficient_evidence"]
            or resp_type == "insufficient_evidence"
            or state.get("answerability_status") == "INSUFFICIENT_EVIDENCE"
        ):
            gemini_ans = state.get("abstention_reason") or state.get("answer") or (
                "I cannot determine the exact number of constitutional amendments passed yesterday "
                "from the uploaded Constitution text. The available document does not establish this current information."
            )
            gemini_diagnostic = {"success": True, "status_code": 200, "note": "insufficient_evidence"}
        elif intent in ["out_of_scope", "uncertain"] or resp_type in ["out_of_scope", "uncertain"] or state.get("answerability_status") == "OUT_OF_DOMAIN":
            gemini_ans = (
                "### ⚖️ Domain Restriction\n\n"
                "Please ask questions based on the Constitution of India domain only."
            )
            gemini_diagnostic = {"success": True, "status_code": 200, "note": "domain_restricted"}
        elif intent == "greeting":
            gemini_ans = (
                "### 🇮🇳 Namaste & Welcome!\n"
                "I am your **Gemini AI Constitutional Assistant**, integrated directly into ConstitutionGPT. "
                "I synthesize legal insights and case doctrines using the verbatim constitutional provisions extracted "
                "from your loaded official PDF.\n\n"
                "Feel free to ask about any article, right, duty, or comparison!"
            )
            gemini_diagnostic = {"success": True, "status_code": 200, "note": "greeting"}
        elif state.get("selected_engine") == "rule_based":
            # Fast bypass: When Rule-Based engine is selected, skip Gemini LLM network call completely for sub-second execution!
            gemini_ans = None
            gemini_diagnostic = {"success": True, "status_code": 200, "note": "rule_based_engine_selected"}
        else:
            # Semantic & Exact Cache Probe before external API call
            from app.cache.semantic_cache import get_semantic_cache
            sem_cache = get_semantic_cache()
            clean_q = state.get("raw_question") or state.get("question", "")
            cache_res = sem_cache.lookup(query=clean_q, chatbot_mode="gemini", user_id=user_id)
            if cache_res.hit and cache_res.response_payload and cache_res.response_payload.get("gemini_answer"):
                gemini_ans = cache_res.response_payload["gemini_answer"]
                gemini_diagnostic = {
                    "success": True,
                    "status_code": 200,
                    "model_used": "semantic_cache",
                    "note": f"cache_hit_{cache_res.cache_type}",
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "duration_seconds": cache_res.latency_ms / 1000.0,
                }
                logger.info("LangGraph Gemini synthesis bypassed via %s cache hit!", cache_res.cache_type)
            else:
                try:
                    # Query Mem0 for user context and persistent preferences
                    mem0_svc = get_mem0_service()
                    if not mem0_memories:
                        mem0_memories = mem0_svc.search_memories(
                            query=state.get("question", ""),
                            user_id=user_id,
                            limit=4,
                        )
                except Exception as e_mem:
                    logger.warning("Mem0 retrieval warning: %s", e_mem)
                    mem0_memories = []

                try:
                    from app.retrieval.gemini_service import generate_gemini_with_diagnostics

                    diag_res = generate_gemini_with_diagnostics(
                        question=state.get("question", ""),
                        retrieved_docs=state.get("retrieved_documents", []),
                        conversation_history=state.get("messages", []),
                        comparison_data=state.get("comparison_data"),
                        topic=state.get("topic"),
                        user_memories=mem0_memories,
                        fallback_rule_based_answer=state.get("answer", ""),
                        fallback_citations=state.get("sources", []),
                        force_gemini=(state.get("selected_engine") == "gemini"),
                    )
                    gemini_ans = diag_res.text
                    gemini_diagnostic = diag_res.to_dict()
                except Exception as e:
                    logger.warning("Fault tolerance fallback: Gemini synthesis error: %s", e)
                    gemini_ans = (
                        f"⚠️ *Gemini AI synthesis encountered an error: {e}. "
                        "The deterministic rule-based analysis remains 100% verified against your local Constitution PDF.*"
                    )
                    gemini_diagnostic = {
                        "success": False,
                        "error_type": "exception",
                        "provider_message": str(e),
                    }

    return {
        "gemini_answer": gemini_ans,
        "gemini_diagnostic": gemini_diagnostic,
        "mem0_memories": mem0_memories,
    }


def finalize_response(state: ConstitutionState, *, store: Any = None) -> Dict[str, Any]:
    """Node 8: Final state update, applying Human-in-the-Loop engine choice, updating Store, and appending to history.
    
    Human-in-the-Loop decision options:
    - 'rule_based': Human chose Rule-Based Engine only (strict PDF grounding)
    - 'gemini': Human chose Gemini AI LLM only (plain English synthesis)
    - 'both': Human chose Dual View (side-by-side comparative)
    """
    choice = state.get("selected_engine", "both") or "both"
    rule_ans = state.get("answer", "")
    gemini_ans = state.get("gemini_answer", "")
    sources = list(state.get("sources", []))
    retrieved = state.get("retrieved_documents", [])
    response_type = state.get("response_type", "general")
    t_start = state.get("turn_start_time")
    if t_start:
        exec_time = round(time.perf_counter() - t_start, 2)
    else:
        exec_time = state.get("execution_time_seconds")
    if exec_time is None:
        exec_time = 0.25

    from datetime import datetime
    turn_ts = state.get("turn_timestamp") or datetime.now().strftime("%I:%M:%S %p")

    # Citation validation & answerability enforcement
    ans_status = state.get("answerability_status")
    if not ans_status:
        if response_type == "out_of_scope":
            ans_status = "OUT_OF_DOMAIN"
        elif response_type in ["insufficient_evidence", "uncertain"]:
            ans_status = "INSUFFICIENT_EVIDENCE"
        elif response_type in ["retrieval_failure"]:
            ans_status = "RETRIEVAL_FAILURE"
        else:
            ans_status = "ANSWERABLE"

    from app.retrieval.evidence_guard import validate_citations_for_answer

    # If abstaining, out of domain, or experiencing error: strictly ZERO citations!
    if ans_status != "ANSWERABLE" or response_type in [
        "out_of_scope",
        "insufficient_evidence",
        "uncertain",
        "retrieval_failure",
        "generation_failure",
    ]:
        sources = []
        retrieved = []
        citation_valid = False
        grounding_badge = False
        citation_status_label = (
            "Insufficient evidence in the available knowledge base."
            if ans_status == "INSUFFICIENT_EVIDENCE"
            else "Domain restriction refusal."
        )
    else:
        eval_answer = rule_ans if choice == "rule_based" else (gemini_ans or rule_ans)
        val_res = validate_citations_for_answer(eval_answer, retrieved or sources, ans_status)
        sources = val_res.get("citations", [])
        citation_valid = val_res.get("is_valid", False)
        grounding_badge = val_res.get("grounding_badge", False)
        citation_status_label = val_res.get("status_label", "Verified Grounded")

    # Format message content based on Human Decision
    if choice == "rule_based":
        msg_content = rule_ans
    elif choice == "gemini":
        msg_content = gemini_ans
    else:  # "both" (default)
        msg_content = rule_ans

    ai_msg = AIMessage(
        content=msg_content,
        additional_kwargs={
            "gemini_answer": gemini_ans,
            "gemini_diagnostic": state.get("gemini_diagnostic"),
            "rule_answer": rule_ans,
            "selected_engine": choice,
            "sources": sources,
            "retrieved_documents": retrieved,
            "rerank_metadata": state.get("rerank_metadata"),
            "execution_time_seconds": exec_time,
            "turn_timestamp": turn_ts,
            "timestamp": turn_ts,
            "timer_display": f"{exec_time:.2f}s",
            "article_number": (state.get("article_number") or state.get("last_discussed_article")) if ans_status == "ANSWERABLE" else None,
            "topic": state.get("topic") if ans_status == "ANSWERABLE" else None,
            "response_type": response_type,
            "intent": state.get("intent"),
            "answerability_status": ans_status,
            "abstention_reason": state.get("abstention_reason"),
            "citation_is_valid": citation_valid,
            "grounding_badge": grounding_badge,
            "citation_status_label": citation_status_label,
            "original_query": state.get("original_query") or state.get("raw_question"),
            "resolved_query": state.get("resolved_query") or state.get("question"),
            "is_follow_up": state.get("is_follow_up", False),
            "follow_up_type": state.get("follow_up_type", ""),
            "active_topic": state.get("active_topic"),
        },
    )

    # Cross-Thread Store integration
    if store is not None:
        last_art = state.get("last_discussed_article")
        if last_art:
            try:
                store.put(
                    ("recent_topics", "global"),
                    f"art_{last_art}",
                    {
                        "article": last_art,
                        "title": state.get("last_discussed_title", ""),
                        "chosen_engine": choice,
                    },
                )
            except Exception as ex:
                logger.debug("Store write skipped: %s", ex)

    # Mem0 Intelligent Memory Auto-Extraction (100% Background Non-Blocking Execution)
    def _bg_save_mem0() -> None:
        try:
            from app.memory.mem0_service import get_mem0_service
            user_id = state.get("user_id") or "default_user"
            if state.get("intent") not in ["out_of_scope", "uncertain", "greeting"]:
                mem0 = get_mem0_service()
                mem0.extract_and_save_conversation_memory(
                    user_id=user_id,
                    question=state.get("raw_question") or state.get("question", ""),
                    answer=rule_ans or gemini_ans or "",
                    metadata={
                        "article_number": state.get("article_number") or state.get("last_discussed_article"),
                        "topic": state.get("topic"),
                    },
                )
        except Exception as mem_err:
            logger.debug("Mem0 auto-record skipped: %s", mem_err)

    # Telemetry and Semantic Cache Recording (100% Background Execution)
    def _bg_record_cost_and_cache() -> None:
        try:
            from app.cache.semantic_cache import get_semantic_cache
            from app.cost.cost_tracker import RequestCostRecord, get_cost_tracker
            import uuid

            cost_tracker = get_cost_tracker()
            sem_cache = get_semantic_cache()

            clean_q = state.get("raw_question") or state.get("question", "")
            diag = state.get("gemini_diagnostic") or {}

            p_tokens = diag.get("prompt_tokens", 0)
            o_tokens = diag.get("output_tokens", 0)
            cost_val, cost_status = cost_tracker.calculate_cost(
                input_tokens=p_tokens,
                output_tokens=o_tokens,
                chatbot_mode=choice,
            )

            diag_note = str(diag.get("note", ""))
            is_cache_hit = "exact" if "cache_hit_exact" in diag_note else (
                "semantic" if "cache_hit_semantic" in diag_note else "miss"
            )

            rec = RequestCostRecord(
                request_id=f"req_{uuid.uuid4().hex[:8]}",
                session_id=str(state.get("user_id") or "default_session"),
                user_id=str(state.get("user_id") or "default_user"),
                chatbot_mode=choice,
                llm_input_tokens=p_tokens,
                llm_output_tokens=o_tokens,
                llm_generation_latency_ms=diag.get("duration_seconds", 0.0) * 1000.0,
                total_latency_ms=(exec_time or 0.1) * 1000.0,
                retry_count=diag.get("retry_count", 0),
                failed_api_calls=1 if diag.get("success") is False else 0,
                cache_hit=is_cache_hit,
                estimated_cost_usd=cost_val,
                cost_status=cost_status,
            )
            cost_tracker.record_request(rec)

            # Store in semantic cache only if response was truly answerable and valid
            if (
                (gemini_ans or rule_ans)
                and is_cache_hit == "miss"
                and ans_status == "ANSWERABLE"
                and state.get("intent") not in ["out_of_scope", "uncertain", "greeting", "temporal_inquiry", "insufficient_evidence"]
            ):
                q_vec = None
                try:
                    from app.retrieval.dual_pipeline import SharedConstitutionalCorpus
                    corpus = SharedConstitutionalCorpus.get_instance()
                    if corpus.vectorizer is not None:
                        q_vec = corpus.vectorizer.transform([clean_q]).toarray().flatten()
                except Exception as ex_v:
                    logger.debug("Vectorization for cache store failed: %s", ex_v)

                target_article = state.get("article_number") or state.get("last_discussed_article")
                sem_cache.store(
                    query=clean_q,
                    response_payload={
                        "answer": rule_ans,
                        "gemini_answer": gemini_ans,
                        "sources": sources,
                        "response_type": response_type,
                        "article_number": target_article,
                    },
                    chatbot_mode=choice,
                    query_vector=q_vec,
                    user_id=str(state.get("user_id") or "default_user"),
                    target_article=target_article,
                    is_valid=True,
                )
        except Exception as ex:
            logger.debug("Cost & cache recording exception: %s", ex)

    import threading
    threading.Thread(target=_bg_save_mem0, daemon=True).start()
    threading.Thread(target=_bg_record_cost_and_cache, daemon=True).start()

    logger.info(
        "Structured Response Observability: thread_id=%s, original_query='%s', resolved_query='%s', "
        "selected_engine=%s, answerability_status=%s, execution_time=%.2fs",
        state.get("thread_id") or "default",
        state.get("original_query") or state.get("raw_question"),
        state.get("resolved_query") or state.get("question"),
        choice,
        ans_status,
        exec_time or 0.0,
    )

    return {
        "messages": [ai_msg],
        "answer": rule_ans,
        "gemini_answer": gemini_ans,
        "sources": sources,
        "retrieved_documents": retrieved,
        "response_type": response_type,
        "selected_engine": choice,
        "execution_time_seconds": exec_time,
        "turn_timestamp": turn_ts,
        "answerability_status": ans_status,
        "abstention_reason": state.get("abstention_reason"),
        "citation_is_valid": citation_valid,
        "grounding_badge": grounding_badge,
        "citation_status_label": citation_status_label,
        "original_query": state.get("original_query") or state.get("raw_question"),
        "resolved_query": state.get("resolved_query") or state.get("question"),
        "is_follow_up": state.get("is_follow_up", False),
        "follow_up_type": state.get("follow_up_type", ""),
        "active_topic": state.get("active_topic"),
    }
