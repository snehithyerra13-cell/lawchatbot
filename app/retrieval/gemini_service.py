"""Gemini AI Integration Service for ConstitutionGPT.

Provides LLM-augmented constitutional explanations, synthesis, and comparative analysis
grounded in the retrieved PDF constitutional text and multi-turn conversation history.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config.settings import get_settings

logger = logging.getLogger(__name__)

FALLBACK_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
]

_HTTP_SESSION = None


def get_http_session():
    """Retrieve shared requests Session with connection pooling for low latency."""
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        sess = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=Retry(total=1, backoff_factor=0.1),
        )
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _HTTP_SESSION = sess
    return _HTTP_SESSION


def format_conversation_history(
    messages: Sequence[BaseMessage], max_turns: int = 8
) -> str:
    """Format prior conversation turns so Gemini has complete context knowledge."""
    if not messages:
        return "No prior conversation in this thread."

    history_lines: List[str] = []
    # Take up to the last max_turns messages
    recent_msgs = messages[-max_turns:]

    for msg in recent_msgs:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = getattr(msg, "content", "").strip()
        if content:
            # Truncate very long prior outputs for prompt efficiency
            snippet = content[:600] + ("..." if len(content) > 600 else "")
            history_lines.append(f"{role}: {snippet}")

    return "\n\n".join(history_lines)


def format_retrieved_evidence(
    docs: List[Dict[str, Any]],
    comparison_data: Optional[Dict[str, Any]] = None,
    topic: Optional[str] = None,
) -> str:
    """Format extracted constitutional PDF text as grounding evidence."""
    if comparison_data:
        art1 = comparison_data["article_1"]
        art2 = comparison_data["article_2"]
        return (
            f"--- GROUNDING EVIDENCE FROM LOADED CONSTITUTION PDF ---\n"
            f"[Article {art1['article_number']}: {art1['article_title']}] (Part: {art1['part']}, PDF Page {art1['page_start']})\n"
            f"Verbatim Text: {art1['constitutional_text']}\n\n"
            f"[Article {art2['article_number']}: {art2['article_title']}] (Part: {art2['part']}, PDF Page {art2['page_start']})\n"
            f"Verbatim Text: {art2['constitutional_text']}\n"
            f"-------------------------------------------------------"
        )

    if topic and docs:
        snippets = []
        for d in docs[:5]:
            snippets.append(
                f"- Article {d['article_number']} ({d['article_title']}, Page {d['page_start']}): {d['constitutional_text'][:400]}"
            )
        return (
            f"--- GROUNDING EVIDENCE FROM LOADED CONSTITUTION PDF ---\n"
            f"Topic: {topic}\n"
            + "\n".join(snippets)
            + "\n-------------------------------------------------------"
        )

    if docs:
        sections = []
        for idx, d in enumerate(docs[:3]):
            score_line = f"Rerank Score: {d.get('rerank_score')}/100 ({d.get('rerank_reason', '')})\n" if d.get("rerank_score") is not None else ""
            sections.append(
                f"[Document #{idx+1}: Article {d.get('article_number', '')} — {d.get('article_title', '')}] (Part: {d.get('part', '')}, Page {d.get('page_start', '')})\n"
                f"{score_line}"
                f"Verbatim Constitutional Text:\n{d.get('constitutional_text', '')}"
            )
        return (
            f"--- GROUNDING EVIDENCE FROM LOADED CONSTITUTION PDF ---\n"
            + "\n\n".join(sections)
            + "\n-------------------------------------------------------"
        )

    return "No specific constitutional articles matched in local PDF."


def clean_gemini_response(text: str) -> str:
    """Strip repetitive meta-commentary, introductory filler, and robotic persona throat-clearing."""
    if not text:
        return text

    lines = text.strip().splitlines()
    pattern = (
        r"^[\s\*\#\-\>]*(?:"
        r"as an? (?:expert|constitutional|legal|assistant|scholar|ai|specialist)|"
        r"based on (?:the|our|your)? (?:official|provided|extracted|context|previous|prior)|"
        r"here is (?:a|an|the)|"
        r"certainly[,!.]?|sure[,!.]?|in this response|allow me to|with regards? to|"
        r"welcome to|in the context of your previous|"
        r"building upon our previous discussion"
        r").*$"
    )
    # Check top lines and remove any matching introductory meta-phrases
    cleaned_lines = []
    checking = True
    removed_count = 0
    for line in lines:
        stripped = line.strip()
        if checking and removed_count < 6:
            if not stripped or re.match(pattern, stripped, re.IGNORECASE):
                removed_count += 1
                continue
            else:
                checking = False
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    return result or text


def generate_gemini_constitutional_response(
    question: str,
    retrieved_docs: List[Dict[str, Any]],
    conversation_history: Sequence[BaseMessage],
    comparison_data: Optional[Dict[str, Any]] = None,
    topic: Optional[str] = None,
    api_key: Optional[str] = None,
    user_memories: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate an AI response using Google Gemini, grounded in PDF evidence, chat history, and Mem0 memories."""
    import requests

    settings = get_settings()
    key = (api_key or settings.gemini_api_key).strip()

    if not key:
        return (
            "*Gemini AI synthesis is not configured. Please provide a valid Gemini API key "
            "in the sidebar or through the `GEMINI_API_KEY` environment variable.*"
        )

    history_str = format_conversation_history(conversation_history)
    evidence_str = format_retrieved_evidence(
        retrieved_docs, comparison_data=comparison_data, topic=topic
    )

    mem_section = ""
    if user_memories:
        mem_lines = []
        for m in user_memories:
            text = m.get("memory") if isinstance(m, dict) else str(m)
            if text:
                mem_lines.append(f"- {text}")
        if mem_lines:
            mem_section = (
                "=== MEM0 PERSISTENT USER MEMORIES & CONTEXT ===\n"
                + "\n".join(mem_lines)
                + "\n\n"
            )

    system_instruction = (
        "You are an expert constitutional legal assistant specializing in the Constitution of India. "
        "Your task is to provide clear, articulate, authoritative, and helpful answers to the user's questions. "
        "Always ground your reasoning directly in the provided verbatim evidence extracted from the official Constitution of India PDF. "
        "Maintain continuity with prior conversation history without explicitly mentioning previous research, sessions, or past articles unless the user directly asks for a comparison. "
        "If Mem0 user context is provided, adapt tone and emphasize relevant constitutional aspects accordingly. "
        "Structure your response with clear headings, bullet points, and plain-English explanations.\n"
        "STRICT STYLE RULE: DO NOT use conversational throat-clearing, meta-announcements, or persona declarations. "
        "NEVER begin your answer with 'As an expert...', 'As a constitutional scholar...', 'Based on the official text...', "
        "'Based on the context of your previous research...', 'Here is a breakdown...', or similar phrases. "
        "DO NOT reference prior conversation history unless the user explicitly requested a comparison or follow-up. Jump directly into the substantive legal analysis and headings."
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"=== PRIOR CONVERSATION HISTORY (Previous context knowledge) ===\n"
        f"{history_str}\n\n"
        f"=== CONSTITUTIONAL PDF EVIDENCE ===\n"
        f"{evidence_str}\n\n"
        f"{mem_section}"
        f"=== LATEST USER QUESTION ===\n"
        f"{question}\n\n"
        f"Provide your comprehensive, accessible response based on the above evidence and context. Jump directly into the answer without introductory phrases."
    )

    # 1. Primary: Direct REST call with fast timeout and persistent HTTP connection
    session = get_http_session()
    for model_name in FALLBACK_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        try:
            resp = session.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=25,
            )
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        raw_ans = parts[0]["text"].strip()
                        return clean_gemini_response(raw_ans)
            else:
                logger.warning("REST model %s returned status %d: %s", model_name, resp.status_code, resp.text[:120])
        except Exception as req_err:
            logger.warning("REST call to %s failed: %s", model_name, req_err)

    # 2. Secondary fallback using google-genai client library
    try:
        from google import genai

        client = genai.Client(api_key=key)
        for model_name in FALLBACK_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return clean_gemini_response(response.text.strip())
            except Exception as model_err:
                logger.warning("google-genai model %s failed: %s", model_name, model_err)
    except Exception as lib_err:
        logger.warning("google-genai client failed: %s", lib_err)

    art_str = f" Article {retrieved_docs[0]['article_number']}" if retrieved_docs and retrieved_docs[0].get("article_number") else ""
    return (
        f"⚠️ *Gemini AI synthesis is temporarily unavailable or rate-limited for{art_str}. "
        "The deterministic rule-based answer on the left remains 100% verified against your local Constitution PDF.*"
    )


REWRITE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]


def extract_articles_from_text(text: str) -> List[str]:
    """Helper to extract article numbers from string."""
    q_upper = text.upper()
    results = []
    if "PREAMBLE" in q_upper:
        results.append("PREAMBLE")
    matches = re.findall(r"\b(?:ARTICLE|ART\.?|A)?\s*(\d+[A-Z]?)\b", q_upper)
    for m in matches:
        base_digits = re.match(r"^(\d+)", m)
        if base_digits:
            num = int(base_digits.group(1))
            if 1 <= num <= 395 and m not in results:
                results.append(m)
    return results


def find_prior_article_in_history(messages: Sequence[BaseMessage]) -> Optional[str]:
    """Search conversation history in reverse for the most recently discussed article number."""
    if not messages:
        return None
    for msg in reversed(messages):
        kwargs = getattr(msg, "additional_kwargs", {}) or {}
        if kwargs.get("article_number"):
            return str(kwargs["article_number"]).strip().upper()
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "")
            arts = extract_articles_from_text(content)
            if arts:
                return arts[0]

    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        m = re.search(r"###\s*(?:Simple English Explanation:\s*)?Article\s+(\d+[A-Z]?|PREAMBLE)", content, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        arts = extract_articles_from_text(content)
        if arts:
            return arts[0]
    return None


CONSTITUTIONAL_INDICATORS = {
    "constitution", "constitutional", "article", "articles", "part", "schedule", "amendment", "amendments",
    "preamble", "sovereign", "socialist", "secular", "democratic", "republic", "justice", "liberty", "equality", "fraternity",
    "fundamental right", "fundamental rights", "fundamental duty", "fundamental duties", "directive principle", "directive principles", "dpsp",
    "writ", "writs", "habeas corpus", "mandamus", "certiorari", "prohibition", "quo warranto",
    "supreme court", "high court", "judiciary", "judicial review", "basic structure", "chief justice", "cji", "judge",
    "president", "vice-president", "vice president", "governor", "prime minister", "council of ministers", "cabinet", "attorney general", "advocate general", "solicitor general",
    "parliament", "lok sabha", "rajya sabha", "speaker", "mp", "mla", "legislature", "legislative assembly", "legislative council", "vidhan sabha", "vidhan parishad",
    "ordinance", "bill", "money bill", "budget", "annual financial statement", "finance bill",
    "citizenship", "citizen", "domicile", "reservation", "untouchability", "scheduled caste", "scheduled tribe", "sc", "st", "obc", "ews", "backward class",
    "emergency", "national emergency", "president's rule", "president rule", "financial emergency",
    "election commission", "upsc", "cag", "comptroller and auditor general", "finance commission", "inter-state council",
    "panchayat", "panchayati raj", "municipality", "municipalities", "tribunal", "tribunals",
    "union list", "state list", "concurrent list", "seventh schedule",
    "discrimination", "speech", "expression", "life", "personal liberty", "privacy", "education",
    "arrest", "detention", "preventive detention", "double jeopardy", "self-incrimination", "self incrimination", "remedy", "remedies",
    "fundamental",
}


def is_person_or_name_query(text: str) -> bool:
    """Detect if query is asking to name an individual, officeholder, or asking 'who is xxx'."""
    t = " " + " ".join(text.strip().lower().split()) + " "

    # Check if asking to name someone / officeholder
    name_patterns = [
        r"\bname\s+(the|of|a|an)?\s+(president|prime\s+minister|chief\s+justice|governor|cji|speaker|attorney\s+general|cag|vice\s+president)\b",
        r"\b(what|tell\s+me)\s+(is\s+)?the\s+name\s+of\b",
        r"^\s*name\s+(the|of)?\b",
    ]
    for p in name_patterns:
        if re.search(p, t):
            return True

    # Check if asking 'who is xxx' or 'who was xxx'
    if re.search(r"^\s*who\s+(is|was|are)\b", t):
        if any(term in t for term in [" eligible", " qualified", " appointed by", " removed by"]):
            return False
        return True

    return False


def is_constitutional_domain_query(
    question: str,
    conversation_history: Sequence[BaseMessage] = (),
) -> bool:
    """Determine whether the question pertains to the Constitution of India domain.
    
    Returns False when there is no constitutional context (e.g. asking about contemporary
    politicians, individuals, names of officeholders, weather, sports, coding, etc.).
    """
    q_norm = " ".join(question.strip().split())
    q_lower = q_norm.lower()

    # 1. Reject questions asking for a person's name or asking 'who is xxx'
    if is_person_or_name_query(question):
        return False

    # 2. Greetings
    greetings = {"hi", "hello", "hey", "namaste", "greetings", "good morning", "good evening", "who are you", "what can you do", "help"}
    if q_lower in greetings or any(q_lower.startswith(g + " ") for g in greetings if len(g) > 2):
        return True

    # 3. Direct article mentions
    if extract_articles_from_text(q_norm):
        return True

    # 4. Explicit constitutional indicators & legal domains
    for kw in CONSTITUTIONAL_INDICATORS:
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            return True

    # 4. Check if it's an explicit follow-up referencing prior context in conversation history
    follow_up_phrases = [
        "what article i had asked",
        "what article i asked",
        "what article did i ask",
        "which article did i ask",
        "what did i ask",
        "what article did we discuss",
        "which article was that",
        "what article was that",
        "earlier article",
        "previous article",
        "what article had i asked",
        "explain it simply",
        "in simpler words",
        "simple english",
        "in simple terms",
        "give an example",
        "give me an example",
        "practical example",
        "case example",
        "tell me more about it",
        "explain it more",
        "simplify it",
    ]
    if any(fp in q_lower for fp in follow_up_phrases):
        prior_art = find_prior_article_in_history(conversation_history)
        if prior_art:
            return True

    # 5. Queries about non-constitutional living people, politicians, or arbitrary subjects
    # e.g. "who is modi", "who is chandra shekar rao", "who is rahul gandhi" have NO constitutional context
    return False


def fallback_heuristic_rewrite(
    question: str, conversation_history: Sequence[BaseMessage] = ()
) -> Dict[str, Any]:
    """Deterministic offline fallback query rewriter when LLM is unavailable."""
    q_norm = " ".join(question.strip().split())
    q_lower = q_norm.lower()

    # Domain context gate: If query has no constitutional context, immediately flag out_of_scope
    if not is_constitutional_domain_query(question, conversation_history):
        return {
            "rewritten_query": None,
            "articles": [],
            "intent": "out_of_scope",
            "topic": None,
            "confidence": 1.0,
        }

    # 1. Greetings
    greetings = {"hi", "hello", "hey", "namaste", "greetings", "good morning", "good evening", "who are you"}
    if q_lower in greetings or any(q_lower.startswith(g + " ") for g in greetings):
        return {
            "rewritten_query": q_norm,
            "articles": [],
            "intent": "greeting",
            "topic": None,
            "confidence": 1.0,
        }

    # 2. History inquiries (e.g. "what article i had asked you")
    history_indicators = [
        "what article i had asked",
        "what article i asked",
        "what article did i ask",
        "which article did i ask",
        "what did i ask",
        "what article did we discuss",
        "which article was that",
        "what article was that",
        "earlier article",
        "previous article",
        "what article had i asked",
    ]
    if any(h in q_lower for h in history_indicators):
        prior_art = find_prior_article_in_history(conversation_history)
        if prior_art:
            return {
                "rewritten_query": f"Which article of the Constitution of India was asked about earlier? (Article {prior_art})",
                "articles": [prior_art],
                "intent": "follow_up_history",
                "topic": "Fundamental Rights",
                "confidence": 0.95,
            }

    # 3. Follow-up simple or example
    if any(p in q_lower for p in ["explain it simply", "in simpler words", "simple english", "in simple terms"]):
        prior_art = find_prior_article_in_history(conversation_history)
        if prior_art:
            return {
                "rewritten_query": f"Explain Article {prior_art} of the Constitution of India in simple plain English.",
                "articles": [prior_art],
                "intent": "follow_up_simple",
                "topic": None,
                "confidence": 0.95,
            }

    if any(p in q_lower for p in ["give an example", "give me an example", "practical example", "case example"]):
        prior_art = find_prior_article_in_history(conversation_history)
        if prior_art:
            return {
                "rewritten_query": f"Provide practical examples and case scenarios for Article {prior_art}.",
                "articles": [prior_art],
                "intent": "follow_up_example",
                "topic": None,
                "confidence": 0.95,
            }

    # 4. Constitutional Thematic Topics (e.g. Fundamental Rights, Directive Principles, Fundamental Duties)
    topic_keywords = {
        "fundamental rights": ("What are the Fundamental Rights under Part III of the Constitution of India?", "Fundamental Rights"),
        "directive principles": ("What are the Directive Principles of State Policy?", "Directive Principles"),
        "fundamental duties": ("What are the Fundamental Duties under Part IVA?", "Fundamental Duties"),
        "emergency": ("What are the Emergency Provisions under Part XVIII?", "Emergency Provisions"),
    }
    for kw, (rewritten, top_name) in topic_keywords.items():
        if kw in q_lower:
            return {
                "rewritten_query": rewritten,
                "articles": [],
                "intent": "topic",
                "topic": top_name,
                "confidence": 0.95,
            }

    # 5. Direct article numbers
    arts = extract_articles_from_text(q_norm)
    if len(arts) >= 2 and any(w in q_lower for w in ["compare", "difference", "vs", "versus"]):
        return {
            "rewritten_query": f"Compare Article {arts[0]} and Article {arts[1]} of the Constitution of India",
            "articles": arts[:2],
            "intent": "comparison",
            "topic": None,
            "confidence": 0.95,
        }
    elif len(arts) == 1:
        return {
            "rewritten_query": f"What is Article {arts[0]} of the Constitution of India?",
            "articles": [arts[0]],
            "intent": "article_lookup",
            "topic": None,
            "confidence": 0.9,
        }

    # 6. Check for constitutional keywords
    constitutional_keywords = {
        "constitution", "constitutional", "article", "amendment", "preamble", "schedule",
        "fundamental right", "fundamental rights", "directive principle", "directive principles",
        "fundamental duty", "fundamental duties", "writ", "habeas corpus", "mandamus",
        "supreme court", "high court", "judiciary", "president", "governor",
        "parliament", "lok sabha", "rajya sabha", "legislature", "ordinance",
        "citizenship", "reservation", "emergency", "equality", "discrimination",
        "untouchability", "speech", "expression", "life", "liberty", "privacy",
        "arrest", "detention", "double jeopardy", "self incrimination", "religion",
        "election commission", "attorney general", "upsc", "tribunal", "cag",
        "remedy", "remedies", "secular", "socialist", "sovereign", "republic",
    }
    has_const_kw = any(kw in q_lower for kw in constitutional_keywords)

    if not has_const_kw:
        return {
            "rewritten_query": None,
            "articles": [],
            "intent": "out_of_scope",
            "topic": None,
            "confidence": 0.95,
        }

    return {
        "rewritten_query": q_norm,
        "articles": arts,
        "intent": "general",
        "topic": None,
        "confidence": 0.5,
    }


def rewrite_constitutional_query(
    question: str,
    conversation_history: Sequence[BaseMessage] = (),
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Directly send question to LLM so the LLM decides if it is domain-related or not.
    
    If domain-related, LLM rewrites it into a canonical constitutional query for both
    Rule-Based and Gemini engines. If not domain-related, LLM flags it as out_of_scope
    so the user is prompted to ask questions based on the domain only.
    """
    settings = get_settings()
    key = (api_key or settings.gemini_api_key).strip()

    if not key:
        return fallback_heuristic_rewrite(question, conversation_history)

    # Fast check: If asking for an individual person's name or who holds an office
    if is_person_or_name_query(question):
        return {
            "is_domain_related": False,
            "rewritten_query": None,
            "articles": [],
            "intent": "out_of_scope",
            "topic": None,
            "confidence": 1.0,
        }

    history_str = format_conversation_history(conversation_history, max_turns=6)

    prompt = (
        "You are the Gatekeeper, Domain Classifier, and Query Rewriter for the Constitution of India legal assistant.\n"
        "Your PRIMARY task is to decide whether the user's question is RELATED to the domain of the Constitution of India or NOT.\n\n"
        "DOMAIN DEFINITION:\n"
        "- IN DOMAIN: Questions specifically about the Constitution of India (Articles 1-395, Parts I-XXII, Schedules, Amendments, Fundamental Rights, Directive Principles, Constitutional Offices like President, Prime Minister, Governor, Parliament, Supreme Court, constitutional legal doctrines and rights), greetings/help, or valid conversational follow-ups.\n"
        "- NOT IN DOMAIN (OUT OF SCOPE):\n"
        "  1. Questions asking for individual people, specific living persons, politicians, or officeholders (e.g., 'who is modi', 'who is chandra shekar rao', 'who is rahul gandhi').\n"
        "  2. Questions asking to NAME an individual person or asking 'who is the [officeholder]' (e.g., 'name the president of india', 'who is the president of india', 'name the prime minister', 'who is the prime minister', 'who is the chief justice', 'what is the name of the president').\n"
        "     NOTE: The Constitution of India establishes constitutional offices (e.g., Article 52 establishes the office of President), but DOES NOT contain the names of individual officeholders or incumbents. Inquiries asking for a person's name or who holds the office are asking about a single person and are NOT in the constitutional domain! Do NOT map 'name the president' or 'who is the president' to Article 52 or the Union Executive!\n"
        "  3. General politics, weather, sports, entertainment, recipes, coding, general trivia.\n\n"
        f"=== PRIOR CHAT HISTORY ===\n{history_str}\n\n"
        f"=== USER QUESTION ===\n\"{question}\"\n\n"
        "Output ONLY a valid JSON object:\n"
        "If the question is NOT domain related:\n"
        "{\n"
        "  \"is_domain_related\": false,\n"
        "  \"intent\": \"out_of_scope\",\n"
        "  \"rewritten_query\": null,\n"
        "  \"articles\": [],\n"
        "  \"topic\": null\n"
        "}\n\n"
        "If the question IS domain related:\n"
        "{\n"
        "  \"is_domain_related\": true,\n"
        "  \"intent\": \"article_lookup\" | \"topic\" | \"comparison\" | \"follow_up_simple\" | \"follow_up_example\" | \"follow_up_history\" | \"greeting\" | \"general\",\n"
        "  \"rewritten_query\": \"Canonical rewritten question referencing specific article numbers and titles\",\n"
        "  \"articles\": [\"specific article numbers if applicable\"],\n"
        "  \"topic\": \"topic name if applicable or null\"\n"
        "}\n"
    )

    session = get_http_session()

    for model_name in REWRITE_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        try:
            resp = session.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"response_mime_type": "application/json"},
                },
                timeout=6,
            )
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        raw_json = parts[0]["text"].strip()
                        parsed = json.loads(raw_json)
                        if isinstance(parsed, dict):
                            is_rel = parsed.get("is_domain_related", True)
                            intent_val = parsed.get("intent", "")
                            if (not is_rel) or intent_val == "out_of_scope" or parsed.get("rewritten_query") is None:
                                return {
                                    "is_domain_related": False,
                                    "rewritten_query": None,
                                    "articles": [],
                                    "intent": "out_of_scope",
                                    "topic": None,
                                    "confidence": 1.0,
                                }
                            if "articles" in parsed and isinstance(parsed["articles"], list):
                                parsed["articles"] = [str(a).strip().upper() for a in parsed["articles"] if a]
                            parsed["is_domain_related"] = True
                            return parsed
        except Exception as e:
            logger.debug("Rewriter model %s exception: %s", model_name, e)

    return fallback_heuristic_rewrite(question, conversation_history)

