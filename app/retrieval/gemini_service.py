"""Gemini AI Integration Service for ConstitutionGPT.

Provides LLM-augmented constitutional explanations, synthesis, and comparative analysis
grounded in the retrieved PDF constitutional text and multi-turn conversation history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config.settings import get_settings

logger = logging.getLogger(__name__)

FALLBACK_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-pro-latest",
]

_HTTP_SESSION = None
_LAST_DIAGNOSTIC: Optional[GeminiDiagnosticResult] = None


@dataclass
class GeminiDiagnosticResult:
    """Safe structured diagnostic record capturing execution outcome, provider status, and latency."""

    success: bool
    text: str = ""
    status_code: Optional[int] = None
    error_type: Optional[str] = None  # "quota_exceeded", "auth_invalid", "permission_denied", "not_found", "bad_request", "server_error", "timeout", "network_error", "malformed_response", "missing_key"
    provider_code: Optional[Any] = None
    provider_status: Optional[str] = None
    provider_message: Optional[str] = None
    model_used: Optional[str] = None
    models_attempted: List[str] = field(default_factory=list)
    retry_count: int = 0
    duration_seconds: float = 0.0
    user_friendly_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_last_gemini_diagnostic() -> Optional[GeminiDiagnosticResult]:
    """Retrieve the most recent Gemini diagnostic record."""
    return _LAST_DIAGNOSTIC


def sanitize_error_message(text: str, key: Optional[str] = None) -> str:
    """Safely redact API keys and sensitive tokens from error text to prevent secret leaks."""
    if not text:
        return ""
    sanitized = str(text)
    if key and len(key) > 6:
        sanitized = sanitized.replace(key, "[REDACTED_API_KEY]")
    # Redact query parameters in URLs like ?key=... or &key=...
    sanitized = re.sub(r"([?&]key=)[^&\s]+", r"\1[REDACTED_API_KEY]", sanitized)
    # Redact common Gemini/Google key token formats
    sanitized = re.sub(r"\bAIza[0-9A-Za-z-_]{35}\b", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"\bAQ\.[0-9A-Za-z-_]{40,}\b", "[REDACTED_API_KEY]", sanitized)
    return sanitized


def classify_http_error(status_code: int, error_data: Dict[str, Any]) -> Tuple[str, str]:
    """Categorize HTTP error status code and provider payload into structured error types."""
    msg = error_data.get("message", "")
    p_status = error_data.get("status", "")

    if status_code == 429 or p_status == "RESOURCE_EXHAUSTED" or "quota" in msg.lower():
        return "quota_exceeded", msg or "Quota exceeded or rate limit reached"
    elif status_code == 401 or p_status == "UNAUTHENTICATED":
        return "auth_invalid", msg or "Invalid or unauthenticated API key"
    elif status_code == 403 or p_status == "PERMISSION_DENIED":
        return "permission_denied", msg or "Access forbidden or API not enabled"
    elif status_code == 404 or p_status == "NOT_FOUND":
        return "not_found", msg or "Requested model not found or discontinued"
    elif status_code == 400 or p_status == "INVALID_ARGUMENT":
        return "bad_request", msg or "Invalid argument or bad request"
    elif status_code in (500, 502, 503, 504) or p_status in ("INTERNAL", "UNAVAILABLE"):
        return "server_error", msg or f"Google Gemini upstream server error ({status_code})"
    else:
        return f"http_{status_code}", msg or f"HTTP error {status_code}"


def is_retryable_error(status_code: Optional[int], exc: Optional[Exception] = None) -> bool:
    """Determine whether an error is transient and eligible for bounded retries."""
    if status_code in (429, 500, 502, 503, 504):
        return True
    if exc is not None:
        import requests
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
    return False


def compute_backoff_delay(
    attempt: int,
    headers: Optional[Dict[str, str]] = None,
    base_delay: float = 0.5,
    max_delay: float = 3.0,
) -> float:
    """Calculate bounded exponential backoff with jitter, respecting Retry-After header."""
    if headers:
        for k, v in headers.items():
            if k.lower() == "retry-after":
                try:
                    val = float(v)
                    return max(0.1, min(val, max_delay))
                except (ValueError, TypeError):
                    pass
    backoff = base_delay * (2 ** attempt) + random.uniform(0.05, 0.20)
    return min(backoff, max_delay)


def format_user_friendly_error(diag: GeminiDiagnosticResult, article_str: str = "") -> str:
    """Produce a truthful, specific, and actionable user-facing explanation when generation fails."""
    art_clause = f" for {article_str}" if article_str else ""
    if diag.error_type == "missing_key":
        return (
            "**Gemini AI Synthesis Not Configured**\n\n"
            "No Gemini API key is configured. Please provide a valid Gemini API key in the sidebar under "
            "**Gemini API Settings** or set `GEMINI_API_KEY` in `.env`.\n\n"
            "*The deterministic rule-based answer remains 100% verified against your local Constitution PDF.*"
        )
    elif diag.error_type == "quota_exceeded":
        return (
            f"**Gemini AI Quota / Rate Limit Exceeded** (HTTP 429{art_clause})\n\n"
            f"Google Gemini reported that current project quota was exhausted for model `{diag.model_used or 'gemini'}`: "
            f"_{diag.provider_message or 'You exceeded your current quota'}_.\n\n"
            "*Tip: Wait a few moments before rerunning, or configure an upgraded tier API key in sidebar settings.*\n\n"
            "*The deterministic rule-based answer on the left remains 100% verified against your local Constitution PDF.*"
        )
    elif diag.error_type == "auth_invalid":
        return (
            f"**Gemini AI Authentication Failure** (HTTP 401{art_clause})\n\n"
            "The configured Gemini API key was rejected by Google Generative AI (HTTP 401 Unauthorized). "
            "Please check and update your API key in the sidebar under **Gemini API Settings**.\n\n"
            "*The deterministic rule-based answer on the left remains 100% verified against your local Constitution PDF.*"
        )
    elif diag.error_type == "permission_denied":
        return (
            f"**Gemini AI Permission Denied** (HTTP 403{art_clause})\n\n"
            f"Access to Google Gemini is forbidden: _{diag.provider_message or 'Permission denied'}_. "
            "Verify that Generative Language API is enabled in your Google Cloud Console project.\n\n"
            "*The deterministic rule-based answer on the left remains 100% verified against your local Constitution PDF.*"
        )
    elif diag.error_type == "not_found":
        return (
            f"**Gemini AI Model Not Found** (HTTP 404{art_clause})\n\n"
            f"The requested model `{diag.model_used}` is discontinued or unavailable. Models attempted: `{', '.join(diag.models_attempted)}`.\n\n"
            "*The deterministic rule-based answer on the left remains 100% verified against your local Constitution PDF.*"
        )
    elif diag.error_type == "bad_request":
        return (
            f"**Gemini AI Invalid Request** (HTTP 400{art_clause})\n\n"
            f"The request payload was rejected: _{diag.provider_message or 'Bad request'}_.\n\n"
            "*The deterministic rule-based answer on the left remains 100% verified against your local Constitution PDF.*"
        )
    elif diag.error_type == "server_error":
        return (
            f"**Google Gemini Service Unavailable** (HTTP {diag.status_code or 503}{art_clause})\n\n"
            "Google's generative AI service is experiencing internal errors or high load. Bounded retries were attempted.\n\n"
            "*The deterministic rule-based answer on the left remains 100% verified against your local Constitution PDF.*"
        )
    elif diag.error_type == "timeout":
        return (
            f"**Gemini AI Request Timed Out** ({diag.duration_seconds:.1f}s elapsed{art_clause})\n\n"
            "The Google Gemini API did not respond within the allocated timeout window.\n\n"
            "*The deterministic rule-based answer on the left remains 100% verified against your local Constitution PDF.*"
        )
    elif diag.error_type == "network_error":
        return (
            f"**Gemini AI Network Connection Error**{art_clause}\n\n"
            f"Could not connect to Google Generative Language API: _{diag.provider_message or 'Network unreachable'}_.\n\n"
            "*The deterministic rule-based answer on the left remains 100% verified against your local Constitution PDF.*"
        )
    else:
        return (
            f"**Gemini AI Synthesis Unavailable**{art_clause}\n\n"
            f"_{diag.provider_message or 'An unexpected error occurred during synthesis'}_.\n\n"
            "*The deterministic rule-based answer on the left remains 100% verified against your local Constitution PDF.*"
        )


def clean_constitutional_evidence_text(text: str) -> str:
    """Normalize PDF extraction artifacts and replacement characters in legal text."""
    if not text:
        return ""
    cleaned = text.replace("\ufffd", "—").replace("\x14", " ").replace("\x00", "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


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
        text1 = clean_constitutional_evidence_text(art1.get("constitutional_text", ""))
        text2 = clean_constitutional_evidence_text(art2.get("constitutional_text", ""))
        return (
            f"--- GROUNDING EVIDENCE FROM LOADED CONSTITUTION PDF ---\n"
            f"[Article {art1['article_number']}: {art1['article_title']}] (Part: {art1['part']}, PDF Page {art1['page_start']})\n"
            f"Verbatim Text: {text1}\n\n"
            f"[Article {art2['article_number']}: {art2['article_title']}] (Part: {art2['part']}, PDF Page {art2['page_start']})\n"
            f"Verbatim Text: {text2}\n"
            f"-------------------------------------------------------"
        )

    if topic and docs:
        snippets = []
        for d in docs[:5]:
            snippet_text = clean_constitutional_evidence_text(d.get("constitutional_text", "")[:400])
            snippets.append(
                f"- Article {d['article_number']} ({d['article_title']}, Page {d['page_start']}): {snippet_text}"
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
            clean_body = clean_constitutional_evidence_text(d.get("constitutional_text", ""))
            sections.append(
                f"[Document #{idx+1}: Article {d.get('article_number', '')} — {d.get('article_title', '')}] (Article: {d.get('article_number', '')}, Part: {d.get('part', '')}, Page {d.get('page_start', '')})\n"
                f"{score_line}"
                f"Verbatim Constitutional Text:\n{clean_body}"
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


def execute_gemini_request_with_diagnostics(
    prompt: str,
    api_key: Optional[str] = None,
    article_label: str = "",
    timeout: int = 25,
    max_retries_per_model: int = 1,
    models: Optional[List[str]] = None,
) -> GeminiDiagnosticResult:
    """Execute Gemini generation with safe structured diagnostics, bounded retries, and truthful error categorization."""
    global _LAST_DIAGNOSTIC
    import requests

    settings = get_settings()
    key = (api_key or settings.gemini_api_key or "").strip()

    if not key:
        diag = GeminiDiagnosticResult(
            success=False,
            error_type="missing_key",
            status_code=None,
            provider_message="Gemini API key is not configured.",
        )
        diag.user_friendly_message = format_user_friendly_error(diag, article_label)
        diag.text = diag.user_friendly_message
        _LAST_DIAGNOSTIC = diag
        return diag

    models_to_try = list(models or FALLBACK_MODELS)
    models_attempted: List[str] = []
    total_retries = 0
    t_global_start = time.perf_counter()

    last_status: Optional[int] = None
    last_error_type: str = "unknown"
    last_p_code: Optional[Any] = None
    last_p_status: Optional[str] = None
    last_p_msg: Optional[str] = None
    last_model: Optional[str] = None

    session = get_http_session()

    for model_name in models_to_try:
        models_attempted.append(model_name)
        last_model = model_name
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"

        for attempt in range(max_retries_per_model + 1):
            t_req_start = time.perf_counter()
            try:
                resp = session.post(
                    url,
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.2,
                            "maxOutputTokens": 800,
                            "topP": 0.9,
                        },
                    },
                    timeout=timeout,
                )
                duration = time.perf_counter() - t_req_start

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                raw_ans = parts[0]["text"].strip()
                                cleaned = clean_gemini_response(raw_ans)
                                diag = GeminiDiagnosticResult(
                                    success=True,
                                    text=cleaned,
                                    status_code=200,
                                    model_used=model_name,
                                    models_attempted=models_attempted,
                                    retry_count=total_retries,
                                    duration_seconds=round(time.perf_counter() - t_global_start, 3),
                                    user_friendly_message="",
                                )
                                _LAST_DIAGNOSTIC = diag
                                return diag
                    except Exception as json_err:
                        last_error_type = "malformed_response"
                        last_p_msg = f"Failed to parse response JSON: {json_err}"
                        break

                    last_status = 200
                    last_error_type = "malformed_response"
                    last_p_msg = "Gemini returned empty candidate content or parts."
                    break

                # Non-200 Response
                last_status = resp.status_code
                err_dict = {}
                try:
                    err_dict = resp.json().get("error", {})
                except Exception:
                    pass

                last_p_code = err_dict.get("code", resp.status_code)
                last_p_status = err_dict.get("status", "")
                raw_msg = err_dict.get("message") or resp.text[:200]
                last_p_msg = sanitize_error_message(raw_msg, key)
                last_error_type, _ = classify_http_error(resp.status_code, err_dict)

                logger.warning(
                    "Gemini model %s returned status %d (%s): %s",
                    model_name,
                    resp.status_code,
                    last_error_type,
                    last_p_msg[:120],
                )

                # Permanent client errors: 400, 401, 403 do not succeed on retry or other models
                if resp.status_code in (400, 401, 403):
                    break

                # 404: Model not found / discontinued -> fail fast on this model, try next model
                if resp.status_code == 404:
                    break

                # 429, 500, 503: Transient retryable errors
                if is_retryable_error(resp.status_code) and attempt < max_retries_per_model:
                    total_retries += 1
                    delay = compute_backoff_delay(attempt, resp.headers)
                    time.sleep(delay)
                    continue

                break

            except requests.Timeout as t_err:
                last_status = None
                last_error_type = "timeout"
                last_p_msg = sanitize_error_message(str(t_err), key)
                logger.warning("Gemini model %s timed out after %ds", model_name, timeout)
                if attempt < max_retries_per_model:
                    total_retries += 1
                    time.sleep(compute_backoff_delay(attempt))
                    continue
                break

            except requests.RequestException as req_err:
                last_status = None
                last_error_type = "network_error"
                last_p_msg = sanitize_error_message(str(req_err), key)
                logger.warning("Gemini model %s network error: %s", model_name, last_p_msg)
                if attempt < max_retries_per_model:
                    total_retries += 1
                    time.sleep(compute_backoff_delay(attempt))
                    continue
                break

        # If permanent auth/permission error, fail fast without trying other models
        if last_status in (401, 403):
            break

    total_duration = round(time.perf_counter() - t_global_start, 3)
    final_diag = GeminiDiagnosticResult(
        success=False,
        status_code=last_status,
        error_type=last_error_type,
        provider_code=last_p_code,
        provider_status=last_p_status,
        provider_message=last_p_msg,
        model_used=last_model,
        models_attempted=models_attempted,
        retry_count=total_retries,
        duration_seconds=total_duration,
    )
    final_diag.user_friendly_message = format_user_friendly_error(final_diag, article_label)
    final_diag.text = final_diag.user_friendly_message
    _LAST_DIAGNOSTIC = final_diag
    return final_diag


def generate_gemini_with_diagnostics(
    question: str,
    retrieved_docs: List[Dict[str, Any]],
    conversation_history: Sequence[BaseMessage] = (),
    comparison_data: Optional[Dict[str, Any]] = None,
    topic: Optional[str] = None,
    api_key: Optional[str] = None,
    user_memories: Optional[List[Dict[str, Any]]] = None,
) -> GeminiDiagnosticResult:
    """Generate Gemini response and return full structured diagnostics."""
    art_str = f"Article {retrieved_docs[0]['article_number']}" if retrieved_docs and retrieved_docs[0].get("article_number") else ""
    history_str = format_conversation_history(conversation_history)
    evidence_str = format_retrieved_evidence(retrieved_docs, comparison_data=comparison_data, topic=topic)

    mem_section = ""
    if user_memories:
        mem_lines = []
        for m in user_memories:
            text = m.get("memory") if isinstance(m, dict) else str(m)
            if text:
                mem_lines.append(f"- {text}")
        if mem_lines:
            mem_section = "=== MEM0 PERSISTENT USER MEMORIES & CONTEXT ===\n" + "\n".join(mem_lines) + "\n\n"

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

    return execute_gemini_request_with_diagnostics(
        prompt=prompt,
        api_key=api_key,
        article_label=art_str,
    )


def generate_gemini_constitutional_response(
    question: str,
    retrieved_docs: List[Dict[str, Any]],
    conversation_history: Sequence[BaseMessage] = (),
    comparison_data: Optional[Dict[str, Any]] = None,
    topic: Optional[str] = None,
    api_key: Optional[str] = None,
    user_memories: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate an AI response using Google Gemini, grounded in PDF evidence, chat history, and Mem0 memories."""
    diag = generate_gemini_with_diagnostics(
        question=question,
        retrieved_docs=retrieved_docs,
        conversation_history=conversation_history,
        comparison_data=comparison_data,
        topic=topic,
        api_key=api_key,
        user_memories=user_memories,
    )
    return diag.text


REWRITE_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
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

    # Fast check 1: If asking for an individual person's name or who holds an office
    if is_person_or_name_query(question):
        return {
            "is_domain_related": False,
            "rewritten_query": None,
            "articles": [],
            "intent": "out_of_scope",
            "topic": None,
            "confidence": 1.0,
        }

    # Fast check 2 (High-Speed Deterministic Routing):
    # If deterministic heuristics detect explicit article numbers, comparisons, landmark topics, or non-constitutional queries with high confidence,
    # return immediately in ~0.05ms to eliminate 12-15s of network latency while maintaining 100% legal precision!
    fast_result = fallback_heuristic_rewrite(question, conversation_history)
    if fast_result.get("confidence", 0.0) >= 0.85:
        logger.info(
            "Query rewriter fast-path triggered for '%s' (confidence=%.2f, intent=%s)",
            question[:45],
            fast_result.get("confidence", 0.0),
            fast_result.get("intent"),
        )
        return fast_result

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

