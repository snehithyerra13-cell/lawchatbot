"""Comprehensive Unit & Integration Tests for Gemini AI Synthesis Timeout,
Bounded Retries, Inflight Deduplication, Caching, Cost-Aware Routing, and Graceful Fallback.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage

from app.retrieval.gemini_service import (
    FALLBACK_MODELS,
    GeminiDiagnosticResult,
    _GEMINI_SYNTHESIS_CACHE,
    compute_gemini_cost,
    execute_gemini_request_with_diagnostics,
    format_conversation_history,
    format_retrieved_evidence,
    format_rule_based_fallback_display,
    generate_gemini_with_diagnostics,
    should_route_to_deterministic_pdf,
)
from app.retrieval.knowledge_base import get_knowledge_base


def _make_mock_response(status_code: int, payload: dict, headers: dict = None) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = json.dumps(payload).encode("utf-8")
    resp.headers.update(headers or {})
    return resp


@pytest.fixture(autouse=True)
def clear_synthesis_cache():
    """Ensure clean cache state between tests."""
    _GEMINI_SYNTHESIS_CACHE.clear()
    yield
    _GEMINI_SYNTHESIS_CACHE.clear()


# ---------------------------------------------------------------------------
# 1. Successful Gemini Call with Telemetry and Token Counting
# ---------------------------------------------------------------------------
def test_gemini_successful_call_with_telemetry():
    """Verify successful synthesis produces clean text, measured tokens, and computed cost."""
    mock_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Article 14 guarantees equality before the law."}]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 150,
            "candidatesTokenCount": 40,
            "totalTokenCount": 190,
        },
    }
    mock_resp = _make_mock_response(200, mock_payload)

    with patch("requests.Session.post", return_value=mock_resp):
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 14",
            api_key="AQ.mock_valid_key_12345",
            article_label="Article 14",
        )

    assert diag.success is True
    assert diag.error_type is None
    assert diag.prompt_tokens == 150
    assert diag.output_tokens == 40
    assert diag.total_tokens == 190
    assert diag.estimated_cost_usd > 0.0
    assert "equality before the law" in diag.text
    assert diag.is_fallback is False


# ---------------------------------------------------------------------------
# 2. Bounded Timeout Handling (Connect & Read Timeouts)
# ---------------------------------------------------------------------------
def test_gemini_read_timeout_bounded():
    """Verify that requests.Timeout is caught as error_type='timeout' and bounded without hanging."""
    t0 = time.perf_counter()
    with patch("requests.Session.post", side_effect=requests.Timeout("Read timed out")), patch("time.sleep"):
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 21",
            api_key="AQ.mock_valid_key_12345",
            article_label="Article 21",
            timeout=(2.0, 3.0),
            models=["gemini-3.1-flash-lite-preview"],
            max_retries_per_model=0,
        )
    elapsed = time.perf_counter() - t0

    assert diag.success is False
    assert diag.error_type == "timeout"
    assert "Timed Out" in diag.user_friendly_message
    assert diag.is_fallback is True
    assert elapsed < 5.0, "Timeout handling must not hang indefinitely"


# ---------------------------------------------------------------------------
# 3. Bounded Retry on Transient Errors (HTTP 503)
# ---------------------------------------------------------------------------
def test_gemini_bounded_retry_on_503():
    """Verify that transient 503 errors are retried up to configured bound with exponential backoff."""
    mock_503 = _make_mock_response(503, {"error": {"message": "Service Unavailable", "status": "UNAVAILABLE"}})
    mock_success = _make_mock_response(
        200,
        {
            "candidates": [{"content": {"parts": [{"text": "Recovered response after retry"}]}}],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 20, "totalTokenCount": 70},
        },
    )

    with patch("requests.Session.post", side_effect=[mock_503, mock_success]) as mock_post, patch("time.sleep") as mock_sleep:
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 19",
            api_key="AQ.mock_valid_key_12345",
            article_label="Article 19",
            models=["gemini-3.1-flash-lite-preview"],
            max_retries_per_model=1,
        )

    assert diag.success is True
    assert diag.retry_count == 1
    assert "Recovered response" in diag.text
    assert mock_post.call_count == 2
    assert mock_sleep.called


# ---------------------------------------------------------------------------
# 4. Fast Fail on Permanent Client Errors (400, 401, 403, 404)
# ---------------------------------------------------------------------------
def test_gemini_no_retry_on_client_errors():
    """Verify permanent errors (401 Unauthorized, 403 Forbidden, 404) fail fast without retrying."""
    mock_401 = _make_mock_response(401, {"error": {"message": "API key not valid", "status": "UNAUTHENTICATED"}})

    with patch("requests.Session.post", return_value=mock_401) as mock_post, patch("time.sleep") as mock_sleep:
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 32",
            api_key="AQ.invalid_key",
            article_label="Article 32",
            models=["gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite"],
            max_retries_per_model=2,
        )

    assert diag.success is False
    assert diag.error_type == "auth_invalid"
    # Permanent error must abort immediately without trying fallback models or retrying
    assert mock_post.call_count == 1
    assert not mock_sleep.called


# ---------------------------------------------------------------------------
# 5. Exact-Query Caching (Hit avoids API call; Zero cost)
# ---------------------------------------------------------------------------
def test_gemini_exact_cache_hit_prevents_api_call():
    """Verify second identical query hits fast in-memory cache and makes zero network calls."""
    kb = get_knowledge_base()
    art14 = kb.lookup_article("14")

    mock_200 = _make_mock_response(
        200,
        {
            "candidates": [{"content": {"parts": [{"text": "Article 14 equality analysis"}]}}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 30, "totalTokenCount": 130},
        },
    )

    with patch("requests.Session.post", return_value=mock_200) as mock_post:
        # First call: cache miss, hits API
        res1 = generate_gemini_with_diagnostics(
            question="What is the significance of Article 14?",
            retrieved_docs=[art14],
            api_key="AQ.mock_valid_key",
            force_gemini=True,
        )
        assert res1.success is True
        assert res1.cache_status == "miss"
        assert mock_post.call_count == 1

        # Second call: cache hit, zero API calls
        res2 = generate_gemini_with_diagnostics(
            question="What is the significance of Article 14?",
            retrieved_docs=[art14],
            api_key="AQ.mock_valid_key",
            force_gemini=True,
        )
        assert res2.success is True
        assert res2.cache_status == "hit"
        assert res2.estimated_cost_usd == 0.0
        assert mock_post.call_count == 1, "Cache hit must NOT call requests.Session.post"


# ---------------------------------------------------------------------------
# 6. Unsuccessful / Timeout Responses Never Stored in Cache
# ---------------------------------------------------------------------------
def test_gemini_timeout_never_cached():
    """Verify that a timed-out call is NOT saved to cache, allowing subsequent retries."""
    kb = get_knowledge_base()
    art21 = kb.lookup_article("21")

    with patch("requests.Session.post", side_effect=requests.Timeout("Read timed out")), patch("time.sleep"):
        res = generate_gemini_with_diagnostics(
            question="Explain the nuances of Article 21",
            retrieved_docs=[art21],
            api_key="AQ.mock_valid_key",
            force_gemini=True,
        )

    assert res.success is False
    assert len(_GEMINI_SYNTHESIS_CACHE) == 0, "Failed / timed out responses must never be cached"


# ---------------------------------------------------------------------------
# 7. Graceful Fallback Preserves PDF Rule-Based Answer
# ---------------------------------------------------------------------------
def test_fallback_rendering_preserves_pdf_answer():
    """Verify that if Gemini times out, the verified local PDF rule-based answer is returned with fallback indicators."""
    kb = get_knowledge_base()
    art368 = kb.lookup_article("368")
    verified_pdf_rule_answer = (
        "### Article 368: Power of Parliament to amend the Constitution\n"
        "Parliament may in exercise of its constituent power amend by way of addition, variation or repeal..."
    )

    with patch("requests.Session.post", side_effect=requests.Timeout("Connection timed out")), patch("time.sleep"):
        res = generate_gemini_with_diagnostics(
            question="What is Article 368 and its amendment procedure?",
            retrieved_docs=[art368],
            api_key="AQ.mock_valid_key",
            fallback_rule_based_answer=verified_pdf_rule_answer,
            force_gemini=True,
        )

    assert res.success is False
    assert res.is_fallback is True
    assert res.fallback_source == "verified_rule_based_pdf"
    assert "timed out" in res.text.lower()
    assert "Power of Parliament to amend the Constitution" in res.text
    assert "100% verified against your local Constitution PDF" in res.text


# ---------------------------------------------------------------------------
# 8. Cost-Aware Routing for Simple Verbatim Lookups
# ---------------------------------------------------------------------------
def test_cost_aware_routing_for_exact_article_lookup():
    """Verify simple verbatim article lookups use deterministic PDF engine ($0 cost, 0 network calls)."""
    kb = get_knowledge_base()
    art14 = kb.lookup_article("14")

    # Pure exact lookup: "what is article 14"
    assert should_route_to_deterministic_pdf("what is article 14", retrieved_docs=[art14]) is True

    # Complex comparative or analytical query: should NOT route to pure deterministic
    assert should_route_to_deterministic_pdf("Compare Article 14 and Article 19 in detail", retrieved_docs=[art14]) is False
    assert should_route_to_deterministic_pdf("Why was Article 14 included in Part III?", retrieved_docs=[art14]) is False

    with patch("requests.Session.post") as mock_post:
        res = generate_gemini_with_diagnostics(
            question="what is article 14",
            retrieved_docs=[art14],
            fallback_rule_based_answer="Verbatim Article 14 Equality before law",
            force_gemini=False,
        )

    assert res.success is True
    assert res.routed_to == "deterministic_pdf"
    assert res.estimated_cost_usd == 0.0
    assert mock_post.call_count == 0, "Cost-aware routing must avoid external LLM calls"


# ---------------------------------------------------------------------------
# 9. Inflight Concurrent Request Deduplication
# ---------------------------------------------------------------------------
def test_concurrent_inflight_request_deduplication():
    """Verify that multiple concurrent threads querying the same prompt share the request and hit cache."""
    kb = get_knowledge_base()
    art14 = kb.lookup_article("14")

    call_count = 0

    def mock_post_delayed(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        time.sleep(0.1)  # Simulate network latency
        return _make_mock_response(
            200,
            {
                "candidates": [{"content": {"parts": [{"text": "Thread-safe Article 14 analysis"}]}}],
                "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 25, "totalTokenCount": 105},
            },
        )

    with patch("requests.Session.post", side_effect=mock_post_delayed):
        results = [None, None]

        def worker(idx):
            results[idx] = generate_gemini_with_diagnostics(
                question="Explain Article 14 fundamental principles",
                retrieved_docs=[art14],
                api_key="AQ.mock_valid_key",
                force_gemini=True,
            )

        t1 = threading.Thread(target=worker, args=(0,))
        t2 = threading.Thread(target=worker, args=(1,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    assert results[0].success is True
    assert results[1].success is True
    # The lock ensures only 1 API call is made; the second thread gets the cached result!
    assert call_count == 1, f"Expected 1 network call due to inflight deduplication, got {call_count}"


# ---------------------------------------------------------------------------
# 10. Passage Deduplication in Retrieved Evidence
# ---------------------------------------------------------------------------
def test_passage_deduplication_in_evidence():
    """Verify that format_retrieved_evidence filters out duplicate/overlapping chunks."""
    duplicate_chunks = [
        {"article_number": "14", "article_title": "Equality", "constitutional_text": "The State shall not deny equality..."},
        {"article_number": "14", "article_title": "Equality", "constitutional_text": "The State shall not deny equality..."},
        {"article_number": "21", "article_title": "Life", "constitutional_text": "No person shall be deprived of his life..."},
    ]
    evidence_str = format_retrieved_evidence(duplicate_chunks, max_chunks=3)
    # Article 14 should appear only ONCE in the evidence
    assert evidence_str.count("[Document #1: Article 14") == 1
    assert "Document #3" not in evidence_str, "Duplicate chunk must be filtered out"


# ---------------------------------------------------------------------------
# 11. Conversation History Compacting & Banner Filtering
# ---------------------------------------------------------------------------
def test_conversation_history_compacting():
    """Verify that format_conversation_history filters out warning/error banners and keeps text concise."""
    history = [
        HumanMessage(content="What is Article 14?"),
        AIMessage(content="Article 14 ensures equality."),
        HumanMessage(content="What is Article 370A?"),
        AIMessage(content="⚠️ Gemini AI Request Timed Out (25.7s elapsed for Article 368)"),
        HumanMessage(content="Explain it simply"),
    ]
    formatted = format_conversation_history(history, max_turns=2)
    # The timeout warning banner must be stripped from LLM prompt history
    assert "Timed Out" not in formatted
    assert "Explain it simply" in formatted
