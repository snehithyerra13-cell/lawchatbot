"""Mocked Unit Tests for Gemini AI Diagnostics, Error Handling, and Retry Policies.

All tests in this module use unittest.mock to simulate API responses.
No live Google Gemini API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.retrieval.gemini_service import (
    FALLBACK_MODELS,
    GeminiDiagnosticResult,
    classify_http_error,
    clean_constitutional_evidence_text,
    compute_backoff_delay,
    execute_gemini_request_with_diagnostics,
    format_retrieved_evidence,
    format_user_friendly_error,
    generate_gemini_with_diagnostics,
    is_retryable_error,
    sanitize_error_message,
)
from app.retrieval.knowledge_base import get_knowledge_base


def _create_mock_response(status_code: int, json_data: dict, headers: dict = None) -> requests.Response:
    """Helper to construct a mock requests.Response object."""
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = json.dumps(json_data).encode("utf-8")
    resp.headers.update(headers or {})
    return resp


# ---------------------------------------------------------------------------
# 1. Successful 200 Generation
# ---------------------------------------------------------------------------
def test_mock_success_200():
    """Verify successful generation with 200 OK returns clean text and success=True."""
    mock_body = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Article 14 ensures equality before law."}]
                }
            }
        ]
    }
    mock_resp = _create_mock_response(200, mock_body)

    with patch("requests.Session.post", return_value=mock_resp):
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 14",
            api_key="AQ.mock_test_key_1234567890abcdefghijklmnopqrstuvwxyz",
            article_label="Article 14",
            models=["gemini-flash-lite-latest"],
        )

    assert diag.success is True
    assert diag.status_code == 200
    assert "equality before law" in diag.text
    assert diag.model_used == "gemini-flash-lite-latest"
    assert diag.error_type is None
    assert diag.retry_count == 0


# ---------------------------------------------------------------------------
# 2. 429 Rate Limit / Quota Exceeded
# ---------------------------------------------------------------------------
def test_mock_quota_exceeded_429():
    """Verify HTTP 429 quota exhaustion is accurately classified and not mislabeled."""
    mock_body = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota, please check your plan and billing details.",
            "status": "RESOURCE_EXHAUSTED",
        }
    }
    mock_resp = _create_mock_response(429, mock_body, headers={"Retry-After": "1"})

    with patch("requests.Session.post", return_value=mock_resp), patch("time.sleep"):
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 14",
            api_key="AQ.mock_test_key_1234567890abcdefghijklmnopqrstuvwxyz",
            article_label="Article 14",
            models=["gemini-flash-lite-latest"],
            max_retries_per_model=1,
        )

    assert diag.success is False
    assert diag.status_code == 429
    assert diag.error_type == "quota_exceeded"
    assert "429" in diag.user_friendly_message
    assert "Quota" in diag.user_friendly_message or "quota" in diag.user_friendly_message
    assert diag.retry_count == 1  # Bounded retry attempted for transient 429


# ---------------------------------------------------------------------------
# 3. 400 Bad Request (Non-retryable)
# ---------------------------------------------------------------------------
def test_mock_bad_request_400_no_retry():
    """Verify HTTP 400 is classified as bad_request and fails fast without retrying."""
    mock_body = {
        "error": {
            "code": 400,
            "message": "Invalid prompt argument contents.",
            "status": "INVALID_ARGUMENT",
        }
    }
    mock_resp = _create_mock_response(400, mock_body)

    with patch("requests.Session.post", return_value=mock_resp) as mock_post:
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 14",
            api_key="AQ.mock_test_key_1234567890abcdefghijklmnopqrstuvwxyz",
            article_label="Article 14",
            models=["gemini-flash-lite-latest"],
            max_retries_per_model=2,
        )

    assert diag.success is False
    assert diag.status_code == 400
    assert diag.error_type == "bad_request"
    assert diag.retry_count == 0  # 400 MUST NOT be retried
    assert mock_post.call_count == 1  # Exactly 1 call made


# ---------------------------------------------------------------------------
# 4. 401 Unauthorized (Invalid Key - Non-retryable)
# ---------------------------------------------------------------------------
def test_mock_auth_invalid_401_no_retry():
    """Verify HTTP 401 is classified as auth_invalid and aborts immediately."""
    mock_body = {
        "error": {
            "code": 401,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "UNAUTHENTICATED",
        }
    }
    mock_resp = _create_mock_response(401, mock_body)

    with patch("requests.Session.post", return_value=mock_resp) as mock_post:
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 14",
            api_key="AQ.mock_test_key_1234567890abcdefghijklmnopqrstuvwxyz",
            article_label="Article 14",
            models=["gemini-flash-lite-latest", "gemini-3.8-flash"],
            max_retries_per_model=2,
        )

    assert diag.success is False
    assert diag.status_code == 401
    assert diag.error_type == "auth_invalid"
    assert "401" in diag.user_friendly_message
    assert "Authentication Failure" in diag.user_friendly_message
    assert diag.retry_count == 0
    # Permanent auth error must not waste time testing secondary models
    assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# 5. 403 Forbidden / Permission Denied
# ---------------------------------------------------------------------------
def test_mock_permission_denied_403_no_retry():
    """Verify HTTP 403 is classified as permission_denied and does not retry."""
    mock_body = {
        "error": {
            "code": 403,
            "message": "Generative Language API has not been used in project before.",
            "status": "PERMISSION_DENIED",
        }
    }
    mock_resp = _create_mock_response(403, mock_body)

    with patch("requests.Session.post", return_value=mock_resp) as mock_post:
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 14",
            api_key="AQ.mock_test_key_1234567890abcdefghijklmnopqrstuvwxyz",
            article_label="Article 14",
            models=["gemini-flash-lite-latest"],
        )

    assert diag.success is False
    assert diag.status_code == 403
    assert diag.error_type == "permission_denied"
    assert "403" in diag.user_friendly_message
    assert diag.retry_count == 0


# ---------------------------------------------------------------------------
# 6. 404 Model Not Found -> Fails Fast to Next Fallback Model
# ---------------------------------------------------------------------------
def test_mock_model_not_found_404_fallback():
    """Verify 404 model discontinued fails fast on model 1 and succeeds on model 2."""
    mock_resp_404 = _create_mock_response(
        404,
        {"error": {"code": 404, "message": "Model discontinued.", "status": "NOT_FOUND"}},
    )
    mock_resp_200 = _create_mock_response(
        200,
        {"candidates": [{"content": {"parts": [{"text": "Article 14 fallback synthesis."}]}}]},
    )

    with patch("requests.Session.post", side_effect=[mock_resp_404, mock_resp_200]) as mock_post:
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 14",
            api_key="AQ.mock_test_key_1234567890abcdefghijklmnopqrstuvwxyz",
            article_label="Article 14",
            models=["gemini-old-model", "gemini-flash-lite-latest"],
            max_retries_per_model=1,
        )

    assert diag.success is True
    assert diag.status_code == 200
    assert diag.model_used == "gemini-flash-lite-latest"
    assert "gemini-old-model" in diag.models_attempted
    assert "gemini-flash-lite-latest" in diag.models_attempted
    # Model 1 was not retried after 404; code moved directly to Model 2
    assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# 7. 500 / 503 Server Error -> Transient Retry with Backoff
# ---------------------------------------------------------------------------
def test_mock_server_error_503_transient_retry():
    """Verify 503 Service Unavailable is retried with backoff and succeeds on retry."""
    mock_resp_503 = _create_mock_response(
        503,
        {"error": {"code": 503, "message": "The service is temporarily overloaded.", "status": "UNAVAILABLE"}},
    )
    mock_resp_200 = _create_mock_response(
        200,
        {"candidates": [{"content": {"parts": [{"text": "Recovered after 503."}]}}]},
    )

    with patch("requests.Session.post", side_effect=[mock_resp_503, mock_resp_200]), patch("time.sleep") as mock_sleep:
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 14",
            api_key="AQ.mock_test_key_1234567890abcdefghijklmnopqrstuvwxyz",
            article_label="Article 14",
            models=["gemini-flash-lite-latest"],
            max_retries_per_model=1,
        )

    assert diag.success is True
    assert diag.status_code == 200
    assert diag.retry_count == 1
    assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# 8. Timeout Handling
# ---------------------------------------------------------------------------
def test_mock_timeout_handling():
    """Verify requests.Timeout is caught, categorized as timeout, and retried once."""
    with patch("requests.Session.post", side_effect=requests.Timeout("Connection timed out")), patch("time.sleep"):
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 14",
            api_key="AQ.mock_test_key_1234567890abcdefghijklmnopqrstuvwxyz",
            article_label="Article 14",
            models=["gemini-flash-lite-latest"],
            max_retries_per_model=1,
        )

    assert diag.success is False
    assert diag.error_type == "timeout"
    assert "Timed Out" in diag.user_friendly_message
    assert diag.retry_count == 1


# ---------------------------------------------------------------------------
# 9. Malformed / Empty Candidates Response
# ---------------------------------------------------------------------------
def test_mock_malformed_empty_candidates():
    """Verify 200 OK with empty candidates is handled cleanly as malformed_response."""
    mock_resp = _create_mock_response(200, {"candidates": []})

    with patch("requests.Session.post", return_value=mock_resp):
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 14",
            api_key="AQ.mock_test_key_1234567890abcdefghijklmnopqrstuvwxyz",
            article_label="Article 14",
            models=["gemini-flash-lite-latest"],
        )

    assert diag.success is False
    assert diag.error_type == "malformed_response"


# ---------------------------------------------------------------------------
# 10. Missing Configuration (Empty / None API Key)
# ---------------------------------------------------------------------------
def test_missing_api_key_configuration():
    """Verify empty or missing key returns missing_key error without making network calls."""
    mock_settings = MagicMock(gemini_api_key="")
    with patch("requests.Session.post") as mock_post, patch("app.retrieval.gemini_service.get_settings", return_value=mock_settings):
        diag = execute_gemini_request_with_diagnostics(
            prompt="Explain Article 14",
            api_key="",
            article_label="Article 14",
        )

    assert diag.success is False
    assert diag.error_type == "missing_key"
    assert "Not Configured" in diag.user_friendly_message
    assert mock_post.call_count == 0


# ---------------------------------------------------------------------------
# 11. Article 14 Evidence Prompt Grounding
# ---------------------------------------------------------------------------
def test_article_14_evidence_prompt_grounding():
    """Verify Article 14 constitutional text from knowledge base is grounded into prompt."""
    kb = get_knowledge_base()
    art14 = kb.lookup_article("14")
    assert art14 is not None
    assert art14.get("article_number") == "14"

    evidence = format_retrieved_evidence([art14])
    assert "Article 14" in evidence
    assert "Equality before law" in evidence
    assert "State shall not deny" in evidence
    assert "Part: PART III" in evidence


# ---------------------------------------------------------------------------
# 12. Secret Sanitization & Error Masking
# ---------------------------------------------------------------------------
def test_secret_sanitization():
    """Verify API keys and sensitive tokens are strictly redacted from error messages."""
    fake_key = "AQ.1234567890abcdefghijklmnopqrstuvwxyz_secret_token"
    raw_error = f"GET https://generativelanguage.googleapis.com/v1beta/models?key={fake_key} failed: invalid key {fake_key}"
    sanitized = sanitize_error_message(raw_error, fake_key)

    assert fake_key not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


# ---------------------------------------------------------------------------
# 13. Retry Delay Calculation & Backoff Jitter
# ---------------------------------------------------------------------------
def test_compute_backoff_delay():
    """Verify backoff calculation respects Retry-After and bounds delay."""
    delay_with_header = compute_backoff_delay(0, headers={"Retry-After": "2.5"})
    assert 2.4 <= delay_with_header <= 2.6

    delay_exp = compute_backoff_delay(1, base_delay=0.5, max_delay=3.0)
    assert 1.0 <= delay_exp <= 2.0
