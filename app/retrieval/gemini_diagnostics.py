"""Gemini AI Safe Diagnostic & Health-Check Utility for ConstitutionGPT.

Usage:
    python -m app.retrieval.gemini_diagnostics
    python -m app.retrieval.gemini_diagnostics --live
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from app.retrieval.gemini_service import (
    FALLBACK_MODELS,
    execute_gemini_request_with_diagnostics,
    format_retrieved_evidence,
    get_http_session,
    sanitize_error_message,
)
from app.retrieval.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


def mask_key(key: Optional[str]) -> str:
    """Safely mask key for display without exposing secrets."""
    if not key:
        return "(not configured)"
    k = key.strip()
    if len(k) <= 8:
        return "****"
    return f"{k[:3]}...{k[-4:]} ({len(k)} chars)"


def run_gemini_health_check(
    live_test: bool = False,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute non-destructive health-check and optional single live generation test.

    Returns a structured dictionary of diagnostic results.
    """
    settings = get_settings()
    key = (settings.gemini_api_key or "").strip()

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "key_configured": bool(key),
        "key_masked": mask_key(key),
        "api_endpoint": "https://generativelanguage.googleapis.com/v1beta",
        "configured_fallback_models": list(FALLBACK_MODELS),
        "models_endpoint_accessible": False,
        "available_gemini_models": [],
        "article_14_evidence_grounded": False,
        "article_14_evidence_chars": 0,
        "live_test_requested": live_test,
        "live_test_result": None,
    }

    # 1. Test Article 14 Evidence extraction from Knowledge Base
    try:
        kb = get_knowledge_base()
        art14 = kb.lookup_article("14")
        if art14 and art14.get("constitutional_text"):
            evidence = format_retrieved_evidence([art14])
            report["article_14_evidence_grounded"] = bool("Article 14" in evidence)
            report["article_14_evidence_chars"] = len(evidence)
    except Exception as e:
        report["article_14_evidence_error"] = str(e)

    # 2. Check API Key validity & list accessible models via GET /v1beta/models
    if key:
        session = get_http_session()
        models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        try:
            t0 = time.perf_counter()
            resp = session.get(models_url, timeout=10)
            latency = round(time.perf_counter() - t0, 3)
            report["models_endpoint_status"] = resp.status_code
            report["models_endpoint_latency"] = latency

            if resp.status_code == 200:
                report["models_endpoint_accessible"] = True
                data = resp.json()
                active_models = [
                    m.get("name", "").replace("models/", "")
                    for m in data.get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                report["available_gemini_models"] = active_models
            else:
                err_dict = resp.json().get("error", {}) if resp.text.startswith("{") else {}
                report["models_endpoint_error"] = {
                    "code": err_dict.get("code", resp.status_code),
                    "status": err_dict.get("status", ""),
                    "message": sanitize_error_message(err_dict.get("message", resp.text[:200]), key),
                }
        except Exception as net_err:
            report["models_endpoint_error"] = {
                "type": "network_or_timeout",
                "message": sanitize_error_message(str(net_err), key),
            }

    # 3. Optional single live generation test (only when explicitly requested)
    if live_test and key:
        kb = get_knowledge_base()
        art14 = kb.lookup_article("14")
        docs = [art14] if art14 else []
        evidence_str = format_retrieved_evidence(docs)
        prompt = (
            "You are an expert constitutional legal assistant specializing in the Constitution of India.\n"
            f"=== CONSTITUTIONAL PDF EVIDENCE ===\n{evidence_str}\n\n"
            "=== USER QUESTION ===\n"
            "Explain Article 14 of the Constitution of India in 2 concise sentences."
        )
        models_to_test = [model] if model else list(FALLBACK_MODELS)
        diag = execute_gemini_request_with_diagnostics(
            prompt=prompt,
            api_key=key,
            article_label="Article 14",
            timeout=15,
            models=models_to_test,
        )
        report["live_test_result"] = {
            "success": diag.success,
            "status_code": diag.status_code,
            "error_type": diag.error_type,
            "model_used": diag.model_used,
            "models_attempted": diag.models_attempted,
            "duration_seconds": diag.duration_seconds,
            "retry_count": diag.retry_count,
            "provider_message": sanitize_error_message(diag.provider_message or "", key),
            "generated_text_preview": diag.text[:200] if diag.success else "(generation failed)",
        }

    return report


def print_diagnostic_report(report: Dict[str, Any]) -> None:
    """Print clean formatted diagnostic summary table to stdout."""
    print("=" * 68)
    print(" CONSTITUTIONGPT -- GEMINI AI DIAGNOSTIC & HEALTH-CHECK REPORT")
    print("=" * 68)
    print(f" Timestamp:                 {report['timestamp']}")
    print(f" API Key Configured:        {'[YES]' if report['key_configured'] else '[NO]'}")
    print(f" API Key Masked:            {report['key_masked']}")
    print(f" Article 14 Grounding PDF:  {'[VERIFIED]' if report['article_14_evidence_grounded'] else '[FAILED]'} ({report['article_14_evidence_chars']} chars)")
    status_text = "HTTP 200 OK [ACCESSIBLE]" if report.get("models_endpoint_accessible") else f"HTTP {report.get('models_endpoint_status', 'ERR')} [FAILED]"
    print(f" Models Endpoint (GET):     {status_text}")
    if report.get("available_gemini_models"):
        models_sample = ", ".join(report["available_gemini_models"][:5])
        print(f" Active Models in Project:  {len(report['available_gemini_models'])} available ({models_sample}...)")
    if report.get("models_endpoint_error"):
        print(f" Endpoint Error:            {report['models_endpoint_error']}")

    print("-" * 68)
    print(" Configured Fallback Priority:")
    for idx, m in enumerate(report["configured_fallback_models"]):
        is_avail = m in report.get("available_gemini_models", [])
        status_flag = "[AVAILABLE]" if is_avail else "[FALLBACK]"
        print(f"   #{idx+1}: {m} {status_flag}")

    if report.get("live_test_requested"):
        print("-" * 68)
        print(" Live Generation Test (Article 14):")
        res = report.get("live_test_result") or {}
        print(f"   Success:                 {'[TRUE]' if res.get('success') else '[FALSE]'}")
        print(f"   HTTP Status:             {res.get('status_code')}")
        print(f"   Error Category:          {res.get('error_type') or 'None'}")
        print(f"   Model Used:              {res.get('model_used')}")
        print(f"   Models Attempted:        {res.get('models_attempted')}")
        print(f"   Duration:                {res.get('duration_seconds')}s")
        print(f"   Retries:                 {res.get('retry_count')}")
        if res.get("provider_message"):
            print(f"   Provider Message:        {res.get('provider_message')[:120]}")
        if res.get("success"):
            print(f"   Text Preview:            {res.get('generated_text_preview')}")
    else:
        print("-" * 68)
        print(" Note: Live generation test was not requested. Run with --live to test.")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description="ConstitutionGPT Gemini Diagnostic Utility")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Perform exactly one live test generation request for Article 14",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to test (e.g. gemini-flash-lite-latest)",
    )
    args = parser.parse_args()

    report = run_gemini_health_check(live_test=args.live, model=args.model)
    print_diagnostic_report(report)

    if args.live:
        res = report.get("live_test_result") or {}
        sys.exit(0 if res.get("success") else 1)


if __name__ == "__main__":
    main()
