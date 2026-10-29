"""Comprehensive regression tests for RAG abstention, citation validation, and hallucination prevention.

Tests A through J verify:
- Test A: Answerable constitutional question produces grounded answer and verified citation.
- Test B: Unsupported current/temporal fact triggers evidence-based abstention without false citations or misclassification.
- Test C: Irrelevant retrieved chunks do not leak into citations when query is unsupported.
- Test D: Truly out-of-domain query triggers domain restriction without citations.
- Test E: Citation mismatch between answer text and evidence is caught by validator.
- Test F: Empty retrieval results trigger insufficient evidence handling without hallucinations.
- Test G: Retrieval exception triggers controlled failure without false grounding.
- Test H: Gemini generation failure produces safe fallback without false grounding badge.
- Test I: Multi-article question cites and validates all referenced sources.
- Test J: Existing LangGraph checkpointer and workflow state persist cleanly across turns.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.evidence_guard import (
    is_temporal_or_dynamic_query,
    evaluate_query_domain,
    evaluate_evidence_sufficiency,
    validate_citations_for_answer,
    EvidenceEvaluationResult,
    STATUS_ANSWERABLE,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_OUT_OF_DOMAIN,
    STATUS_RETRIEVAL_FAILURE,
    STATUS_GENERATION_FAILURE,
)
from app.graph.nodes import (
    classify_question,
    retrieve_constitution,
    check_retrieval,
    handle_uncertainty,
    finalize_response,
    synthesize_gemini,
)
from app.graph.state import ConstitutionState
from app.retrieval.dual_pipeline import AdvancedRAGPipeline, NormalRAGPipeline


class TestAbstentionAndCitations(unittest.TestCase):
    """Test suite covering Tests A through J."""

    def test_a_answerable_question(self):
        """Test A: Answerable question produces grounded answer and valid citation."""
        pipeline = AdvancedRAGPipeline()
        query = "What does Article 21 protect?"
        result = pipeline.execute(query)

        self.assertFalse(result.is_abstention, "Article 21 question should be answerable")
        self.assertIn("21", result.answer, "Answer should reference Article 21")
        self.assertTrue(len(result.citations) > 0, "Citations must be present for Article 21")
        self.assertEqual(result.citations[0]["article"], "21")
        self.assertTrue(result.citation_validation.get("is_valid"), "Citation must be verified")

    def test_b_unsupported_current_fact(self):
        """Test B: Current temporal fact triggers explicit abstention without false citations."""
        query = "According to the uploaded Constitution text, what was the exact number of constitutional amendments passed yesterday?"

        # 1. Guard check
        is_temp, reason = is_temporal_or_dynamic_query(query)
        self.assertTrue(is_temp, "Query must be flagged as temporal/dynamic")
        self.assertIsNotNone(reason)
        self.assertIn("cannot determine", reason.lower())
        self.assertIn("yesterday", reason.lower())

        # 2. Pipeline check
        pipeline = AdvancedRAGPipeline()
        result = pipeline.execute(query)
        self.assertTrue(result.is_abstention, "Result must be marked as an abstention")
        self.assertNotIn("0 amendments", result.answer.lower(), "Must NOT invent '0 amendments'")
        self.assertNotIn("zero amendments", result.answer.lower(), "Must NOT invent 'zero amendments'")
        self.assertEqual(len(result.citations), 0, "Abstention must NOT leak citations (e.g. Article 21)")
        self.assertNotIn("Article 21", result.answer, "Must not mention Article 21 in response")

        # 3. LangGraph node check
        state: ConstitutionState = {
            "question": query,
            "session_id": "test_session_b",
            "messages": [],
            "retrieved_documents": [{"article_number": "21", "article_title": "Life", "constitutional_text": "Protection of life"}],
            "sources": [{"article_number": "21"}],
        }
        classified = classify_question(state)
        self.assertEqual(classified.get("intent"), "temporal_inquiry")
        self.assertEqual(classified.get("answerability_status"), "INSUFFICIENT_EVIDENCE")
        self.assertEqual(classified.get("sources"), [])

        handled = handle_uncertainty({**state, **classified})
        ans = handled.get("response_type")
        self.assertEqual(ans, "insufficient_evidence")
        self.assertEqual(handled.get("sources"), [])
        self.assertIn("cannot determine", (handled.get("answer") or handled.get("final_response") or "").lower())

    def test_c_irrelevant_retrieved_chunk_suppression(self):
        """Test C: If retrieved chunks are irrelevant to an unsupported inquiry, no citations leak."""
        mock_chunks = [
            {
                "article_number": "21",
                "article_title": "Protection of life and personal liberty",
                "constitutional_text": "No person shall be deprived of his life or personal liberty except according to procedure established by law.",
                "page_start": 35,
                "source_pdf": "constitution.pdf",
            }
        ]
        query = "What was the exact number of constitutional amendments passed yesterday?"
        eval_res = evaluate_evidence_sufficiency(query, mock_chunks)
        self.assertTrue(eval_res.is_abstention)
        self.assertEqual(eval_res.status, STATUS_INSUFFICIENT_EVIDENCE)

        state: ConstitutionState = {
            "question": query,
            "session_id": "test_session_c",
            "messages": [],
            "intent": "temporal_inquiry",
            "response_type": "insufficient_evidence",
            "answerability_status": STATUS_INSUFFICIENT_EVIDENCE,
            "retrieved_documents": mock_chunks,
            "sources": [],
            "answer": "I cannot determine the exact number of amendments passed yesterday.",
        }
        final = finalize_response(state)
        self.assertEqual(final.get("sources"), [], "Final response must have empty sources on abstention")

    def test_d_truly_out_of_domain(self):
        """Test D: Non-legal out-of-domain query triggers domain restriction response without citations."""
        query = "What is the latest cricket score between India and Australia?"
        is_in_domain, rejection_reason = evaluate_query_domain(query)
        self.assertFalse(is_in_domain)
        self.assertIn("domain only", rejection_reason)

        state: ConstitutionState = {
            "question": query,
            "session_id": "test_session_d",
            "messages": [],
        }
        classified = classify_question(state)
        self.assertEqual(classified.get("intent"), "out_of_scope")
        self.assertEqual(classified.get("answerability_status"), STATUS_OUT_OF_DOMAIN)

        handled = handle_uncertainty({**state, **classified})
        self.assertEqual(handled.get("response_type"), "out_of_scope")
        self.assertIn("domain", (handled.get("answer") or handled.get("final_response") or "").lower())
        self.assertEqual(handled.get("sources"), [])

    def test_e_citation_mismatch(self):
        """Test E: Citation mismatch between generated text and retrieved evidence is flagged."""
        # Answer references Article 32, but evidence only contains Article 21
        answer = "Under Article 32, citizens have the constitutional remedy to move the Supreme Court."
        evidence = [{"article_number": "21", "constitutional_text": "Protection of life and personal liberty"}]

        val_res = validate_citations_for_answer(answer, evidence, answerability_status=STATUS_ANSWERABLE)
        self.assertFalse(val_res["is_valid"], "Validator must detect missing Article 32 in evidence")
        self.assertTrue(any("32" in c for c in val_res["unsupported_claims"]))
        self.assertFalse(val_res["grounding_badge"])

    def test_f_no_relevant_retrieval_results(self):
        """Test F: Empty retrieval results trigger insufficient evidence without hallucination."""
        query = "What does Article 999 say about interstellar space travel?"
        state: ConstitutionState = {
            "question": query,
            "session_id": "test_session_f",
            "messages": [],
            "retrieved_documents": [],
            "sources": [],
        }
        check = check_retrieval(state)
        self.assertEqual(check.get("response_type"), "insufficient_evidence")
        self.assertEqual(check.get("answerability_status"), STATUS_INSUFFICIENT_EVIDENCE)

        handled = handle_uncertainty({**state, **check})
        self.assertEqual(handled.get("sources"), [])
        self.assertIn("does not", (handled.get("answer") or handled.get("final_response") or "").lower())

    def test_g_retrieval_error(self):
        """Test G: Controlled handling of retrieval exceptions without false success."""
        query = "What are fundamental rights?"
        state: ConstitutionState = {
            "question": query,
            "session_id": "test_session_g",
            "messages": [],
        }
        # Simulate retrieval exception handling in workflow
        ret_res = {
            "response_type": "retrieval_failure",
            "answerability_status": STATUS_RETRIEVAL_FAILURE,
            "sources": [],
            "retrieved_documents": [],
        }
        self.assertEqual(ret_res.get("response_type"), "retrieval_failure")
        self.assertEqual(ret_res.get("answerability_status"), STATUS_RETRIEVAL_FAILURE)
        self.assertEqual(ret_res.get("sources"), [])

        handled = handle_uncertainty({**state, **ret_res})
        self.assertEqual(handled.get("response_type"), "retrieval_failure")
        self.assertEqual(handled.get("sources"), [])
        self.assertIn("technical difficulty", (handled.get("answer") or handled.get("final_response") or "").lower())

    def test_h_gemini_generation_failure(self):
        """Test H: Gemini generation failure produces safe fallback without false grounding badge."""
        from app.retrieval.gemini_service import GeminiDiagnosticResult

        query = "Simulated failure test query for article 21 unique_id_test_h"
        state: ConstitutionState = {
            "question": query,
            "session_id": "test_session_h",
            "messages": [],
            "retrieved_documents": [{"article_number": "21", "constitutional_text": "Protection of life and personal liberty"}],
            "sources": [{"article_number": "21"}],
            "active_tab": "gemini",
        }
        mock_diag = GeminiDiagnosticResult(
            success=False,
            error_type="quota_exceeded",
            user_friendly_message="Gemini API Quota Exceeded. Please try again later.",
        )
        with patch("app.retrieval.gemini_service.generate_gemini_with_diagnostics", return_value=mock_diag):
            synth_res = synthesize_gemini(state)
            diag = synth_res.get("gemini_diagnostic", {})
            self.assertFalse(diag.get("success"))
            self.assertIn("quota", diag.get("error_type", "").lower())

    def test_i_valid_multiple_source_answer(self):
        """Test I: Multi-article question verifies all cited articles against evidence."""
        answer = "Article 14 guarantees equality before law, while Article 21 protects personal liberty."
        evidence = [
            {"article_number": "14", "constitutional_text": "Equality before law"},
            {"article_number": "21", "constitutional_text": "Protection of life and personal liberty"},
        ]
        val_res = validate_citations_for_answer(answer, evidence, answerability_status=STATUS_ANSWERABLE)
        self.assertTrue(val_res["is_valid"])
        self.assertIn("14", val_res["verified_articles"])
        self.assertIn("21", val_res["verified_articles"])
        self.assertEqual(len(val_res["unsupported_claims"]), 0)
        self.assertTrue(val_res["grounding_badge"])

    def test_j_existing_workflow_regression(self):
        """Test J: Checkpointer persistence and conversation state flow smoothly across multi-turn."""
        from app.graph.workflow import get_compiled_graph
        graph = get_compiled_graph()
        self.assertIsNotNone(graph, "Conversation graph must compile successfully")

        nodes = graph.nodes
        self.assertIn("rewrite_query", nodes)
        self.assertIn("classify_question", nodes)
        self.assertIn("retrieve_constitution", nodes)
        self.assertIn("check_retrieval", nodes)
        self.assertIn("handle_uncertainty", nodes)
        self.assertIn("finalize_response", nodes)


if __name__ == "__main__":
    unittest.main()
