"""Comprehensive regression test suite for ConstitutionGPT Follow-Up Context, Domain Guard, and Dual-Engine RAG.

Verifies Tests 1 through 12:
- Test 1: "Explain Article 14" -> "in detail" -> Article 14 explanation, NOT domain restriction.
- Test 2: "Explain Article 14" -> "explain about it in detail and crisp" -> concise but detailed Article 14.
- Test 3: "Explain Article 14" -> "give an example" -> relevant Article 14 example.
- Test 4: "Explain Article 14" -> "make it crisp" -> concise summary of Article 14.
- Test 5: "Explain Article 14" -> "What about Article 21?" -> Article 21 response (no Article 14 bleed).
- Test 6: "Explain Article 14" -> "What is the capital of France?" -> domain restriction rejection.
- Test 7: New thread with "in detail" -> asks for clarification without false citations.
- Test 8: Two independent user sessions -> zero cross-user history leakage.
- Test 9: Advanced RAG and Normal RAG both receive resolved query and execute cleanly.
- Test 10: Streaming and multi-turn execution preserve context cleanly without duplicate messages.
- Test 11: Verbatim PDF mode with "explain in detail" preserves full exact-source fidelity.
- Test 12: Follow-up on unrecorded temporal fact triggers insufficient evidence without false citations.
"""

import os
import sys
import unittest
import uuid

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.workflow import run_constitution_chat, run_constitution_chat_stream
from app.persistence.checkpointer import get_sqlite_checkpointer
from app.retrieval.context_resolver import resolve_follow_up_query
from app.retrieval.dual_pipeline import AdvancedRAGPipeline, NormalRAGPipeline
from app.retrieval.evidence_guard import (
    STATUS_ANSWERABLE,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_OUT_OF_DOMAIN,
    evaluate_query_domain,
)
from langchain_core.messages import AIMessage, HumanMessage


class TestFollowUpContextAndDomainGuard(unittest.TestCase):
    """Test suite covering Tests 1 through 12."""

    def setUp(self):
        self.checkpointer = get_sqlite_checkpointer()

    def test_01_followup_in_detail_not_domain_rejected(self):
        """Test 1: User asks 'Explain Article 14 in simple English', then 'in detail'.
        Must produce a detailed explanation of Article 14, NOT a domain rejection.
        """
        thread_id = f"test_thread_01_{uuid.uuid4().hex[:8]}"

        # Turn 1
        res1 = run_constitution_chat(
            "Explain Article 14 in simple English",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )
        self.assertEqual(res1.get("answerability_status"), STATUS_ANSWERABLE)

        # Turn 2: Follow-up
        res2 = run_constitution_chat(
            "in detail",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        self.assertNotEqual(
            res2.get("answerability_status"),
            STATUS_OUT_OF_DOMAIN,
            "Follow-up 'in detail' must NOT be classified as OUT_OF_DOMAIN",
        )
        self.assertEqual(res2.get("answerability_status"), STATUS_ANSWERABLE)
        self.assertEqual(str(res2.get("article_number")), "14")
        self.assertIn("14", res2.get("answer", ""))
        self.assertTrue(len(res2.get("sources", [])) > 0)
        self.assertNotIn("Please ask questions based on the Constitution", res2.get("answer", ""))

    def test_02_followup_explain_about_it_in_detail_and_crisp(self):
        """Test 2: User asks 'explain about it in detail and crisp' after Article 14.
        Must provide a detailed yet crisp explanation grounded in Article 14.
        """
        thread_id = f"test_thread_02_{uuid.uuid4().hex[:8]}"

        run_constitution_chat(
            "Explain Article 14 in simple English",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        res2 = run_constitution_chat(
            "explain about it in detail and crisp",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        self.assertEqual(res2.get("answerability_status"), STATUS_ANSWERABLE)
        self.assertEqual(str(res2.get("article_number")), "14")
        self.assertIn("14", res2.get("answer", ""))
        self.assertTrue(len(res2.get("sources", [])) > 0)

    def test_03_followup_give_an_example(self):
        """Test 3: User asks 'give an example' after Article 14.
        Must provide relevant examples/scenarios for Article 14.
        """
        thread_id = f"test_thread_03_{uuid.uuid4().hex[:8]}"

        run_constitution_chat(
            "Explain Article 14 in simple English",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        res2 = run_constitution_chat(
            "give an example",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        self.assertEqual(res2.get("answerability_status"), STATUS_ANSWERABLE)
        self.assertEqual(str(res2.get("article_number")), "14")
        self.assertIn("14", res2.get("answer", ""))
        # Verify examples section exists in answer
        ans_text = res2.get("answer", "").lower()
        self.assertTrue("example" in ans_text or "scenario" in ans_text or "practical" in ans_text)

    def test_04_followup_make_it_crisp(self):
        """Test 4: User asks 'make it crisp' after Article 14.
        Must provide a crisp, concise summary of Article 14.
        """
        thread_id = f"test_thread_04_{uuid.uuid4().hex[:8]}"

        run_constitution_chat(
            "Explain Article 14 in simple English",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        res2 = run_constitution_chat(
            "make it crisp",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        self.assertEqual(res2.get("answerability_status"), STATUS_ANSWERABLE)
        self.assertEqual(str(res2.get("article_number")), "14")
        self.assertIn("14", res2.get("answer", ""))
        self.assertIn("Crisp", res2.get("answer", ""))

    def test_05_followup_pivot_to_article_21(self):
        """Test 5: User asks 'What about Article 21?' after Article 14.
        Must switch focus to Article 21 without Article 14 bleeding into active focus.
        """
        thread_id = f"test_thread_05_{uuid.uuid4().hex[:8]}"

        run_constitution_chat(
            "Explain Article 14 in simple English",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        res2 = run_constitution_chat(
            "What about Article 21?",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        self.assertEqual(res2.get("answerability_status"), STATUS_ANSWERABLE)
        self.assertEqual(str(res2.get("article_number")), "21")
        self.assertIn("21", res2.get("answer", ""))
        # Sources must be Article 21
        sources = res2.get("sources", [])
        self.assertTrue(len(sources) > 0)
        art_found = str(sources[0].get("article") or sources[0].get("article_number"))
        self.assertEqual(art_found, "21")

    def test_06_unrelated_query_remains_domain_restricted(self):
        """Test 6: User asks 'What is the capital of France?' after Article 14.
        Must trigger domain restriction and strictly have zero citations.
        """
        thread_id = f"test_thread_06_{uuid.uuid4().hex[:8]}"

        run_constitution_chat(
            "Explain Article 14 in simple English",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        res2 = run_constitution_chat(
            "What is the capital of France?",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        self.assertEqual(res2.get("answerability_status"), STATUS_OUT_OF_DOMAIN)
        self.assertEqual(res2.get("response_type"), "out_of_scope")
        self.assertEqual(len(res2.get("sources", [])), 0, "Out of domain query must NOT leak citations")
        self.assertIn("Domain Restriction", res2.get("answer", ""))

    def test_07_fresh_thread_isolated_fragment_asks_clarification(self):
        """Test 7: Fresh thread (no history) with 'in detail'.
        Must return clarification prompt asking which article or topic, without crashing or citing Article 21.
        """
        thread_id = f"test_thread_07_{uuid.uuid4().hex[:8]}"

        res = run_constitution_chat(
            "in detail",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        self.assertIn(
            res.get("answerability_status"),
            [STATUS_INSUFFICIENT_EVIDENCE, "INSUFFICIENT_EVIDENCE"],
        )
        self.assertEqual(len(res.get("sources", [])), 0, "Clarification must NOT have citations")
        self.assertTrue(
            "specify which constitutional article" in res.get("answer", "").lower()
            or "specify an article" in res.get("answer", "").lower()
            or "specify" in res.get("answer", "").lower()
        )
        self.assertNotIn("Article 21", res.get("answer", ""))

    def test_08_multi_session_history_isolation(self):
        """Test 8: Two independent user sessions with different thread IDs.
        Session A asks Article 14, Session B asks Article 19.
        Both ask 'in detail' -> Session A gets Article 14, Session B gets Article 19.
        """
        thread_a = f"test_session_a_{uuid.uuid4().hex[:8]}"
        thread_b = f"test_session_b_{uuid.uuid4().hex[:8]}"

        run_constitution_chat("Explain Article 14 in simple English", thread_a, checkpointer=self.checkpointer)
        run_constitution_chat("Explain Article 19 in simple English", thread_b, checkpointer=self.checkpointer)

        res_a = run_constitution_chat("in detail", thread_a, checkpointer=self.checkpointer)
        res_b = run_constitution_chat("in detail", thread_b, checkpointer=self.checkpointer)

        self.assertEqual(str(res_a.get("article_number")), "14", "Thread A must resolve to Article 14")
        self.assertEqual(str(res_b.get("article_number")), "19", "Thread B must resolve to Article 19")
        self.assertIn("14", res_a.get("answer", ""))
        self.assertIn("19", res_b.get("answer", ""))

    def test_09_dual_pipeline_advanced_and_normal_rag_both_receive_resolved_query(self):
        """Test 9: Advanced RAG and Normal RAG both execute with resolved query or history."""
        history = [
            HumanMessage(content="Explain Article 14 in simple English"),
            AIMessage(content="Article 14 provides equality before the law.", additional_kwargs={"article_number": "14"}),
        ]

        # Context resolver resolves "in detail"
        cr = resolve_follow_up_query("in detail", history)
        self.assertTrue(cr.is_follow_up)
        self.assertIn("14", cr.resolved_query)

        # Normal RAG with resolved query
        norm_rag = NormalRAGPipeline()
        norm_out = norm_rag.execute(cr.resolved_query)
        self.assertFalse(norm_out.is_abstention)
        self.assertIn("14", norm_out.answer)

        # Advanced RAG with conversation_history
        adv_rag = AdvancedRAGPipeline()
        adv_out = adv_rag.execute("in detail", conversation_history=history)
        self.assertFalse(adv_out.is_abstention)
        self.assertIn("14", adv_out.answer)
        self.assertTrue(len(adv_out.citations) > 0)
        self.assertEqual(str(adv_out.citations[0]["article"]), "14")

    def test_10_streaming_and_multiturn_state_persistence(self):
        """Test 10: Streaming and multi-turn execution preserve context cleanly without duplicate messages."""
        thread_id = f"test_thread_10_{uuid.uuid4().hex[:8]}"

        # Turn 1 via stream
        events_t1 = list(run_constitution_chat_stream(
            "Explain Article 14 in simple English",
            thread_id,
            checkpointer=self.checkpointer,
        ))
        complete_t1 = [e for e in events_t1 if e.get("type") == "complete"]
        self.assertEqual(len(complete_t1), 1)

        # Turn 2 via stream
        events_t2 = list(run_constitution_chat_stream(
            "in detail",
            thread_id,
            checkpointer=self.checkpointer,
        ))
        complete_t2 = [e for e in events_t2 if e.get("type") == "complete"]
        self.assertEqual(len(complete_t2), 1)

        res2 = complete_t2[0]["result"]
        self.assertEqual(str(res2.get("article_number")), "14")
        self.assertEqual(res2.get("answerability_status"), STATUS_ANSWERABLE)

        # Verify message count: exactly 2 human and 2 AI messages (no duplicates)
        messages = res2.get("messages", [])
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
        self.assertEqual(len(human_msgs), 2)
        self.assertEqual(len(ai_msgs), 2)

    def test_11_verbatim_pdf_mode_preserves_exact_source_without_paraphrase(self):
        """Test 11: Verbatim PDF mode with 'explain in detail' preserves full exact-source fidelity."""
        thread_id = f"test_thread_11_{uuid.uuid4().hex[:8]}"

        run_constitution_chat(
            "Explain Article 14 in simple English",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="rule_based",
        )

        res2 = run_constitution_chat(
            "explain in detail",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="rule_based",
        )

        self.assertEqual(res2.get("answerability_status"), STATUS_ANSWERABLE)
        self.assertEqual(str(res2.get("article_number")), "14")
        ans = res2.get("answer", "")
        # Verbatim text check: Article 14 text
        self.assertIn("The State shall not deny to any person equality before the law", ans)
        self.assertIn("Full clause fidelity without truncation", ans)

    def test_12_followup_absent_fact_triggers_insufficient_evidence(self):
        """Test 12: Follow-up asking an unrecorded temporal fact triggers insufficient evidence."""
        thread_id = f"test_thread_12_{uuid.uuid4().hex[:8]}"

        run_constitution_chat(
            "Explain Article 14 in simple English",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        res2 = run_constitution_chat(
            "How many amendments were passed yesterday to Article 14?",
            thread_id,
            checkpointer=self.checkpointer,
            selected_engine="both",
        )

        self.assertEqual(res2.get("answerability_status"), STATUS_INSUFFICIENT_EVIDENCE)
        self.assertEqual(len(res2.get("sources", [])), 0, "Abstentions must NOT leak citations")
        self.assertIn("cannot determine", res2.get("answer", "").lower())


if __name__ == "__main__":
    unittest.main()
