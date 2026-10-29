import unittest
from app.adversarial.query_classifier import (
    AdversarialQueryClassifier,
    QUERY_TYPE_FALSE_PREMISE,
    QUERY_TYPE_AMBIGUOUS,
    QUERY_TYPE_TEMPORAL_DYNAMIC,
    QUERY_TYPE_MULTI_HOP,
)
from app.adversarial.claim_verifier import ClaimAndCitationVerifier
from app.retrieval.reranker import LegalReranker
from app.retrieval.knowledge_base import ConstitutionalKnowledgeBase
from app.graph.state import ConstitutionState
from app.graph.nodes import rewrite_query
from app.graph.routing import route_classification

class TestAdversarialHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = AdversarialQueryClassifier()
        cls.verifier = ClaimAndCitationVerifier()
        cls.reranker = LegalReranker()
        cls.kb = ConstitutionalKnowledgeBase()

    def test_false_premise_classification(self):
        """Verify queries with false constitutional premises are detected and refuted."""
        query = "Explain how Article 14 grants titles of nobility to citizens."
        analysis = self.classifier.analyze(query)
        self.assertTrue(analysis.is_false_premise)
        self.assertEqual(analysis.query_type, QUERY_TYPE_FALSE_PREMISE)
        self.assertIn("18", analysis.refutation)
        self.assertIn("14", analysis.refutation)

        query2 = "What procedure does Article 21 specify for the President to grant pardons?"
        analysis2 = self.classifier.analyze(query2)
        self.assertTrue(analysis2.is_false_premise)
        self.assertIn("72", analysis2.refutation)

    def test_ambiguous_query_classification(self):
        """Verify ambiguous and underspecified queries trigger clarification needs."""
        query = "who can issue ordinances?"
        analysis = self.classifier.analyze(query)
        self.assertTrue(analysis.is_ambiguous)
        self.assertEqual(analysis.query_type, QUERY_TYPE_AMBIGUOUS)
        self.assertIsNotNone(analysis.clarification_prompt)
        self.assertTrue("clarify" in analysis.clarification_prompt.lower() or "specify" in analysis.clarification_prompt.lower())

    def test_temporal_out_of_scope_classification(self):
        """Verify temporal inquiries not in the static constitution are flagged."""
        query = "What amendments were passed by parliament yesterday?"
        analysis = self.classifier.analyze(query)
        self.assertEqual(analysis.query_type, QUERY_TYPE_TEMPORAL_DYNAMIC)
        self.assertFalse(analysis.is_answerable_from_text)

    def test_multi_hop_query_classification(self):
        """Verify multi-hop queries identify multiple article targets."""
        query = "Compare the writ jurisdiction of the Supreme Court under Article 32 with the High Courts under Article 226."
        analysis = self.classifier.analyze(query)
        self.assertEqual(analysis.query_type, QUERY_TYPE_MULTI_HOP)
        self.assertIn("32", analysis.target_articles)
        self.assertIn("226", analysis.target_articles)

    def test_graph_node_false_premise_routing(self):
        """Verify LangGraph node cleanly routes false premise queries directly to refutation without hallucination."""
        state: ConstitutionState = {
            "question": "How does Article 14 grant titles of nobility to citizens?",
            "messages": [],
        }
        res_state = rewrite_query(state)
        # Merged state should carry false premise intent or refutation answer
        merged = {**state, **res_state}
        self.assertEqual(merged.get("intent"), "false_premise_refutation")
        self.assertIn("Article 18", merged.get("answer", ""))

        route = route_classification(merged)
        self.assertEqual(route, "finalize_response")

    def test_claim_and_citation_verifier(self):
        """Verify quotation fidelity and citation ground checking."""
        response_text = 'As stated in Article 14 of the Constitution: "The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India."'
        sources = [
            {"article_number": "14", "text": "The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India."}
        ]
        audit = self.verifier.verify(response_text, sources)
        self.assertTrue(audit.is_grounded)
        self.assertGreaterEqual(audit.citation_precision, 0.8)
        self.assertEqual(len(audit.unsupported_claims), 0)

    def test_claim_verifier_detects_fabricated_citation(self):
        """Verify fabricated articles (e.g. Article 999) are flagged as ungrounded."""
        response_text = "Article 999 establishes an AI regulation commission."
        audit = self.verifier.verify(response_text, [])
        self.assertFalse(audit.is_grounded)
        self.assertIn("999", audit.fabricated_articles)

    def test_reranker_false_premise_penalty(self):
        """Verify reranker penalizes chunks when a query contains a known false premise."""
        docs = [
            {"article_number": "14", "article_title": "Equality before law", "constitutional_text": "The State shall not deny...", "score": 0.8},
            {"article_number": "18", "article_title": "Abolition of titles", "constitutional_text": "No title, not being a military or academic distinction...", "score": 0.5}
        ]
        reranked = self.reranker.rerank("How does Article 14 grant titles of nobility?", docs, top_k=2)
        # Article 18 should be boosted or Article 14 penalized for false title attribution
        top_art = reranked[0].get("article_number")
        self.assertEqual(top_art, "18")

if __name__ == "__main__":
    unittest.main()
