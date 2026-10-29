"""Unit tests for Standard Information Retrieval (IR) metrics.

Verifies hand-checkable mathematical cases for:
- Recall@K
- Precision@K
- R-Precision
- MRR
- Hit Rate@K
- NDCG@K (Binary and Graded 0-3 scale)
- Duplicate handling
- Empty relevance sets (|Relevant| == 0)
"""

import math
import unittest

from app.retrieval.metrics import (
    compute_accuracy,
    compute_binary_precision,
    compute_binary_recall,
    compute_confusion_matrix,
    compute_dcg_at_k,
    compute_f1_score,
    compute_hit_rate_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_r_precision,
    compute_recall_at_k,
    compute_reciprocal_rank,
    deduplicate_preserving_order,
)


class TestRetrievalMetrics(unittest.TestCase):
    def test_deduplication_preserving_order(self):
        """Verify duplicate chunk IDs are collapsed while preserving 1st occurrence rank."""
        raw = ["chunk_1", "chunk_2", "chunk_1", "chunk_3", "chunk_2", ""]
        deduped = deduplicate_preserving_order(raw)
        self.assertEqual(deduped, ["chunk_1", "chunk_2", "chunk_3"])

    def test_single_relevant_at_rank_1(self):
        """Case 1: 1 relevant chunk, retrieved at rank 1."""
        retrieved = ["doc_A", "doc_B", "doc_C"]
        relevant = ["doc_A"]

        # Recall
        self.assertEqual(compute_recall_at_k(retrieved, relevant, 1), 1.0)
        self.assertEqual(compute_recall_at_k(retrieved, relevant, 3), 1.0)

        # Precision
        self.assertEqual(compute_precision_at_k(retrieved, relevant, 1), 1.0)
        self.assertAlmostEqual(compute_precision_at_k(retrieved, relevant, 3), 1.0 / 3.0)
        self.assertAlmostEqual(compute_precision_at_k(retrieved, relevant, 5), 1.0 / 5.0)

        # R-Precision (R = 1)
        self.assertEqual(compute_r_precision(retrieved, relevant), 1.0)

        # MRR & Hit Rate
        self.assertEqual(compute_reciprocal_rank(retrieved, relevant), 1.0)
        self.assertEqual(compute_hit_rate_at_k(retrieved, relevant, 1), 1.0)

        # NDCG
        self.assertEqual(compute_ndcg_at_k(retrieved, {}, 1, relevant_chunk_ids=relevant), 1.0)

    def test_single_relevant_at_rank_2(self):
        """Case 2: 1 relevant chunk, retrieved at rank 2."""
        retrieved = ["doc_B", "doc_A", "doc_C"]
        relevant = ["doc_A"]

        # Recall
        self.assertEqual(compute_recall_at_k(retrieved, relevant, 1), 0.0)
        self.assertEqual(compute_recall_at_k(retrieved, relevant, 2), 1.0)

        # Precision
        self.assertEqual(compute_precision_at_k(retrieved, relevant, 1), 0.0)
        self.assertEqual(compute_precision_at_k(retrieved, relevant, 2), 0.5)

        # R-Precision (R = 1, top 1 is doc_B)
        self.assertEqual(compute_r_precision(retrieved, relevant), 0.0)

        # MRR (rank 2 -> 1/2)
        self.assertEqual(compute_reciprocal_rank(retrieved, relevant), 0.5)

        # Hit Rate
        self.assertEqual(compute_hit_rate_at_k(retrieved, relevant, 1), 0.0)
        self.assertEqual(compute_hit_rate_at_k(retrieved, relevant, 2), 1.0)

        # NDCG@2: DCG = 1 / log2(3); IDCG = 1 / log2(2) = 1.0; NDCG = 1 / log2(3)
        expected_ndcg = (1.0 / math.log2(3.0)) / (1.0 / math.log2(2.0))
        self.assertAlmostEqual(
            compute_ndcg_at_k(retrieved, {}, 2, relevant_chunk_ids=relevant),
            expected_ndcg,
            places=4,
        )

    def test_multi_relevant_synthesis(self):
        """Case 3: 3 relevant chunks, 2 retrieved in top 5."""
        retrieved = ["doc_A", "doc_X", "doc_B", "doc_Y", "doc_Z"]
        relevant = ["doc_A", "doc_B", "doc_C"]

        self.assertAlmostEqual(compute_recall_at_k(retrieved, relevant, 1), 1.0 / 3.0)
        self.assertAlmostEqual(compute_recall_at_k(retrieved, relevant, 3), 2.0 / 3.0)
        self.assertAlmostEqual(compute_recall_at_k(retrieved, relevant, 5), 2.0 / 3.0)

        # Standard Precision (denominator strictly K)
        self.assertEqual(compute_precision_at_k(retrieved, relevant, 1), 1.0)
        self.assertAlmostEqual(compute_precision_at_k(retrieved, relevant, 3), 2.0 / 3.0)
        self.assertAlmostEqual(compute_precision_at_k(retrieved, relevant, 5), 2.0 / 5.0)

        # R-Precision (R = 3 -> top 3 are doc_A, doc_X, doc_B -> 2 hits -> 2/3)
        self.assertAlmostEqual(compute_r_precision(retrieved, relevant), 2.0 / 3.0)

    def test_duplicate_retrieved_chunks_not_overcounted(self):
        """Case 4: Duplicates in retrieved list must be deduplicated."""
        retrieved = ["doc_A", "doc_A", "doc_A", "doc_B"]
        relevant = ["doc_A", "doc_B"]

        # Deduplication produces ["doc_A", "doc_B"]
        # In top-2, both doc_A and doc_B are present
        self.assertEqual(compute_recall_at_k(retrieved, relevant, 2), 1.0)
        self.assertEqual(compute_precision_at_k(retrieved, relevant, 2), 1.0)

    def test_empty_relevance_set_handling(self):
        """Case 5: Unanswerable query with 0 relevant chunks must return None for Recall."""
        retrieved = ["doc_A", "doc_B"]
        relevant = []

        self.assertIsNone(compute_recall_at_k(retrieved, relevant, 5))
        self.assertIsNone(compute_r_precision(retrieved, relevant))
        self.assertIsNone(compute_reciprocal_rank(retrieved, relevant))
        self.assertIsNone(compute_hit_rate_at_k(retrieved, relevant, 5))
        self.assertIsNone(compute_ndcg_at_k(retrieved, {}, 5, relevant_chunk_ids=relevant))

        # Precision@K with 0 relevant chunks is 0.0
        self.assertEqual(compute_precision_at_k(retrieved, relevant, 5), 0.0)

    def test_graded_ndcg_calculation(self):
        """Case 6: Graded 0-3 relevance NDCG calculation."""
        retrieved = ["doc_A", "doc_B", "doc_C"]
        judgments = {"doc_A": 3, "doc_B": 1, "doc_C": 0, "doc_D": 2}

        # Actual:
        # doc_A (rel=3): (2^3 - 1) / log2(2) = 7.0 / 1.0 = 7.0
        # doc_B (rel=1): (2^1 - 1) / log2(3) = 1.0 / 1.58496 = 0.63093
        # doc_C (rel=0): 0.0
        actual_dcg = 7.0 + 0.63092975

        # Ideal (sorted: 3, 2, 1):
        # rank 1: (2^3 - 1) / log2(2) = 7.0
        # rank 2: (2^2 - 1) / log2(3) = 3.0 / 1.5849625 = 1.892789
        # rank 3: (2^1 - 1) / log2(4) = 1.0 / 2.0 = 0.5
        ideal_dcg = 7.0 + (3.0 / math.log2(3.0)) + 0.5
        expected_ndcg = actual_dcg / ideal_dcg

        ndcg_val = compute_ndcg_at_k(retrieved, judgments, 3)
        self.assertAlmostEqual(ndcg_val, expected_ndcg, places=4)

    def test_confusion_matrix_and_f1_accuracy(self):
        """Case 7: Hand-calculated verification of binary classification metrics."""
        # 10 examples: 6 positives, 4 negatives
        ground_truth = [True,  True,  True,  True,  True,  True,  False, False, False, False]
        predictions  = [True,  True,  True,  True,  False, False, False, False, False, True]

        cm = compute_confusion_matrix(predictions, ground_truth)
        # TP = 4 (first 4 true positives)
        # FN = 2 (indices 4, 5 were true but predicted false)
        # TN = 3 (indices 6, 7, 8 were false and predicted false)
        # FP = 1 (index 9 was false but predicted true)
        self.assertEqual(cm["tp"], 4)
        self.assertEqual(cm["tn"], 3)
        self.assertEqual(cm["fp"], 1)
        self.assertEqual(cm["fn"], 2)
        self.assertEqual(cm["total"], 10)

        # Precision = 4 / (4 + 1) = 0.8
        self.assertAlmostEqual(compute_binary_precision(4, 1), 0.8)

        # Recall = 4 / (4 + 2) = 4/6 = 0.66667
        self.assertAlmostEqual(compute_binary_recall(4, 2), 4.0 / 6.0)

        # F1 = 2 * (0.8 * 0.66667) / (0.8 + 0.66667) = 2*4 / (2*4 + 1 + 2) = 8 / 11 = 0.72727
        expected_f1 = 8.0 / 11.0
        self.assertAlmostEqual(compute_f1_score(4, 1, 2), expected_f1, places=5)

        # Accuracy = (4 + 3) / 10 = 0.7 (70.0%)
        self.assertAlmostEqual(compute_accuracy(4, 3, 1, 2), 0.7)

    def test_zero_denominator_edge_cases(self):
        """Case 8: Zero denominator safety in F1 and Accuracy."""
        self.assertEqual(compute_f1_score(0, 0, 0), 0.0)
        self.assertEqual(compute_accuracy(0, 0, 0, 0), 0.0)
        self.assertEqual(compute_binary_precision(0, 0), 0.0)
        self.assertEqual(compute_binary_recall(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
