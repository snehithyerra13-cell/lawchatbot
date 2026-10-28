"""Golden Reranking Evaluation Dataset Builder and Validator.

Creates and validates Dataset B: tests/evaluation/golden_reranking_dataset.jsonl
used strictly for offline ranking and reranking evaluation.

Schema per record:
- query_id: Stable unique identifier (e.g. 'gold_001')
- query: User legal question
- relevant_chunk_ids: List of verified relevant chunk IDs from Dataset A (grades >= 2)
- relevance_judgments: Dict mapping candidate chunk IDs to graded relevance labels:
    0: Not relevant
    1: Related / background, but does not adequately answer the query
    2: Relevant and provides useful supporting information
    3: Highly relevant and directly answers the query
- expected_article_numbers: List of canonical article numbers (e.g. ['14'])
- expected_source_ids: List of source identifiers (e.g. ['const_art_14'])
- query_category: exact_article_lookup | paraphrase | conceptual_query | multi_article_query | hard_negative
- annotation_status: 'manually_verified'
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)

GOLDEN_EVALUATION_RECORDS: List[Dict[str, Any]] = [
    # -------------------------------------------------------------
    # Category 1: Exact Article Lookup (5 queries)
    # -------------------------------------------------------------
    {
        "query_id": "gold_001",
        "query": "What is Article 14 of the Constitution of India?",
        "relevant_chunk_ids": ["const_14_p37_c01"],
        "relevance_judgments": {
            "const_14_p37_c01": 3,
            "const_15_p342_c01": 1,
            "const_21_p42_c01": 0,
            "const_preamble_p32_c01": 0,
        },
        "expected_article_numbers": ["14"],
        "expected_source_ids": ["const_art_14"],
        "query_category": "exact_article_lookup",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_002",
        "query": "Explain Article 21 protection of life and personal liberty",
        "relevant_chunk_ids": ["const_21_p42_c01"],
        "relevance_judgments": {
            "const_21_p42_c01": 3,
            "const_22_p42_c01": 1,
            "const_20_p42_c01": 1,
            "const_14_p37_c01": 0,
        },
        "expected_article_numbers": ["21"],
        "expected_source_ids": ["const_art_21"],
        "query_category": "exact_article_lookup",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_003",
        "query": "What does Article 32 say about constitutional remedies and writ jurisdiction?",
        "relevant_chunk_ids": ["const_32_p50_c01", "const_32_p50_c02"],
        "relevance_judgments": {
            "const_32_p50_c01": 3,
            "const_32_p50_c02": 2,
            "const_226_p135_c01": 1,
            "const_14_p37_c01": 0,
        },
        "expected_article_numbers": ["32"],
        "expected_source_ids": ["const_art_32"],
        "query_category": "exact_article_lookup",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_004",
        "query": "What are the Fundamental Duties under Article 51A?",
        "relevant_chunk_ids": ["const_51a_p56_c01"],
        "relevance_judgments": {
            "const_51a_p56_c01": 3,
            "const_preamble_p32_c01": 1,
            "const_21a_p42_c01": 1,
            "const_14_p37_c01": 0,
        },
        "expected_article_numbers": ["51A"],
        "expected_source_ids": ["const_art_51a"],
        "query_category": "exact_article_lookup",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_005",
        "query": "What does the Preamble to the Constitution of India declare?",
        "relevant_chunk_ids": ["const_preamble_p32_c01"],
        "relevance_judgments": {
            "const_preamble_p32_c01": 3,
            "const_1_p33_c01": 1,
            "const_14_p37_c01": 0,
            "const_51a_p56_c01": 0,
        },
        "expected_article_numbers": ["PREAMBLE"],
        "expected_source_ids": ["const_art_preamble"],
        "query_category": "exact_article_lookup",
        "annotation_status": "manually_verified",
    },

    # -------------------------------------------------------------
    # Category 2: Paraphrase Queries (5 queries)
    # -------------------------------------------------------------
    {
        "query_id": "gold_006",
        "query": "Can the government arbitrarily discriminate against any citizen without reasonable basis?",
        "relevant_chunk_ids": ["const_14_p37_c01", "const_15_p342_c01"],
        "relevance_judgments": {
            "const_14_p37_c01": 3,
            "const_15_p342_c01": 2,
            "const_16_p344_c01": 1,
            "const_19_p40_c01": 0,
        },
        "expected_article_numbers": ["14", "15"],
        "expected_source_ids": ["const_art_14", "const_art_15"],
        "query_category": "paraphrase",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_007",
        "query": "Can a person be deprived of their liberty without following fair procedure established by law?",
        "relevant_chunk_ids": ["const_21_p42_c01"],
        "relevance_judgments": {
            "const_21_p42_c01": 3,
            "const_22_p42_c01": 2,
            "const_20_p42_c01": 1,
            "const_14_p37_c01": 1,
        },
        "expected_article_numbers": ["21"],
        "expected_source_ids": ["const_art_21"],
        "query_category": "paraphrase",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_008",
        "query": "How can citizens directly move the Supreme Court when fundamental rights are violated?",
        "relevant_chunk_ids": ["const_32_p50_c01"],
        "relevance_judgments": {
            "const_32_p50_c01": 3,
            "const_32_p50_c02": 2,
            "const_226_p135_c01": 1,
            "const_124_p88_c01": 0,
        },
        "expected_article_numbers": ["32"],
        "expected_source_ids": ["const_art_32"],
        "query_category": "paraphrase",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_009",
        "query": "Is the practice of untouchability strictly forbidden and punishable by law in India?",
        "relevant_chunk_ids": ["const_17_p40_c01"],
        "relevance_judgments": {
            "const_17_p40_c01": 3,
            "const_15_p342_c01": 1,
            "const_14_p37_c01": 1,
            "const_23_p45_c01": 0,
        },
        "expected_article_numbers": ["17"],
        "expected_source_ids": ["const_art_17"],
        "query_category": "paraphrase",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_010",
        "query": "Can children below the age of fourteen years be employed in hazardous factories or mines?",
        "relevant_chunk_ids": ["const_24_p45_c01"],
        "relevance_judgments": {
            "const_24_p45_c01": 3,
            "const_23_p45_c01": 1,
            "const_21a_p42_c01": 1,
            "const_14_p37_c01": 0,
        },
        "expected_article_numbers": ["24"],
        "expected_source_ids": ["const_art_24"],
        "query_category": "paraphrase",
        "annotation_status": "manually_verified",
    },

    # -------------------------------------------------------------
    # Category 3: Conceptual Queries (5 queries)
    # -------------------------------------------------------------
    {
        "query_id": "gold_011",
        "query": "Protection against double jeopardy and compelled self-incrimination in criminal trials",
        "relevant_chunk_ids": ["const_20_p42_c01"],
        "relevance_judgments": {
            "const_20_p42_c01": 3,
            "const_21_p42_c01": 1,
            "const_22_p42_c01": 1,
            "const_14_p37_c01": 0,
        },
        "expected_article_numbers": ["20"],
        "expected_source_ids": ["const_art_20"],
        "query_category": "conceptual_query",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_012",
        "query": "Right of children to free and compulsory primary education between ages 6 and 14",
        "relevant_chunk_ids": ["const_21a_p42_c01"],
        "relevance_judgments": {
            "const_21a_p42_c01": 3,
            "const_51a_p56_c01": 1,
            "const_21_p42_c01": 1,
            "const_24_p45_c01": 0,
        },
        "expected_article_numbers": ["21A"],
        "expected_source_ids": ["const_art_21a"],
        "query_category": "conceptual_query",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_013",
        "query": "Freedom of conscience and the right freely to profess, practise and propagate religion",
        "relevant_chunk_ids": ["const_25_p349_c01"],
        "relevance_judgments": {
            "const_25_p349_c01": 3,
            "const_preamble_p32_c01": 1,
            "const_19_p40_c01": 0,
            "const_14_p37_c01": 0,
        },
        "expected_article_numbers": ["25"],
        "expected_source_ids": ["const_art_25"],
        "query_category": "conceptual_query",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_014",
        "query": "Separation of judiciary from the executive branch in public administration",
        "relevant_chunk_ids": ["const_50_p55_c01"],
        "relevance_judgments": {
            "const_50_p55_c01": 3,
            "const_124_p88_c01": 1,
            "const_32_p50_c01": 0,
            "const_14_p37_c01": 0,
        },
        "expected_article_numbers": ["50"],
        "expected_source_ids": ["const_art_50"],
        "query_category": "conceptual_query",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_015",
        "query": "Power of the President of India to grant pardons, reprieves, respites or remissions of punishment",
        "relevant_chunk_ids": ["const_72_p362_c01"],
        "relevance_judgments": {
            "const_72_p362_c01": 3,
            "const_21_p42_c01": 1,
            "const_32_p50_c01": 0,
            "const_14_p37_c01": 0,
        },
        "expected_article_numbers": ["72"],
        "expected_source_ids": ["const_art_72"],
        "query_category": "conceptual_query",
        "annotation_status": "manually_verified",
    },

    # -------------------------------------------------------------
    # Category 4: Multi-Article Queries (5 queries)
    # -------------------------------------------------------------
    {
        "query_id": "gold_016",
        "query": "What constitutes the Golden Triangle doctrine of Fundamental Rights in the Indian Constitution?",
        "relevant_chunk_ids": ["const_14_p37_c01", "const_19_p40_c01", "const_21_p42_c01"],
        "relevance_judgments": {
            "const_14_p37_c01": 3,
            "const_19_p40_c01": 3,
            "const_21_p42_c01": 3,
            "const_32_p50_c01": 1,
            "const_preamble_p32_c01": 0,
        },
        "expected_article_numbers": ["14", "19", "21"],
        "expected_source_ids": ["const_art_14", "const_art_19", "const_art_21"],
        "query_category": "multi_article_query",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_017",
        "query": "Compare general equality before law under Article 14 with prohibition of discrimination under Article 15",
        "relevant_chunk_ids": ["const_14_p37_c01", "const_15_p342_c01"],
        "relevance_judgments": {
            "const_14_p37_c01": 3,
            "const_15_p342_c01": 3,
            "const_16_p344_c01": 1,
            "const_21_p42_c01": 0,
        },
        "expected_article_numbers": ["14", "15"],
        "expected_source_ids": ["const_art_14", "const_art_15"],
        "query_category": "multi_article_query",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_018",
        "query": "Compare National Emergency under Article 352 with failure of constitutional machinery in States under Article 356",
        "relevant_chunk_ids": ["const_352_p239_c01", "const_356_p35_c01"],
        "relevance_judgments": {
            "const_352_p239_c01": 3,
            "const_356_p35_c01": 3,
            "const_352_p239_c02": 2,
            "const_368_p35_c01": 0,
        },
        "expected_article_numbers": ["352", "356"],
        "expected_source_ids": ["const_art_352", "const_art_356"],
        "query_category": "multi_article_query",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_019",
        "query": "Provisions protecting citizens against human trafficking, begar, and hazardous child employment",
        "relevant_chunk_ids": ["const_23_p45_c01", "const_24_p45_c01"],
        "relevance_judgments": {
            "const_23_p45_c01": 3,
            "const_24_p45_c01": 3,
            "const_21_p42_c01": 1,
            "const_14_p37_c01": 0,
        },
        "expected_article_numbers": ["23", "24"],
        "expected_source_ids": ["const_art_23", "const_art_24"],
        "query_category": "multi_article_query",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_020",
        "query": "What is the procedure and power of Parliament to amend the Constitution of India under Article 368?",
        "relevant_chunk_ids": ["const_368_p35_c01"],
        "relevance_judgments": {
            "const_368_p35_c01": 3,
            "const_preamble_p32_c01": 1,
            "const_14_p37_c01": 0,
            "const_356_p35_c01": 0,
        },
        "expected_article_numbers": ["368"],
        "expected_source_ids": ["const_art_368"],
        "query_category": "multi_article_query",
        "annotation_status": "manually_verified",
    },

    # -------------------------------------------------------------
    # Category 5: Hard Negatives (5 queries)
    # -------------------------------------------------------------
    {
        "query_id": "gold_021",
        "query": "Can the Supreme Court issue writs against High Courts under Article 32 as opposed to Article 226 powers?",
        "relevant_chunk_ids": ["const_32_p50_c01"],
        "relevance_judgments": {
            "const_32_p50_c01": 3,
            "const_32_p50_c02": 2,
            "const_226_p135_c01": 1,  # Hard negative / related distractor
            "const_124_p88_c01": 0,   # Distractor
            "const_14_p37_c01": 0,    # Distractor
        },
        "expected_article_numbers": ["32"],
        "expected_source_ids": ["const_art_32"],
        "query_category": "hard_negative",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_022",
        "query": "Does freedom of speech under Article 19 include absolute right to provoke violence without reasonable restrictions?",
        "relevant_chunk_ids": ["const_19_p40_c01"],
        "relevance_judgments": {
            "const_19_p40_c01": 3,
            "const_21_p42_c01": 1,   # Related distractor
            "const_14_p37_c01": 0,   # Distractor
            "const_352_p239_c01": 0, # Distractor
        },
        "expected_article_numbers": ["19"],
        "expected_source_ids": ["const_art_19"],
        "query_category": "hard_negative",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_023",
        "query": "Does imposing President's Rule under Article 356 automatically extinguish personal liberty under Article 21?",
        "relevant_chunk_ids": ["const_356_p35_c01", "const_21_p42_c01"],
        "relevance_judgments": {
            "const_356_p35_c01": 3,
            "const_21_p42_c01": 2,
            "const_352_p239_c01": 1, # Hard negative / related emergency
            "const_14_p37_c01": 0,
        },
        "expected_article_numbers": ["356", "21"],
        "expected_source_ids": ["const_art_356", "const_art_21"],
        "query_category": "hard_negative",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_024",
        "query": "Directive principle regarding Uniform Civil Code for all citizens across India",
        "relevant_chunk_ids": ["const_44_p54_c01"],
        "relevance_judgments": {
            "const_44_p54_c01": 3,
            "const_25_p349_c01": 1,  # Hard negative (freedom of religion distractor)
            "const_14_p37_c01": 1,   # Related equality distractor
            "const_50_p55_c01": 0,
        },
        "expected_article_numbers": ["44"],
        "expected_source_ids": ["const_art_44"],
        "query_category": "hard_negative",
        "annotation_status": "manually_verified",
    },
    {
        "query_id": "gold_025",
        "query": "Does Article 1 allow individual Indian States to secede or declare sovereignty away from Bharat?",
        "relevant_chunk_ids": ["const_1_p33_c01"],
        "relevance_judgments": {
            "const_1_p33_c01": 3,
            "const_preamble_p32_c01": 1, # Hard negative distractor
            "const_14_p37_c01": 0,
            "const_51a_p56_c01": 0,
        },
        "expected_article_numbers": ["1"],
        "expected_source_ids": ["const_art_1"],
        "query_category": "hard_negative",
        "annotation_status": "manually_verified",
    },
]


def build_golden_evaluation_dataset(
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Write the verified golden evaluation dataset to disk as UTF-8 JSON Lines,
    including train/val/test splits, answer gold, and normal evaluation queries.
    """
    settings = get_settings()
    settings.ensure_directories()
    out_path = Path(output_path or settings.golden_evaluation_dataset_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Primary golden evaluation file (for test_rag_evaluation compatibility)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in GOLDEN_EVALUATION_RECORDS:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 2. Mirror to data/evaluation/golden/retrieval_gold.jsonl
    retrieval_gold_path = settings.golden_eval_dir / "retrieval_gold.jsonl"
    with open(retrieval_gold_path, "w", encoding="utf-8") as f:
        for rec in GOLDEN_EVALUATION_RECORDS:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3. Deterministic Splits (by Query Families to prevent data leakage)
    train_ids = {"gold_001", "gold_002", "gold_004", "gold_006", "gold_007", "gold_009", "gold_011", "gold_012", "gold_014", "gold_017", "gold_019", "gold_022"}
    val_ids = {"gold_005", "gold_010", "gold_015", "gold_018", "gold_024"}
    test_ids = {"gold_003", "gold_008", "gold_013", "gold_016", "gold_020", "gold_021", "gold_023", "gold_025"}

    train_recs = [r for r in GOLDEN_EVALUATION_RECORDS if r["query_id"] in train_ids]
    val_recs = [r for r in GOLDEN_EVALUATION_RECORDS if r["query_id"] in val_ids]
    test_recs = [r for r in GOLDEN_EVALUATION_RECORDS if r["query_id"] in test_ids]

    train_path = settings.golden_eval_dir / "reranker_train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for rec in train_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    val_path = settings.golden_eval_dir / "reranker_validation.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for rec in val_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    test_path = settings.golden_eval_dir / "reranker_test.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for rec in test_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 4. Answer Gold dataset with citations and unanswerable queries
    answer_gold_records = []
    for r in GOLDEN_EVALUATION_RECORDS:
        answer_gold_records.append({
            "query_id": r["query_id"],
            "query": r["query"],
            "is_answerable": True,
            "expected_abstention": False,
            "expected_articles": r["expected_article_numbers"],
            "expected_sources": r["expected_source_ids"],
            "relevant_chunk_ids": r["relevant_chunk_ids"],
            "category": r["query_category"],
        })

    # Add unanswerable / abstention test queries
    unanswerable_cases = [
        {
            "query_id": "gold_unans_001",
            "query": "What is the procedure for amending the United States Constitution under Article 5?",
            "is_answerable": False,
            "expected_abstention": True,
            "expected_articles": [],
            "expected_sources": [],
            "relevant_chunk_ids": [],
            "category": "unanswerable_out_of_domain",
        },
        {
            "query_id": "gold_unans_002",
            "query": "Who is the Chief Justice of India in 2026?",
            "is_answerable": False,
            "expected_abstention": True,
            "expected_articles": ["124"],
            "expected_sources": ["const_art_124"],
            "relevant_chunk_ids": [],
            "category": "unanswerable_current_affairs",
        },
        {
            "query_id": "gold_unans_003",
            "query": "What does Article 500 of the Constitution of India state regarding taxation?",
            "is_answerable": False,
            "expected_abstention": True,
            "expected_articles": [],
            "expected_sources": [],
            "relevant_chunk_ids": [],
            "category": "unanswerable_nonexistent_article",
        },
    ]
    answer_gold_records.extend(unanswerable_cases)

    answer_gold_path = settings.golden_eval_dir / "answer_gold.jsonl"
    with open(answer_gold_path, "w", encoding="utf-8") as f:
        for rec in answer_gold_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 5. Normal evaluation queries (stripped of relevance labels to avoid data leakage)
    normal_eval_queries = [
        {
            "query_id": r["query_id"],
            "query": r["query"],
            "query_category": r["query_category"],
        }
        for r in GOLDEN_EVALUATION_RECORDS
    ]
    normal_eval_path = settings.normal_eval_dir / "evaluation_queries.jsonl"
    with open(normal_eval_path, "w", encoding="utf-8") as f:
        for rec in normal_eval_queries:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Wrote golden evaluation suite: %d total, %d train, %d val, %d test, %d answers to %s",
        len(GOLDEN_EVALUATION_RECORDS),
        len(train_recs),
        len(val_recs),
        len(test_recs),
        len(answer_gold_records),
        settings.golden_eval_dir,
    )
    return GOLDEN_EVALUATION_RECORDS


def load_golden_evaluation_dataset(
    dataset_path: Optional[Path] = None, auto_build_if_missing: bool = True
) -> List[Dict[str, Any]]:
    """Load Dataset B from disk, building it if missing."""
    settings = get_settings()
    path = Path(dataset_path or settings.golden_evaluation_dataset_path)

    if not path.exists():
        if auto_build_if_missing:
            logger.info("Golden dataset not found at %s. Creating it...", path)
            return build_golden_evaluation_dataset(output_path=path)
        raise FileNotFoundError(f"Golden dataset not found at: {path}")

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                records.append(data)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_no} in {path}: {exc}") from exc

    return records


def validate_golden_evaluation_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate Dataset B integrity, schema, non-empty labels, and valid score ranges."""
    required_fields = {
        "query_id",
        "query",
        "relevant_chunk_ids",
        "relevance_judgments",
        "expected_article_numbers",
        "expected_source_ids",
        "query_category",
        "annotation_status",
    }
    allowed_categories = {
        "exact_article_lookup",
        "paraphrase",
        "conceptual_query",
        "multi_article_query",
        "hard_negative",
    }

    errors: List[str] = []
    query_ids = set()
    category_counts: Dict[str, int] = {}

    if not records:
        return {"valid": False, "errors": ["Golden evaluation dataset is empty"]}

    for idx, rec in enumerate(records):
        missing = required_fields - set(rec.keys())
        if missing:
            errors.append(f"Record {idx} missing required fields: {missing}")

        qid = rec.get("query_id", "")
        if not qid or not isinstance(qid, str):
            errors.append(f"Record {idx} invalid query_id: {qid}")
        elif qid in query_ids:
            errors.append(f"Record {idx} duplicate query_id: {qid}")
        else:
            query_ids.add(qid)

        q = rec.get("query", "")
        if not q or not isinstance(q, str) or not q.strip():
            errors.append(f"Record {qid} has empty query")

        rel_chunks = rec.get("relevant_chunk_ids", [])
        if not isinstance(rel_chunks, list) or not rel_chunks:
            errors.append(f"Record {qid} has no relevant_chunk_ids")

        judgments = rec.get("relevance_judgments", {})
        if not isinstance(judgments, dict) or not judgments:
            errors.append(f"Record {qid} has empty relevance_judgments")
        else:
            for cid, score in judgments.items():
                if score not in (0, 1, 2, 3):
                    errors.append(f"Record {qid} invalid relevance score for {cid}: {score} (must be 0, 1, 2, or 3)")

        exp_arts = rec.get("expected_article_numbers", [])
        if not isinstance(exp_arts, list) or not exp_arts:
            errors.append(f"Record {qid} has empty expected_article_numbers")

        cat = rec.get("query_category", "")
        if cat not in allowed_categories:
            errors.append(f"Record {qid} invalid category: {cat}")
        else:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "valid": len(errors) == 0,
        "total_queries": len(records),
        "unique_query_ids": len(query_ids),
        "category_breakdown": category_counts,
        "errors": errors[:20],
        "error_count": len(errors),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Building Golden Reranking Evaluation Dataset...")
    dataset = build_golden_evaluation_dataset()
    val = validate_golden_evaluation_dataset(dataset)
    print("=" * 60)
    print("DATASET B: GOLDEN RERANKING DATASET BUILT SUCCESSFULLY")
    print("=" * 60)
    print(f"Total Queries:       {val['total_queries']}")
    print(f"Unique Query IDs:    {val['unique_query_ids']}")
    print("Category Breakdown:")
    for cat, count in val["category_breakdown"].items():
        print(f"  - {cat:22s}: {count}")
    print(f"Validation Status:   {'PASSED' if val['valid'] else 'FAILED'}")
    if val["errors"]:
        print(f"Errors ({val['error_count']}): {val['errors']}")
    print("=" * 60)
