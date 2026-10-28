# ConstitutionGPT: RAG Dataset Design & Specification

## 1. Overview and Architectural Goals

ConstitutionGPT operates as an authoritative legal intelligence system for the **Constitution of India**. To guarantee zero hallucinations, strict constitutional provenance, and rigorous quantitative evaluation, the knowledge base is decoupled into two strictly isolated datasets:

1. **Dataset A: Normal Retrieval Dataset (`data/rag/normal_retrieval_dataset.jsonl`)**
   - The production knowledge corpus indexed by the application.
   - Contains 440 deterministically chunked records across 320 constitutional articles extracted from `data/constitution.pdf` (Official Gazette).
   - Traceable by page, article number, clause, and SHA-256 content hash.
   - **Zero evaluation labels or test queries.**

2. **Dataset B: Golden Reranking Evaluation Dataset (`tests/evaluation/golden_reranking_dataset.jsonl`)**
   - Held-out benchmark dataset containing 25 verified test queries.
   - Spans five distinct difficulty categories: exact lookups, paraphrases, conceptual queries, multi-article interactions, and hard negatives.
   - Evaluated using a 4-tier graded relevance scale (0–3).
   - **Strictly isolated from production retrieval paths.**

---

## 2. Dataset A: Normal Retrieval Dataset Specification

- **Path:** `data/rag/normal_retrieval_dataset.jsonl`
- **Format:** JSON Lines (UTF-8, one valid JSON object per line)
- **Total Chunks:** 440
- **Articles Covered:** 320
- **Pages Covered:** 188

### Schema Definition

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `chunk_id` | `str` | Stable, deterministic identifier: `const_{art}_p{page}_c{chunk}` | `"const_14_p37_c01"` |
| `source_id` | `str` | Stable source provision identifier | `"const_art_14"` |
| `document_id` | `str` | Document identifier | `"constitution_of_india"` |
| `article_number`| `str` | Canonical Article number (e.g., `"14"`, `"21A"`, `"PREAMBLE"`) | `"14"` |
| `section` | `str` | Article title / section header | `"Equality before law"` |
| `page_number` | `int` | Primary page in the official PDF | `37` |
| `text` | `str` | Full verbatim constitutional text for this chunk | `"The State shall not deny to any person..."` |
| `metadata` | `dict`| Part number, page range, chunk index, source filename | `{"part": "Part III", "source_pdf": "constitution.pdf"}` |
| `content_hash` | `str` | SHA-256 hash of the exact chunk text | `"9b12e8..."` |

### Chunking Rules & Segmentation Logic
1. **Length Threshold:** Articles under 1,400 characters remain as single cohesive chunks to preserve full constitutional context.
2. **Clause Preservation:** Multi-clause articles (e.g., Articles 19, 32, 124, 226, 352) are segmented along numbered clause boundaries `(1)`, `(2)`, `(3)` or distinct sub-clauses, ensuring legal integrity is preserved without mid-sentence truncations.
3. **Deterministic Slug IDs:** Identifiers are created deterministically based on canonical article number, page number, and chunk index (`const_{art}_p{page}_c{chunk:02d}`).

### Rebuilding the Normal Retrieval Dataset
The dataset can be deterministically rebuilt at any time:
```bash
python -m app.retrieval.dataset_builder
```

---

## 3. Dataset B: Golden Reranking Evaluation Dataset Specification

- **Path:** `tests/evaluation/golden_reranking_dataset.jsonl`
- **Format:** JSON Lines (UTF-8)
- **Total Test Cases:** 25

### Graded Relevance Scale (0–3)

| Grade | Level | Semantic Definition |
| :---: | :--- | :--- |
| **0** | **Not Relevant** | Unrelated provision, spurious keyword match, or incorrect jurisdiction. |
| **1** | **Related Distractor** | Tangentially related concept (e.g. general equality, procedural clauses) that does not directly answer the query. |
| **2** | **Supporting Provision** | Relevant contextual provision that provides necessary complementary support (e.g., Article 19 in Golden Triangle questions). |
| **3** | **Highly Relevant / Target** | The primary authoritative constitutional provision that directly answers the user's inquiry. |

### Query Categories

1. **`exact_article_lookup` (5 queries):** Direct questions naming the article (e.g., "What is Article 14?", "Explain Article 21").
2. **`paraphrase` (5 queries):** Inquiries asking about constitutional protections in lay language without citing the Article number (e.g., "Can a person be deprived of their liberty without fair legal procedure?").
3. **`conceptual_query` (5 queries):** High-level legal doctrines and principles (e.g., "Protection against double jeopardy and compelled self-incrimination", "Separation of judiciary from executive").
4. **`multi_article_query` (5 queries):** Interconnected constitutional frameworks (e.g., The Golden Triangle: Articles 14, 19, 21; Emergency powers: Articles 352 vs 356).
5. **`hard_negative` (5 queries):** Queries specifically designed with high lexical similarity to distractor articles (e.g., Supreme Court Article 32 vs High Court Article 226; Article 356 President's rule vs Article 21 non-derogable rights).

### Rebuilding the Golden Dataset
```bash
python -m app.retrieval.golden_dataset_builder
```

---

## 4. Strict Isolation & Data Leakage Prevention

To guarantee that evaluation results are valid and reflect true generalization:
1. **Physical Isolation:** Dataset A is stored under `data/rag/`, while Dataset B is stored under `tests/evaluation/`.
2. **Import Isolation:** No module in `app/retrieval/` (except evaluation runners) or `app/graph/` imports `golden_dataset_builder` or references `golden_reranking_dataset.jsonl`.
3. **Runtime Protection:** The production `ConstitutionalKnowledgeBase` and `rerank_candidates` functions only accept query strings and candidates; they have no access to expected answers, target chunk IDs, or relevance judgments.
4. **Automated Leakage Guard:** `tests/test_rag_evaluation.py::test_production_code_strictly_isolated_from_golden_evaluation_data` scans production source files and asserts that zero golden evaluation references exist.
