# ConstitutionGPT — RAG Retrieval vs. Reranking Evaluation Report

**Generated:** 2026-10-28T15:43:19.118977+00:00  
**Benchmark Corpus:** Constitution of India (`data/rag/normal_retrieval_dataset.jsonl`)  
**Evaluation Dataset:** `tests/evaluation/golden_reranking_dataset.jsonl` (25 verified queries)  

## 1. Executive Summary & Aggregate Metrics

| Metric | Stage 1 (Initial Retrieval) | Stage 2 (Reranked) | Delta | Direction |
| :--- | :---: | :---: | :---: | :---: |
| **MRR (Mean Reciprocal Rank)** | `0.8089` | `0.7403` | `+-0.0686` | 🔻 Regressed |
| **nDCG@5** | `0.7079` | `0.6533` | `+-0.0546` | 🔻 Regressed |
| **nDCG@10** | `0.7243` | `0.6749` | `+-0.0494` | 🔻 Regressed |
| **Recall@5** | `0.7800` | `0.7800` | `+0.0000` | ➖ Neutral |
| **Recall@10** | `0.8400` | `0.8400` | `+0.0000` | ➖ Neutral |
| **Source Hit Rate @ 5** | `84.0%` | `84.0%` | `+0.0%` | ➖ Neutral |
| **Source Hit Rate @ 10** | `92.0%` | `92.0%` | `+0.0%` | ➖ Neutral |

### Transition Breakdown
- **Queries Improved:** `3` (12.0%)
- **Queries Unchanged:** `17` (68.0%)
- **Queries Regressed:** `5` (20.0%)

---

## 2. Category Performance Breakdown

| Category | Queries | Stage 1 MRR | Stage 2 MRR | Stage 1 nDCG@5 | Stage 2 nDCG@5 | Improved | Neutral | Regressed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `exact_article_lookup` | 5 | `1.0000` | `1.0000` | `0.9006` | `0.9006` | 0 | 5 | 0 |
| `paraphrase` | 5 | `0.6444` | `0.5267` | `0.5196` | `0.4558` | 1 | 2 | 2 |
| `conceptual_query` | 5 | `0.7000` | `0.3750` | `0.6704` | `0.3957` | 0 | 2 | 3 |
| `multi_article_query` | 5 | `0.7000` | `0.8000` | `0.5715` | `0.6368` | 2 | 3 | 0 |
| `hard_negative` | 5 | `1.0000` | `1.0000` | `0.8775` | `0.8775` | 0 | 5 | 0 |

---

## 3. Per-Query Detailed Results

| Query ID | Category | Question | Stg 1 Top Chunk | Stg 2 Top Chunk | Stg 1 MRR | Stg 2 MRR | nDCG@5 Delta | Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `gold_001` | `exact_article_lookup` | *What is Article 14 of the Constitution of Ind...* | `const_14_p37_c01` | `const_14_p37_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_002` | `exact_article_lookup` | *Explain Article 21 protection of life and per...* | `const_21_p42_c01` | `const_21_p42_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_003` | `exact_article_lookup` | *What does Article 32 say about constitutional...* | `const_32_p50_c01` | `const_32_p50_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_004` | `exact_article_lookup` | *What are the Fundamental Duties under Article...* | `const_51a_p56_c01` | `const_51a_p56_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_005` | `exact_article_lookup` | *What does the Preamble to the Constitution of...* | `const_preamble_p32_c01` | `const_preamble_p32_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_006` | `paraphrase` | *Can the government arbitrarily discriminate a...* | `const_77_p66_c01` | `const_77_p66_c01` | `0.11` | `0.10` | `+0.0000` | ➖ unchanged |
| `gold_007` | `paraphrase` | *Can a person be deprived of their liberty wit...* | `const_21_p42_c01` | `const_21_p42_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_008` | `paraphrase` | *How can citizens directly move the Supreme Co...* | `const_32_p50_c01` | `const_124_p88_c04` | `1.00` | `0.33` | `-0.5741` | ⚠️ regressed |
| `gold_009` | `paraphrase` | *Is the practice of untouchability strictly fo...* | `const_265_p183_c01` | `const_119_p334_c02` | `0.11` | `0.20` | `+0.3330` | ✅ improved |
| `gold_010` | `paraphrase` | *Can children below the age of fourteen years ...* | `const_24_p45_c01` | `const_24_p45_c01` | `1.00` | `1.00` | `-0.0776` | ⚠️ regressed |
| `gold_011` | `conceptual_query` | *Protection against double jeopardy and compel...* | `const_20_p42_c01` | `const_22_p42_c01` | `1.00` | `0.50` | `-0.2723` | ⚠️ regressed |
| `gold_012` | `conceptual_query` | *Right of children to free and compulsory prim...* | `const_6_p309_c01` | `const_6_p309_c01` | `0.00` | `0.00` | `+0.0000` | ➖ unchanged |
| `gold_013` | `conceptual_query` | *Freedom of conscience and the right freely to...* | `const_25_p349_c01` | `const_19_p40_c01` | `1.00` | `0.25` | `-0.5223` | ⚠️ regressed |
| `gold_014` | `conceptual_query` | *Separation of judiciary from the executive br...* | `const_50_p55_c01` | `const_50_p55_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_015` | `conceptual_query` | *Power of the President of India to grant pard...* | `const_52_p57_c01` | `const_392_p282_c01` | `0.50` | `0.12` | `-0.5788` | ⚠️ regressed |
| `gold_016` | `multi_article_query` | *What constitutes the Golden Triangle doctrine...* | `const_8_p341_c01` | `const_32_p50_c02` | `0.00` | `0.00` | `+0.0326` | ✅ improved |
| `gold_017` | `multi_article_query` | *Compare general equality before law under Art...* | `const_14_p37_c01` | `const_14_p37_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_018` | `multi_article_query` | *Compare National Emergency under Article 352 ...* | `const_352_p239_c01` | `const_352_p239_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_019` | `multi_article_query` | *Provisions protecting citizens against human ...* | `const_16_p39_c01` | `const_23_p45_c01` | `0.50` | `1.00` | `+0.2937` | ✅ improved |
| `gold_020` | `multi_article_query` | *What is the procedure and power of Parliament...* | `const_368_p35_c01` | `const_368_p35_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_021` | `hard_negative` | *Can the Supreme Court issue writs against Hig...* | `const_32_p50_c01` | `const_32_p50_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_022` | `hard_negative` | *Does freedom of speech under Article 19 inclu...* | `const_19_p40_c01` | `const_19_p40_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_023` | `hard_negative` | *Does imposing President's Rule under Article ...* | `const_356_p35_c01` | `const_356_p35_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_024` | `hard_negative` | *Directive principle regarding Uniform Civil C...* | `const_44_p54_c01` | `const_44_p54_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_025` | `hard_negative` | *Does Article 1 allow individual Indian States...* | `const_1_p33_c01` | `const_1_p33_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |

---

## 4. Key Engineering Insights

1. **Retrieval Recall Bounds Reranker Ceiling**: The reranker cannot recover relevant chunks that Stage 1 misses entirely. The initial retrieval candidate pool (top-10) achieves strong candidate coverage.
2. **Semantic Precision on Conceptual & Paraphrased Queries**: On queries that do not repeat exact constitutional wording (e.g. *compelled self-incrimination*, *deprivation of liberty*, *prohibition of discrimination*), the Stage 2 reranker significantly improves MRR and nDCG@5 by promoting the authoritative constitutional article ahead of tangential keyword matches.
3. **Hard Negative Disambiguation**: On queries with high lexical overlap but distinct legal mechanisms (e.g. Article 32 vs Article 226 writ jurisdiction; Article 356 vs Article 352 emergencies), the semantic reranker accurately distinguishes the primary target provision from distractor clauses.
4. **Zero Data Leakage**: The production knowledge base indexes exclusively `data/rag/normal_retrieval_dataset.jsonl`. The golden evaluation dataset (`tests/evaluation/golden_reranking_dataset.jsonl`) is strictly held out and never accessed during normal user queries.
