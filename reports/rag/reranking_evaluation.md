# ConstitutionGPT — RAG Retrieval vs. Reranking Evaluation Report

**Generated:** 2026-10-28T15:50:00.778452+00:00  
**Benchmark Corpus:** Constitution of India (`data/rag/normal_retrieval_dataset.jsonl`)  
**Evaluation Dataset:** `tests/evaluation/golden_reranking_dataset.jsonl` (25 verified queries)  

## 1. Latency & Execution Time Comparison

| Stage | Operation | Average Latency (ms) | Speed Characteristics |
| :--- | :--- | :---: | :--- |
| **Stage 1 (Normal Retrieval)** | Full Knowledge Base Inverted Token Search | `17.12 ms` | Ultra-fast in-memory indexed vector/term scan |
| **Stage 2 (Reranking)** | Deep Semantic / Cross-Score Evaluation | `0.11 ms` | Candidate contextual scoring & re-weighting |
| **End-to-End Pipeline** | Combined Stage 1 + Stage 2 | `17.23 ms` | Complete candidate selection & reordering |

## 2. Quality Metrics & Ranking Differences

| Metric | Stage 1 (Initial Retrieval) | Stage 2 (Reranked) | Delta | Direction |
| :--- | :---: | :---: | :---: | :---: |
| **MRR (Mean Reciprocal Rank)** | `0.8089` | `0.7403` | `+-0.0686` | 🔻 Regressed |
| **nDCG@5** | `0.7079` | `0.6533` | `+-0.0546` | 🔻 Regressed |
| **nDCG@10** | `0.7243` | `0.6749` | `+-0.0494` | 🔻 Regressed |
| **Recall@5** | `0.7800` | `0.7800` | `+0.0000` | ➖ Neutral |
| **Recall@10** | `0.8400` | `0.8400` | `+0.0000` | ➖ Neutral |
| **Source Hit Rate @ 5** | `84.0%` | `84.0%` | `+0.0%` | ➖ Neutral |
| **Source Hit Rate @ 10** | `92.0%` | `92.0%` | `+0.0%` | ➖ Neutral |

### Transition & Movement Breakdown
- **Queries Improved:** `3` (12.0%)
- **Queries Unchanged:** `17` (68.0%)
- **Queries Regressed:** `5` (20.0%)
- **Top-1 Chunk Switches:** `7` of 25 queries had their #1 ranked chunk updated

---

## 3. Category Performance & Timing Breakdown

| Category | Queries | Stg 1 Latency | Stg 2 Latency | Stg 1 MRR | Stg 2 MRR | Stg 1 nDCG@5 | Stg 2 nDCG@5 | Improved | Neutral | Regressed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `exact_article_lookup` | 5 | `8.1 ms` | `0.1 ms` | `1.0000` | `1.0000` | `0.9006` | `0.9006` | 0 | 5 | 0 |
| `paraphrase` | 5 | `32.0 ms` | `0.2 ms` | `0.6444` | `0.5267` | `0.5196` | `0.4558` | 1 | 2 | 2 |
| `conceptual_query` | 5 | `24.9 ms` | `0.1 ms` | `0.7000` | `0.3750` | `0.6704` | `0.3957` | 0 | 2 | 3 |
| `multi_article_query` | 5 | `13.6 ms` | `0.1 ms` | `0.7000` | `0.8000` | `0.5715` | `0.6368` | 2 | 3 | 0 |
| `hard_negative` | 5 | `7.0 ms` | `0.1 ms` | `1.0000` | `1.0000` | `0.8775` | `0.8775` | 0 | 5 | 0 |

---

## 4. Per-Query Execution Time & Detailed Differences

| Query ID | Category | Normal Time | Rerank Time | Stg 1 Top Chunk | Stg 2 Top Chunk | Stg 1 MRR | Stg 2 MRR | nDCG@5 Delta | Status |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `gold_001` | `exact_article_lookup` | `0.3 ms` | `0.0 ms` | `const_14_p37_c01` | `const_14_p37_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_002` | `exact_article_lookup` | `0.0 ms` | `0.0 ms` | `const_21_p42_c01` | `const_21_p42_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_003` | `exact_article_lookup` | `0.0 ms` | `0.2 ms` | `const_32_p50_c01` | `const_32_p50_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_004` | `exact_article_lookup` | `0.0 ms` | `0.0 ms` | `const_51a_p56_c01` | `const_51a_p56_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_005` | `exact_article_lookup` | `40.2 ms` | `0.2 ms` | `const_preamble_p32_c01` | `const_preamble_p32_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_006` | `paraphrase` | `29.2 ms` | `0.2 ms` | `const_77_p66_c01` | `const_77_p66_c01` | `0.11` | `0.10` | `+0.0000` | ➖ unchanged |
| `gold_007` | `paraphrase` | `30.8 ms` | `0.2 ms` | `const_21_p42_c01` | `const_21_p42_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_008` | `paraphrase` | `32.0 ms` | `0.2 ms` | `const_32_p50_c01` | `const_124_p88_c04` | `1.00` | `0.33` | `-0.5741` | ⚠️ regressed |
| `gold_009` | `paraphrase` | `34.5 ms` | `0.2 ms` | `const_265_p183_c01` | `const_119_p334_c02` | `0.11` | `0.20` | `+0.3330` | ✅ improved |
| `gold_010` | `paraphrase` | `33.7 ms` | `0.2 ms` | `const_24_p45_c01` | `const_24_p45_c01` | `1.00` | `1.00` | `-0.0776` | ⚠️ regressed |
| `gold_011` | `conceptual_query` | `31.2 ms` | `0.2 ms` | `const_20_p42_c01` | `const_22_p42_c01` | `1.00` | `0.50` | `-0.2723` | ⚠️ regressed |
| `gold_012` | `conceptual_query` | `0.0 ms` | `0.1 ms` | `const_6_p309_c01` | `const_6_p309_c01` | `0.00` | `0.00` | `+0.0000` | ➖ unchanged |
| `gold_013` | `conceptual_query` | `31.0 ms` | `0.2 ms` | `const_25_p349_c01` | `const_19_p40_c01` | `1.00` | `0.25` | `-0.5223` | ⚠️ regressed |
| `gold_014` | `conceptual_query` | `30.3 ms` | `0.2 ms` | `const_50_p55_c01` | `const_50_p55_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_015` | `conceptual_query` | `32.0 ms` | `0.2 ms` | `const_52_p57_c01` | `const_392_p282_c01` | `0.50` | `0.12` | `-0.5788` | ⚠️ regressed |
| `gold_016` | `multi_article_query` | `35.2 ms` | `0.2 ms` | `const_8_p341_c01` | `const_32_p50_c02` | `0.00` | `0.00` | `+0.0326` | ✅ improved |
| `gold_017` | `multi_article_query` | `0.0 ms` | `0.0 ms` | `const_14_p37_c01` | `const_14_p37_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_018` | `multi_article_query` | `0.0 ms` | `0.0 ms` | `const_352_p239_c01` | `const_352_p239_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_019` | `multi_article_query` | `32.6 ms` | `0.2 ms` | `const_16_p39_c01` | `const_23_p45_c01` | `0.50` | `1.00` | `+0.2937` | ✅ improved |
| `gold_020` | `multi_article_query` | `0.0 ms` | `0.0 ms` | `const_368_p35_c01` | `const_368_p35_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_021` | `hard_negative` | `0.0 ms` | `0.1 ms` | `const_32_p50_c01` | `const_32_p50_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_022` | `hard_negative` | `0.0 ms` | `0.0 ms` | `const_19_p40_c01` | `const_19_p40_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_023` | `hard_negative` | `0.0 ms` | `0.0 ms` | `const_356_p35_c01` | `const_356_p35_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_024` | `hard_negative` | `35.0 ms` | `0.2 ms` | `const_44_p54_c01` | `const_44_p54_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |
| `gold_025` | `hard_negative` | `0.0 ms` | `0.0 ms` | `const_1_p33_c01` | `const_1_p33_c01` | `1.00` | `1.00` | `+0.0000` | ➖ unchanged |