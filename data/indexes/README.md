# ConstitutionGPT — Index Specifications & Storage

## Overview
This directory contains index artifacts and manifests for ConstitutionGPT's dual-retrieval system:

1. **Document Corpus:** `data/processed/constitution_chunks.jsonl` (440 chunks, 320 articles, 188 pages).
2. **Dense-Sparse Vector Index:** In-memory Term Frequency-Inverse Document Frequency (TF-IDF) feature matrix with sublinear scaling, unigram+bigram n-grams, and L2 unit-sphere vector normalization.
3. **Lexical Index:** Okapi BM25 (`rank_bm25.BM25Okapi`) initialized over pre-tokenized constitutional text chunks.
4. **Reciprocal Rank Fusion (RRF):**
   $$RRF(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + r_m(d)}$$
5. **Cross-Encoder Reranker:** Multi-factor analytical cross-scorer with exact provision matching, title overlap, and text density weighting.
