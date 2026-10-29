"""Cost Optimization & Performance Dashboard Component for ConstitutionGPT.

Renders real-time telemetry, cache statistics, indexing analytics,
reproducible golden evaluation metrics, and interactive cost controls in Streamlit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from app.cache.semantic_cache import get_semantic_cache
from app.cost.cost_tracker import get_cost_tracker
from app.retrieval.incremental_indexer import get_incremental_indexer
from config.settings import get_settings

logger = logging.getLogger(__name__)


def render_cost_optimization_dashboard() -> None:
    """Render the comprehensive Cost Optimization & Performance Section."""
    settings = get_settings()
    cost_tracker = get_cost_tracker()
    sem_cache = get_semantic_cache()
    indexer = get_incremental_indexer()

    st.markdown("### 💰 Cost Optimization & Performance Dashboard")
    st.caption("Live SQLite telemetry, token analytics, semantic caching, and incremental indexing stats.")

    tab_overview, tab_retrieval, tab_cache, tab_indexing, tab_controls = st.tabs([
        "📊 Overview",
        "🎯 Retrieval",
        "⚡ Caching",
        "📦 Indexing",
        "⚙️ Controls",
    ])

    summary = cost_tracker.get_summary_metrics()
    cache_metrics = sem_cache.get_metrics()
    indexing_report = indexer.get_last_report()

    # -----------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # -----------------------------------------------------------------------
    with tab_overview:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", f"{summary['total_queries']:,}")
            st.metric("Avg Cost / Query", f"${summary['avg_cost_per_query']:.5f}")
        with col2:
            st.metric("Cost / 1K Queries", f"${summary['cost_per_1000_queries']:.4f}")
            st.metric("Total Tokens", f"{summary['total_tokens']:,}")
        with col3:
            st.metric("Cache Hit Rate", f"{summary['cache_hit_rate_pct']:.1f}%")
            st.metric("p95 Latency", f"{summary['p95_latency_ms']:.1f} ms")

        st.markdown("---")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(f"**Total Estimated Cloud Cost:** `${summary['total_cost_usd']:.5f} USD`")
            st.caption(f"Configured Pricing: Input `${settings.gemini_input_price_per_million}/1M` · Output `${settings.gemini_output_price_per_million}/1M`")
        with col_t2:
            st.markdown(f"**Estimated Cache Savings:** `${summary['estimated_savings_usd']:.5f} USD`")
            st.caption("Calculation Method: Avoided external Gemini LLM invocations × average query cost.")

        # Recent Requests Log Table
        st.markdown("##### 🕒 Recent Request Telemetry")
        recent_reqs = cost_tracker.get_recent_requests(limit=10)
        if not recent_reqs:
            st.info("No queries recorded yet in this telemetry session. Try asking a constitutional question!")
        else:
            table_rows = []
            for r in recent_reqs:
                table_rows.append({
                    "Timestamp": r["timestamp"][:19].replace("T", " "),
                    "Mode": r["chatbot_mode"],
                    "Tokens (In/Out)": f"{r['llm_input_tokens']} / {r['llm_output_tokens']}",
                    "Latency (ms)": f"{r['total_latency_ms']:.1f}",
                    "Cache": r["cache_hit"],
                    "Cost ($)": f"${r['estimated_cost_usd']:.6f}",
                })
            st.dataframe(table_rows, use_container_width=True)

    # -----------------------------------------------------------------------
    # TAB 2: RETRIEVAL ANALYTICS
    # -----------------------------------------------------------------------
    with tab_retrieval:
        st.markdown("##### 🔍 Retrieval & Reranker Diagnostics")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Configured top_k", settings.rag_top_k)
            st.metric("Configured rerank_top_k", settings.rerank_top_k)
            st.markdown(f"**Reranking Status:** {'✅ Enabled' if settings.reranking_enabled else '⚡ Bypassed (Low-Cost)'}")
        with col_r2:
            st.metric("Avg Retrieval Latency", f"{summary['avg_retrieval_latency_ms']:.2f} ms")
            st.metric("Avg Reranking Latency", f"{summary['avg_reranking_latency_ms']:.2f} ms")

        # Load Golden Evaluation Metrics if available
        comp_file = settings.reports_dir / "comparison.json"
        if comp_file.exists():
            try:
                with open(comp_file, "r", encoding="utf-8") as f:
                    comp_data = json.load(f)
                st.markdown("---")
                st.markdown("##### 🏆 Held-Out Golden Retrieval Benchmark Metrics")
                m_cols = st.columns(4)
                with m_cols[0]:
                    r1 = comp_data.get("recall_at_1", {}).get("advanced", 0.0) * 100.0
                    st.metric("Recall@1", f"{r1:.1f}%", f"+{comp_data.get('recall_at_1', {}).get('pct_change', 0):.1f}%")
                with m_cols[1]:
                    r3 = comp_data.get("recall_at_3", {}).get("advanced", 0.0) * 100.0
                    st.metric("Recall@3", f"{r3:.1f}%", f"+{comp_data.get('recall_at_3', {}).get('pct_change', 0):.1f}%")
                with m_cols[2]:
                    r10 = comp_data.get("recall_at_10", {}).get("advanced", 0.0) * 100.0
                    st.metric("Recall@10", f"{r10:.1f}%", f"+{comp_data.get('recall_at_10', {}).get('pct_change', 0):.1f}%")
                with m_cols[3]:
                    hit10 = comp_data.get("source_hit_at_10_pct", {}).get("advanced", 0.0)
                    st.metric("Hit Rate@10", f"{hit10:.1f}%", f"+{comp_data.get('source_hit_at_10_pct', {}).get('pct_change', 0):.1f}%")
            except Exception as e:
                st.caption(f"Could not parse comparison.json: {e}")
        else:
            st.caption("No offline benchmark run found. You can run one from the Controls tab.")

    # -----------------------------------------------------------------------
    # TAB 3: CACHE ANALYTICS
    # -----------------------------------------------------------------------
    with tab_cache:
        st.markdown("##### ⚡ Two-Tier Semantic & Exact Query Cache")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Cached Entries", cache_metrics["total_entries"])
            st.metric("Exact Cache Hits", cache_metrics["exact_hits"])
        with c2:
            st.metric("Semantic Cache Hits", cache_metrics["semantic_hits"])
            st.metric("Cache Misses", cache_metrics["misses"])
        with c3:
            st.metric("Cache Bypasses", cache_metrics["bypasses"])
            st.metric("Stale Invalidations", cache_metrics["invalidations"])

        st.markdown("---")
        st.markdown(
            f"""
            - **Exact Match Cache:** `{'✅ Active' if settings.exact_cache_enabled else '❌ Disabled'}` (0.05ms bypass)
            - **Semantic Match Cache:** `{'✅ Active' if settings.semantic_cache_enabled else '❌ Disabled'}` (Cosine similarity threshold $\\ge$ `{settings.semantic_cache_threshold}`)
            - **TTL Expiration Policy:** `{settings.cache_ttl_seconds // 3600} hours`
            - **Max Capacity:** `{settings.cache_max_entries} entries` (LRU eviction)
            - **Constitutional Guard:** Enforces exact article number matching to prevent semantic crosstalk between different legal articles.
            """
        )

    # -----------------------------------------------------------------------
    # TAB 4: INDEXING ANALYTICS
    # -----------------------------------------------------------------------
    with tab_indexing:
        st.markdown("##### 📦 Incremental Ingestion & Vector Manifest")
        manifest_data = indexer.load_manifest()
        chunks_map = manifest_data.get("chunks", {})
        total_indexed = manifest_data.get("total_chunks", len(chunks_map))

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.metric("Total Chunks in Manifest", f"{total_indexed}")
            st.metric("Embedding Space", f"{manifest_data.get('model', 'tfidf-sublinear-l2')}")
        with col_i2:
            if indexing_report:
                st.metric("Last Ingestion Status", indexing_report.status.replace("_", " ").title())
                st.metric("Chunks Reused", f"{indexing_report.reused_count}")
            else:
                st.metric("Chunks Reused", f"{total_indexed}")
                st.metric("Last Ingestion Status", "Reused Cached Index")

        st.markdown("---")
        st.markdown(f"**Manifest Path:** `{settings.indexing_manifest_path}`")
        st.markdown(f"**Vector Store:** `{settings.persisted_embeddings_path.parent / 'persisted_vectors.pkl'}`")
        st.caption("Change detection computes SHA-256 content hashes per chunk. Unchanged documents reuse existing representations with zero re-embedding overhead.")

    # -----------------------------------------------------------------------
    # TAB 5: CONTROLS & MANAGEMENT
    # -----------------------------------------------------------------------
    with tab_controls:
        st.markdown("##### ⚙️ Interactive Optimization Controls")

        c_tgl1, c_tgl2 = st.columns(2)
        with c_tgl1:
            new_exact = st.checkbox("Enable Exact Query Cache", value=settings.exact_cache_enabled)
            new_sem = st.checkbox("Enable Semantic Query Cache", value=settings.semantic_cache_enabled)
            new_rerank = st.checkbox("Enable Cross-Encoder Reranking", value=settings.reranking_enabled)
        with c_tgl2:
            new_top_k = st.slider("Retrieval top_k", min_value=1, max_value=15, value=settings.rag_top_k)
            new_threshold = st.slider("Semantic Similarity Threshold", min_value=0.70, max_value=0.98, value=settings.semantic_cache_threshold, step=0.01)
            new_tokens = st.slider("Max Output Tokens", min_value=200, max_value=1200, value=settings.max_output_tokens, step=50)

        if st.button("💾 Apply Configuration Changes", key="btn_apply_cost_configs", use_container_width=True):
            settings.exact_cache_enabled = new_exact
            settings.semantic_cache_enabled = new_sem
            settings.reranking_enabled = new_rerank
            settings.rag_top_k = new_top_k
            settings.semantic_cache_threshold = new_threshold
            settings.max_output_tokens = new_tokens
            sem_cache.threshold = new_threshold
            st.toast("Optimization settings applied successfully!", icon="✅")
            st.rerun()

        st.markdown("---")
        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            if st.button("🧹 Clear Response Cache", key="btn_clear_cache", use_container_width=True):
                sem_cache.clear()
                st.toast("Semantic and exact cache purged!", icon="🧹")
                st.rerun()

        with col_act2:
            if st.button("🔄 Trigger Incremental Reindex", key="btn_trigger_reindex", use_container_width=True):
                with st.spinner("Executing incremental indexing and vector hash verification..."):
                    from app.retrieval.dual_pipeline import SharedConstitutionalCorpus
                    rep = SharedConstitutionalCorpus.reindex(force=True)
                    st.toast(f"Reindexing complete! Status: {rep.status} ({rep.total_chunks} chunks)", icon="📦")
                    st.rerun()

        with col_act3:
            if st.button("📊 Clear Telemetry Records", key="btn_clear_telemetry", use_container_width=True):
                cost_tracker.clear_telemetry()
                st.toast("Cost telemetry records reset!", icon="📊")
                st.rerun()
