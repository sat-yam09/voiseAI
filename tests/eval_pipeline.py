"""Evaluation of the retrieval pipeline on local chunks.

Reuses src/evaluate.py metrics + src/retrieval.py pipeline classes.
Captures per-stage latency by timing each stage separately.

Usage:
    python tests/eval_pipeline.py
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.evaluate import evaluate_query, load_chunks_as_dataset
from src.hybrid_search import HybridSearch
from src.reranker import Reranker


def main() -> int:
    cfg = Config()
    MAX_QUERIES = 10
    top_k = cfg.top_k  # 5

    # ------------------------------------------------------------------
    # 1. Load chunks and find queries with ground truth
    # ------------------------------------------------------------------
    print("=" * 72)
    print("RETRIEVAL PIPELINE EVALUATION")
    print("=" * 72)

    chunks_by_query = load_chunks_as_dataset(cfg.chunks_path())
    print(f"\nTotal queries in chunk file: {len(chunks_by_query)}")

    # Filter to queries with at least one relevant chunk (is_selected=True)
    eval_queries = {}
    for qid, chunks in chunks_by_query.items():
        relevant = [c for c in chunks if c.get("is_selected", False)]
        if relevant:
            eval_queries[qid] = chunks

    print(f"Queries with ground truth (is_selected=True): {len(eval_queries)}")

    # Take up to MAX_QUERIES
    sample_qids = list(eval_queries.keys())[:MAX_QUERIES]
    sample = {qid: eval_queries[qid] for qid in sample_qids}
    print(f"Evaluating on: {len(sample)} queries (Top-K={top_k})")
    print()

    # ------------------------------------------------------------------
    # 2. Load indices + reranker
    # ------------------------------------------------------------------
    print("[INIT] Loading hybrid search (vector + BM25 indices) ...")
    t0 = time.perf_counter()
    hybrid = HybridSearch(cfg)
    hybrid.load_index()
    init_hybrid_ms = (time.perf_counter() - t0) * 1000
    print(f"       Hybrid search loaded in {init_hybrid_ms:.0f}ms")

    print("[INIT] Loading cross-encoder reranker ...")
    t0 = time.perf_counter()
    reranker = Reranker(cfg)
    # Force model load by doing a dummy predict is not needed; accessing .model loads it.
    _ = reranker.model
    init_rerank_ms = (time.perf_counter() - t0) * 1000
    print(f"       Reranker loaded in {init_rerank_ms:.0f}ms")
    print()

    # ------------------------------------------------------------------
    # 3. Evaluate each query
    # ------------------------------------------------------------------
    per_query_results = []
    all_recall = {k: [] for k in cfg.eval_metrics_k}
    all_precision = {k: [] for k in cfg.eval_metrics_k}
    all_hit = {k: [] for k in cfg.eval_metrics_k}
    all_mrr = []
    vec_latencies = []
    bm25_latencies = []
    hybrid_latencies = []
    rerank_latencies = []
    total_latencies = []

    for idx, (qid, chunks) in enumerate(sample.items(), start=1):
        query_text = chunks[0].get("query", "")
        relevant_ids = [c["chunk_id"] for c in chunks if c.get("is_selected", False)]

        if not query_text or not relevant_ids:
            continue

        print(f"[{idx}/{len(sample)}] Query ID: {qid}")
        print(f"  Query: {query_text}")
        print(f"  Relevant chunks: {len(relevant_ids)} ({', '.join(relevant_ids)})")

        # --- Vector retrieval ---
        t_vec = time.perf_counter()
        vec_results = hybrid.vector_retriever.search(
            query_text, top_k=cfg.vector_top_k
        )
        vec_ms = (time.perf_counter() - t_vec) * 1000

        # --- BM25 retrieval ---
        t_bm = time.perf_counter()
        bm25_results = hybrid.bm25_retriever.search(
            query_text, top_k=cfg.bm25_top_k
        )
        bm25_ms = (time.perf_counter() - t_bm) * 1000

        # --- RRF fusion ---
        t_hyb = time.perf_counter()
        candidates = hybrid.search(query_text, top_k=cfg.rerank_top_n)
        hybrid_ms = (time.perf_counter() - t_hyb) * 1000

        # --- Reranking ---
        t_rr = time.perf_counter()
        reranked = reranker.rerank(query_text, candidates, top_n=cfg.rerank_top_n)
        rerank_ms = (time.perf_counter() - t_rr) * 1000

        # --- Final Top-K ---
        final = reranked[:top_k]
        retrieved_ids = [r.get("chunk_id", "") for r in final]
        total_ms = vec_ms + bm25_ms + rerank_ms  # hybrid includes vec+bm25, so don't double count

        # Store latencies
        vec_latencies.append(vec_ms)
        bm25_latencies.append(bm25_ms)
        hybrid_latencies.append(hybrid_ms)
        rerank_latencies.append(rerank_ms)
        total_latencies.append(total_ms)

        # Compute metrics for each K
        for k in cfg.eval_metrics_k:
            m = evaluate_query(relevant_ids, retrieved_ids, k=k)
            all_recall[k].append(m["recall_k"])
            all_precision[k].append(m["precision_k"])
            all_hit[k].append(m["hit_k"])

        mrr_m = evaluate_query(relevant_ids, retrieved_ids, k=100)
        all_mrr.append(mrr_m["mrr"])

        print(f"  Retrieved: {retrieved_ids}")
        print(f"  Vec: {vec_ms:.0f}ms | BM25: {bm25_ms:.0f}ms | "
              f"Hybrid(RRF): {hybrid_ms:.0f}ms | Rerank: {rerank_ms:.0f}ms | "
              f"Total: {total_ms:.0f}ms")
        print()

    # ------------------------------------------------------------------
    # 4. Aggregate metrics
    # ------------------------------------------------------------------
    n = len(sample)

    print("=" * 72)
    print("AGGREGATE RESULTS")
    print("=" * 72)
    print(f"\nQueries evaluated: {n}")
    print(f"Top-K: {top_k}")
    print(f"Eval K values: {cfg.eval_metrics_k}")

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print("\n--- Retrieval Quality ---")
    for k in cfg.eval_metrics_k:
        print(f"  Recall@{k}:    {mean(all_recall[k]):.4f}")
        print(f"  Precision@{k}: {mean(all_precision[k]):.4f}")
        print(f"  Hit Rate@{k}:  {mean(all_hit[k]):.4f}")
    print(f"  MRR:           {mean(all_mrr):.4f}")

    print("\n--- Latency (ms) ---")
    print(f"  Vector retrieval:     mean={mean(vec_latencies):.1f}  "
          f"median={sorted(vec_latencies)[len(vec_latencies)//2]:.1f}")
    print(f"  BM25 retrieval:       mean={mean(bm25_latencies):.1f}  "
          f"median={sorted(bm25_latencies)[len(bm25_latencies)//2]:.1f}")
    print(f"  Hybrid (RRF):         mean={mean(hybrid_latencies):.1f}  "
          f"median={sorted(hybrid_latencies)[len(hybrid_latencies)//2]:.1f}")
    print(f"  Reranking:            mean={mean(rerank_latencies):.1f}  "
          f"median={sorted(rerank_latencies)[len(rerank_latencies)//2]:.1f}")
    print(f"  Total pipeline:       mean={mean(total_latencies):.1f}  "
          f"median={sorted(total_latencies)[len(total_latencies)//2]:.1f}")
    print()

    # ------------------------------------------------------------------
    # 5. Chunking config availability
    # ------------------------------------------------------------------
    print("--- Chunking Configurations ---")
    configs_to_check = [
        (256, 32),
        (512, 64),
        (1024, 128),
    ]
    for size, overlap in configs_to_check:
        path = os.path.join(cfg.chunks_dir, f"chunks_{size}_{overlap}.jsonl")
        exists = os.path.exists(path)
        status = "EXISTS" if exists else "MISSING"
        print(f"  chunks_{size}_{overlap}.jsonl: {status}")

    print("\n" + "=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
