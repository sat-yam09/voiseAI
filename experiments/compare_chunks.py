"""Compare retrieval quality across chunk configurations WITHOUT reranker.

Measures Recall, MRR, Hit Rate, and per-stage latency for each config.
Saves results to experiments/chunk_comparison.json.

Usage:
    python experiments/compare_chunks.py
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
from src.retriever import VectorRetriever
from src.bm25 import BM25Retriever
from src.hybrid_search import HybridSearch


CONFIGS = [
    {"chunk_size_words": 256, "overlap_words": 32},
    {"chunk_size_words": 512, "overlap_words": 64},
    {"chunk_size_words": 1024, "overlap_words": 128},
]

MAX_QUERIES = 20  # queries with ground truth to evaluate


def load_chunks_from_path(path: str) -> List[Dict]:
    """Load chunk JSONL into a list of dicts."""
    chunks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_indices_for_config(cfg: Config) -> None:
    """Build vector + BM25 indices from the config's chunk file."""
    chunks_path = cfg.chunks_path()
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks not found: {chunks_path}")

    chunks = load_chunks_from_path(chunks_path)
    print(f"    Loaded {len(chunks)} chunks from {os.path.basename(chunks_path)}")

    # Build vector index
    print(f"    Embedding chunks ...")
    vr = VectorRetriever(cfg)
    t0 = time.perf_counter()
    n = vr.build_index(chunks)
    vec_build = time.perf_counter() - t0
    print(f"    Vector index: {n} vectors in {vec_build:.1f}s")

    # Build BM25 index
    bm = BM25Retriever(cfg)
    bm.build_index(chunks)
    bm.save()
    print(f"    BM25 index: {bm.size} docs")


def evaluate_config(cfg: Config, eval_queries: Dict, ks: tuple) -> Dict[str, Any]:
    """Run hybrid search (no reranker) on eval queries and compute metrics."""
    hybrid = HybridSearch(cfg)
    hybrid.load_index()

    all_recall = {k: [] for k in ks}
    all_precision = {k: [] for k in ks}
    all_hit = {k: [] for k in ks}
    all_mrr = []
    vec_latencies = []
    bm25_latencies = []
    hybrid_latencies = []
    evaluated = 0

    for qid, chunks in eval_queries.items():
        query_text = chunks[0].get("query", "")
        relevant_ids = [c["chunk_id"] for c in chunks if c.get("is_selected", False)]
        if not query_text or not relevant_ids:
            continue

        # Vector retrieval timing
        t_v = time.perf_counter()
        vec_res = hybrid.vector_retriever.search(query_text, top_k=cfg.vector_top_k)
        vec_ms = (time.perf_counter() - t_v) * 1000

        # BM25 retrieval timing
        t_b = time.perf_counter()
        bm25_res = hybrid.bm25_retriever.search(query_text, top_k=cfg.bm25_top_k)
        bm25_ms = (time.perf_counter() - t_b) * 1000

        # Hybrid (RRF) timing
        t_h = time.perf_counter()
        candidates = hybrid.search(query_text, top_k=cfg.hybrid_top_k)
        hybrid_ms = (time.perf_counter() - t_h) * 1000

        vec_latencies.append(vec_ms)
        bm25_latencies.append(bm25_ms)
        hybrid_latencies.append(hybrid_ms)
        evaluated += 1

        # Use hybrid candidates as retrieved results (no reranker)
        retrieved_ids = [r.get("chunk_id", "") for r in candidates]

        for k in ks:
            m = evaluate_query(relevant_ids, retrieved_ids, k=k)
            all_recall[k].append(m["recall_k"])
            all_precision[k].append(m["precision_k"])
            all_hit[k].append(m["hit_k"])

        mrr_m = evaluate_query(relevant_ids, retrieved_ids, k=100)
        all_mrr.append(mrr_m["mrr"])

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    result = {
        "evaluated_queries": evaluated,
        "ks": list(ks),
    }
    for k in ks:
        result[f"recall@{k}"] = round(mean(all_recall[k]), 4)
        result[f"precision@{k}"] = round(mean(all_precision[k]), 4)
        result[f"hit_rate@{k}"] = round(mean(all_hit[k]), 4)
    result["mrr"] = round(mean(all_mrr), 4)
    result["vec_latency_mean_ms"] = round(mean(vec_latencies), 1)
    result["bm25_latency_mean_ms"] = round(mean(bm25_latencies), 1)
    result["hybrid_latency_mean_ms"] = round(mean(hybrid_latencies), 1)

    return result


def main() -> int:
    print("=" * 72)
    print("CHUNK CONFIGURATION COMPARISON (no reranker)")
    print("=" * 72)

    base_cfg = Config()
    ks = base_cfg.eval_metrics_k

    # Load evaluation queries from 256/32 chunks (ground truth source)
    print("\n[1/7] Loading ground truth from 256/32 chunks ...")
    chunks_by_query = load_chunks_as_dataset(base_cfg.chunks_path())
    eval_pool = {}
    for qid, chunks in chunks_by_query.items():
        if any(c.get("is_selected") for c in chunks):
            eval_pool[qid] = chunks
    eval_qids = list(eval_pool.keys())[:MAX_QUERIES]
    eval_queries = {qid: eval_pool[qid] for qid in eval_qids}
    print(f"       {len(eval_queries)} queries with ground truth selected")

    all_results = {}

    for step, cfg_params in enumerate(CONFIGS):
        size, overlap = cfg_params["chunk_size_words"], cfg_params["overlap_words"]
        label = f"{size}/{overlap}"
        print(f"\n{'='*72}")
        print(f"[{step+2}/7] Config: {label} (chunk_size={size}, overlap={overlap})")
        print(f"{'='*72}")

        # Create a Config with a temp index dir for this config
        idx_dir = os.path.join(base_cfg.index_dir, f"eval_{size}_{overlap}")
        cfg = Config(
            chunk_size_words=size,
            overlap_words=overlap,
            index_dir=idx_dir,
        )

        chunks_path = cfg.chunks_path()
        if not os.path.exists(chunks_path):
            print(f"  SKIP: {chunks_path} not found")
            continue

        # Count chunks
        chunk_count = sum(1 for _ in open(chunks_path, "r", encoding="utf-8"))
        print(f"  Chunks: {chunk_count}")

        # Each config has different chunk texts/IDs, so indices must be
        # config-specific. Check if already built in this config's dir.
        vec_file = os.path.join(idx_dir, cfg.vector_index_name)
        meta_file = os.path.join(idx_dir, cfg.metadata_name)
        bm25_file = os.path.join(idx_dir, "bm25_index.json")

        if os.path.exists(vec_file) and os.path.exists(meta_file) and os.path.exists(bm25_file):
            print("  Indices found on disk — loading.")
        else:
            print("  Building indices ...")
            build_indices_for_config(cfg)

        # Evaluate
        print(f"  Evaluating {len(eval_queries)} queries (no reranker) ...")
        t0 = time.perf_counter()
        metrics = evaluate_config(cfg, eval_queries, ks)
        eval_time = time.perf_counter() - t0
        print(f"  Done in {eval_time:.1f}s")

        all_results[label] = metrics

        # Print summary for this config
        print(f"  Recall@5={metrics['recall@5']:.4f}  "
              f"Recall@10={metrics['recall@10']:.4f}  "
              f"MRR={metrics['mrr']:.4f}  "
              f"Hit@5={metrics['hit_rate@5']:.4f}  "
              f"Hybrid={metrics['hybrid_latency_mean_ms']:.0f}ms")

    # Save results
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "chunk_comparison.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Print comparison table
    print("\n" + "=" * 72)
    print("COMPARISON TABLE (hybrid search, no reranker)")
    print("=" * 72)
    header = f"{'Config':<10} {'Recall@1':>9} {'Recall@3':>9} {'Recall@5':>9} {'Recall@10':>10} {'MRR':>7} {'Hit@5':>7} {'Hybrid(ms)':>11}"
    print(header)
    print("-" * len(header))
    for label, m in all_results.items():
        print(f"{label:<10} {m['recall@1']:>9.4f} {m['recall@3']:>9.4f} "
              f"{m['recall@5']:>9.4f} {m['recall@10']:>10.4f} "
              f"{m['mrr']:>7.4f} {m['hit_rate@5']:>7.4f} "
              f"{m['hybrid_latency_mean_ms']:>11.1f}")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
