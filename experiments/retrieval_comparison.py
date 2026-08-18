"""Retrieval method comparison experiment.

Compares three retrieval approaches on the current sample:
  - BM25 only (keyword matching)
  - Vector only (semantic matching)
  - Hybrid RRF (vector + BM25 with Reciprocal Rank Fusion)

Measures:
  - Recall@K (K=1,3,5,10)
  - Hit Rate@K
  - MRR
  - Latency per query

Results are saved to ``experiments/retrieval_comparison.json``.

Usage
-----
    .venv\\Scripts\\python.exe experiments\\retrieval_comparison.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.evaluate import evaluate_query, load_chunks_as_dataset

CHUNKS_DIR = os.path.join("data", "chunks")
RESULTS_DIR = "experiments"


def load_chunks(path: str) -> list:
    """Load chunk JSONL into a list of dicts."""
    chunks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def evaluate_bm25(chunks: list, ks=(1, 3, 5, 10)) -> dict:
    """Evaluate BM25-only retrieval."""
    from src.bm25 import BM25Retriever

    cfg = Config()
    bm25 = BM25Retriever(cfg)
    bm25.build_index(chunks)

    grouped = {}
    for c in chunks:
        qid = str(c.get("query_id", ""))
        if qid:
            grouped.setdefault(qid, []).append(c)

    results = {f"recall@{k}": [] for k in ks}
    results.update({f"hit@{k}": [] for k in ks})
    mrrs = []
    latencies = []

    for qid, qchunks in grouped.items():
        relevant = [c["chunk_id"] for c in qchunks if c.get("is_selected")]
        if not relevant:
            continue
        query_text = qchunks[0].get("query", "")
        if not query_text:
            continue

        t0 = time.perf_counter()
        bm25_results = bm25.search(query_text, top_k=max(ks))
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        retrieved_ids = [r["chunk_id"] for r in bm25_results]

        for k in ks:
            m = evaluate_query(relevant, retrieved_ids, k=k)
            results[f"recall@{k}"].append(m["recall_k"])
            results[f"hit@{k}"].append(m["hit_k"])
        mrrs.append(evaluate_query(relevant, retrieved_ids, k=max(ks))["mrr"])

    summary = {}
    for key, vals in results.items():
        summary[key] = round(sum(vals) / max(len(vals), 1), 4) if vals else 0.0
    summary["mrr"] = round(sum(mrrs) / max(len(mrrs), 1), 4) if mrrs else 0.0
    summary["latency_ms_mean"] = round(sum(latencies) / max(len(latencies), 1), 1) if latencies else 0.0
    summary["total_queries"] = len([v for v in results.get("recall@1", []) if v >= 0])
    return summary


def evaluate_vector(chunks: list, ks=(1, 3, 5, 10)) -> dict:
    """Evaluate vector-only retrieval (requires embeddings)."""
    from src.retriever import VectorRetriever

    cfg = Config()
    vr = VectorRetriever(cfg)

    # Try to load existing index, build if needed
    index_exists = os.path.exists(os.path.join(cfg.index_dir, cfg.vector_index_name))
    if index_exists:
        try:
            vr.load_index()
        except Exception:
            print("  Building vector index (first run)...")
            vr.build_index(chunks)
    else:
        print("  Building vector index (first run, may take a minute)...")
        vr.build_index(chunks)

    grouped = {}
    for c in chunks:
        qid = str(c.get("query_id", ""))
        if qid:
            grouped.setdefault(qid, []).append(c)

    results = {f"recall@{k}": [] for k in ks}
    results.update({f"hit@{k}": [] for k in ks})
    mrrs = []
    latencies = []

    for qid, qchunks in grouped.items():
        relevant = [c["chunk_id"] for c in qchunks if c.get("is_selected")]
        if not relevant:
            continue
        query_text = qchunks[0].get("query", "")
        if not query_text:
            continue

        t0 = time.perf_counter()
        vec_results = vr.search(query_text, top_k=max(ks))
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        retrieved_ids = [r["chunk_id"] for r in vec_results]

        for k in ks:
            m = evaluate_query(relevant, retrieved_ids, k=k)
            results[f"recall@{k}"].append(m["recall_k"])
            results[f"hit@{k}"].append(m["hit_k"])
        mrrs.append(evaluate_query(relevant, retrieved_ids, k=max(ks))["mrr"])

    summary = {}
    for key, vals in results.items():
        summary[key] = round(sum(vals) / max(len(vals), 1), 4) if vals else 0.0
    summary["mrr"] = round(sum(mrrs) / max(len(mrrs), 1), 4) if mrrs else 0.0
    summary["latency_ms_mean"] = round(sum(latencies) / max(len(latencies), 1), 1) if latencies else 0.0
    summary["total_queries"] = len([v for v in results.get("recall@1", []) if v >= 0])
    return summary


def evaluate_hybrid(chunks: list, ks=(1, 3, 5, 10)) -> dict:
    """Evaluate hybrid (vector + BM25 + RRF) retrieval."""
    from src.hybrid_search import HybridSearch

    cfg = Config()
    hs = HybridSearch(cfg)

    # Try to load existing indices
    vec_index_exists = os.path.exists(os.path.join(cfg.index_dir, cfg.vector_index_name))
    bm25_index_exists = os.path.exists(os.path.join(cfg.index_dir, "bm25_index.json"))
    if vec_index_exists and bm25_index_exists:
        try:
            hs.load_index()
        except Exception:
            print("  Building hybrid index (first run)...")
            hs.build_index(chunks)
    else:
        print("  Building hybrid index (first run, may take a minute)...")
        hs.build_index(chunks)

    grouped = {}
    for c in chunks:
        qid = str(c.get("query_id", ""))
        if qid:
            grouped.setdefault(qid, []).append(c)

    results = {f"recall@{k}": [] for k in ks}
    results.update({f"hit@{k}": [] for k in ks})
    mrrs = []
    latencies = []

    for qid, qchunks in grouped.items():
        relevant = [c["chunk_id"] for c in qchunks if c.get("is_selected")]
        if not relevant:
            continue
        query_text = qchunks[0].get("query", "")
        if not query_text:
            continue

        t0 = time.perf_counter()
        hybrid_results = hs.search(query_text, top_k=max(ks))
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        retrieved_ids = [r["chunk_id"] for r in hybrid_results]

        for k in ks:
            m = evaluate_query(relevant, retrieved_ids, k=k)
            results[f"recall@{k}"].append(m["recall_k"])
            results[f"hit@{k}"].append(m["hit_k"])
        mrrs.append(evaluate_query(relevant, retrieved_ids, k=max(ks))["mrr"])

    summary = {}
    for key, vals in results.items():
        summary[key] = round(sum(vals) / max(len(vals), 1), 4) if vals else 0.0
    summary["mrr"] = round(sum(mrrs) / max(len(mrrs), 1), 4) if mrrs else 0.0
    summary["latency_ms_mean"] = round(sum(latencies) / max(len(latencies), 1), 1) if latencies else 0.0
    summary["total_queries"] = len([v for v in results.get("recall@1", []) if v >= 0])
    return summary


def main():
    cfg = Config()
    chunks_path = cfg.chunks_path()

    if not os.path.exists(chunks_path):
        print(f"ERROR: chunks not found at {chunks_path}")
        print("Run chunking first: python src/chunking.py")
        return 1

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    print(f"Config: {cfg.chunk_size_words} words / {cfg.overlap_words} overlap")
    print()

    all_results = {}

    # BM25
    print("=" * 60)
    print("Evaluating BM25 (keyword) retrieval...")
    print("=" * 60)
    t0 = time.perf_counter()
    bm25_results = evaluate_bm25(chunks)
    bm25_time = time.perf_counter() - t0
    bm25_results["evaluation_time_s"] = round(bm25_time, 2)
    all_results["bm25"] = bm25_results
    print(f"  MRR: {bm25_results['mrr']}")
    print(f"  Recall@5: {bm25_results.get('recall@5', 'N/A')}")
    print(f"  Hit@5: {bm25_results.get('hit@5', 'N/A')}")
    print(f"  Latency: {bm25_results['latency_ms_mean']}ms")
    print()

    # Vector (requires embedding model download on first run)
    print("=" * 60)
    print("Evaluating Vector (semantic) retrieval...")
    print("=" * 60)
    try:
        t0 = time.perf_counter()
        vec_results = evaluate_vector(chunks)
        vec_time = time.perf_counter() - t0
        vec_results["evaluation_time_s"] = round(vec_time, 2)
        all_results["vector"] = vec_results
        print(f"  MRR: {vec_results['mrr']}")
        print(f"  Recall@5: {vec_results.get('recall@5', 'N/A')}")
        print(f"  Hit@5: {vec_results.get('hit@5', 'N/A')}")
        print(f"  Latency: {vec_results['latency_ms_mean']}ms")
    except Exception as e:
        print(f"  SKIPPED (embedding model not available): {e}")
        all_results["vector"] = {"skipped": str(e)}
    print()

    # Hybrid
    print("=" * 60)
    print("Evaluating Hybrid (vector + BM25 + RRF) retrieval...")
    print("=" * 60)
    try:
        t0 = time.perf_counter()
        hybrid_results = evaluate_hybrid(chunks)
        hybrid_time = time.perf_counter() - t0
        hybrid_results["evaluation_time_s"] = round(hybrid_time, 2)
        all_results["hybrid"] = hybrid_results
        print(f"  MRR: {hybrid_results['mrr']}")
        print(f"  Recall@5: {hybrid_results.get('recall@5', 'N/A')}")
        print(f"  Hit@5: {hybrid_results.get('hit@5', 'N/A')}")
        print(f"  Latency: {hybrid_results['latency_ms_mean']}ms")
    except Exception as e:
        print(f"  SKIPPED (embedding model not available): {e}")
        all_results["hybrid"] = {"skipped": str(e)}
    print()

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "retrieval_comparison.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<12} {'MRR':>8} {'Recall@5':>10} {'Hit@5':>8} {'Latency':>10}")
    print("-" * 60)
    for method in ["bm25", "vector", "hybrid"]:
        r = all_results.get(method, {})
        if "skipped" in r:
            print(f"{method:<12} {'SKIPPED':>8}")
        else:
            print(
                f"{method:<12} "
                f"{r.get('mrr', 0):>8.4f} "
                f"{r.get('recall@5', 0):>10.4f} "
                f"{r.get('hit@5', 0):>8.4f} "
                f"{r.get('latency_ms_mean', 0):>8.1f}ms"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
