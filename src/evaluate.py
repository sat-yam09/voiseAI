"""Retrieval evaluation metrics.

Evaluates retrieval quality using the ``is_selected`` field from MSMARCO-XI
as the ground-truth relevance label.

Ground-truth definition
-----------------------
In MSMARCO-XI, each query has multiple candidate passages.  The dataset
marks certain passages as ``is_selected = True`` by human annotators.
These are the gold-standard relevant passages.  We treat:

- **Relevant** = passage with ``is_selected == True`` in the original
  chunk metadata.
- **Not relevant** = passage with ``is_selected == False``.

This is a binary relevance judgment, which is the standard approach for
MSMARCO-based evaluation.

Metrics
-------
- **Recall@K**  : fraction of relevant documents found in top-K.
- **Precision@K** : fraction of top-K documents that are relevant.
- **MRR** (Mean Reciprocal Rank) : 1 / rank of the first relevant result.
- **Hit@K** : 1 if at least one relevant document is in top-K, else 0.
- **Latency** : retrieval, reranking, and total pipeline latency.

Usage
-----
    from src.evaluate import evaluate_query, evaluate_dataset
    from src.config import Config

    cfg = Config()
    results = evaluate_query("query", ["chunk_1", "chunk_5"], cfg)
    summary = evaluate_dataset(dataset, retrieval_fn, cfg)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from src.config import Config


def evaluate_query(
    relevant_ids: List[str],
    retrieved_ids: List[str],
    k: int = 5,
) -> Dict[str, float]:
    """Compute retrieval metrics for a single query.

    Parameters
    ----------
    relevant_ids : chunk_ids that are ground-truth relevant (is_selected=True).
    retrieved_ids : ordered chunk_ids returned by the retriever.
    k : cutoff for @K metrics.

    Returns
    -------
    Dict with keys: recall_k, precision_k, mrr, hit_k.
    """
    if not relevant_ids:
        return {"recall_k": 0.0, "precision_k": 0.0, "mrr": 0.0, "hit_k": 0.0}

    relevant_set = set(relevant_ids)
    retrieved_at_k = retrieved_ids[:k]

    # Recall@K
    found = sum(1 for cid in retrieved_at_k if cid in relevant_set)
    recall_k = found / len(relevant_set) if relevant_set else 0.0

    # Precision@K
    precision_k = found / k if k > 0 else 0.0

    # MRR (reciprocal rank of first relevant result)
    mrr = 0.0
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            mrr = 1.0 / i
            break

    # Hit@K
    hit_k = 1.0 if any(cid in relevant_set for cid in retrieved_at_k) else 0.0

    return {
        "recall_k": recall_k,
        "precision_k": precision_k,
        "mrr": mrr,
        "hit_k": hit_k,
    }


def evaluate_dataset(
    chunks_by_query: Dict[str, List[Dict]],
    retrieval_fn: Callable[[str], List[Dict[str, Any]]],
    config: Optional[Config] = None,
    ks: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Evaluate retrieval over a set of queries.

    Parameters
    ----------
    chunks_by_query : dict mapping query_id -> list of chunk dicts.
        Each chunk dict must have ``chunk_id`` and ``is_selected``.
    retrieval_fn : callable that takes a query string and returns an ordered
        list of result dicts with ``chunk_id``.
    config : pipeline config (for k values).
    ks : tuple of K values to evaluate (default from config).

    Returns
    -------
    Dict with per-K aggregated metrics and latency stats.
    """
    cfg = config or Config()
    k_values = ks or cfg.eval_metrics_k

    # Collect per-query metrics.
    all_metrics: Dict[int, List[Dict]] = {k: [] for k in k_values}
    latencies: List[float] = []
    total_queries = 0

    for qid, chunks in chunks_by_query.items():
        if not chunks:
            continue

        # Extract query text and relevant chunk IDs.
        query_text = chunks[0].get("query", "")
        if not query_text:
            continue

        relevant_ids = [
            c["chunk_id"] for c in chunks if c.get("is_selected", False)
        ]
        if not relevant_ids:
            continue  # no ground truth for this query

        # Retrieve.
        t0 = time.perf_counter()
        results = retrieval_fn(query_text)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        retrieved_ids = [r.get("chunk_id", "") for r in results]
        total_queries += 1

        for k in k_values:
            metrics = evaluate_query(relevant_ids, retrieved_ids, k=k)
            all_metrics[k].append(metrics)

    # Aggregate.
    summary: Dict[str, Any] = {
        "total_queries": total_queries,
        "ks": list(k_values),
    }
    for k in k_values:
        m = all_metrics[k]
        if m:
            summary[f"recall@{k}"] = _mean([x["recall_k"] for x in m])
            summary[f"precision@{k}"] = _mean([x["precision_k"] for x in m])
            summary[f"hit_rate@{k}"] = _mean([x["hit_k"] for x in m])
        else:
            summary[f"recall@{k}"] = 0.0
            summary[f"precision@{k}"] = 0.0
            summary[f"hit_rate@{k}"] = 0.0

    summary["mrr"] = _mean([x["mrr"] for x in all_metrics[k_values[0]]]) if all_metrics[k_values[0]] else 0.0

    # Latency stats.
    if latencies:
        summary["latency_ms_mean"] = round(_mean(latencies), 1)
        summary["latency_ms_median"] = round(_median(latencies), 1)
        summary["latency_ms_p95"] = round(_percentile(latencies, 95), 1)
    else:
        summary["latency_ms_mean"] = 0.0

    return summary


def load_chunks_as_dataset(
    chunks_path: str,
) -> Dict[str, List[Dict]]:
    """Load chunk JSONL and group by query_id for evaluation.

    Returns {query_id: [chunk_dict, ...]}.
    """
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    grouped: Dict[str, List[Dict]] = {}
    with open(chunks_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            qid = str(chunk.get("query_id", ""))
            if qid:
                grouped.setdefault(qid, []).append(chunk)
    return grouped


def save_results(results: Dict[str, Any], path: str) -> None:
    """Save evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _percentile(values: List[float], p: float) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    d = k - f
    return s[f] + d * (s[c] - s[f])
