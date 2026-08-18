"""Chunking configuration comparison experiment.

Compares three chunking configurations on the current sample:
  - 256 words / 32 overlap (baseline)
  - 512 words / 64 overlap
  - 1024 words / 128 overlap

Measures:
  - Number of chunks produced
  - Average chunk size (words, chars)
  - Retrieval quality (recall@5, hit@5, MRR)
  - Retrieval latency

Results are saved to ``experiments/chunking_results.json``.

Usage
-----
    .venv\\Scripts\\python.exe experiments\\chunking_comparison.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.chunking import chunk_passage, validate_params
from src.evaluate import (
    evaluate_query,
    load_chunks_as_dataset,
    save_results,
)

# Chunking configs to compare.
CONFIGS = [
    {"chunk_size_words": 256, "overlap_words": 32},
    {"chunk_size_words": 512, "overlap_words": 64},
    {"chunk_size_words": 1024, "overlap_words": 128},
]

INPUT_PATH = os.path.join("data", "cleaned", "preprocessed.jsonl")
CHUNKS_DIR = os.path.join("data", "chunks")
RESULTS_DIR = "experiments"


def load_cleaned_records(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def chunk_records(records: list, chunk_size: int, overlap: int) -> list:
    """Run chunking on the translated passages and return chunk dicts."""
    chunks = []
    for record in records:
        translated = record.get("passages", {}).get("translated") or []
        for pidx, passage in enumerate(translated):
            raw_text = passage.get("text")
            if not isinstance(raw_text, str) or not raw_text.strip():
                continue
            # Normalize
            from src.chunking import normalize_text
            normalized = normalize_text(raw_text)
            passage_chunks = chunk_passage(normalized, chunk_size, overlap)
            for cidx, chunk_text in enumerate(passage_chunks):
                chunks.append({
                    "chunk_id": f"{record['query_id']}_p{pidx}_c{cidx}",
                    "query_id": record.get("query_id"),
                    "query": record.get("query"),
                    "query_type": record.get("query_type"),
                    "source_lang": record.get("source_lang"),
                    "target_lang": record.get("target_lang"),
                    "passage_index": pidx,
                    "is_selected": bool(passage.get("is_selected")),
                    "chunk_index": cidx,
                    "num_words": len(chunk_text.split()),
                    "chunk_text": chunk_text,
                })
    return chunks


def compute_stats(chunks: list) -> dict:
    """Compute basic chunk statistics."""
    if not chunks:
        return {"count": 0, "avg_words": 0, "avg_chars": 0, "selected_pct": 0}
    words = [c["num_words"] for c in chunks]
    chars = [len(c["chunk_text"]) for c in chunks]
    selected = sum(1 for c in chunks if c.get("is_selected", False))
    return {
        "count": len(chunks),
        "avg_words": round(sum(words) / len(words), 1),
        "avg_chars": round(sum(chars) / len(chars), 1),
        "selected_pct": round(selected / len(chunks) * 100, 1),
    }


def evaluate_retrieval(chunks: list, ks=(1, 3, 5, 10)) -> dict:
    """Evaluate retrieval using a simple BM25 oracle (no model download)."""
    from src.bm25 import BM25Retriever

    cfg = Config()
    bm25 = BM25Retriever(cfg)
    bm25.build_index(chunks)

    # Group chunks by query_id for evaluation.
    grouped = {}
    for c in chunks:
        qid = str(c.get("query_id", ""))
        if qid:
            grouped.setdefault(qid, []).append(c)

    results = {}
    for k in ks:
        recalls = []
        hits = []
        mrrs = []
        for qid, qchunks in grouped.items():
            relevant = [c["chunk_id"] for c in qchunks if c.get("is_selected")]
            if not relevant:
                continue
            query_text = qchunks[0].get("query", "")
            if not query_text:
                continue

            bm25_results = bm25.search(query_text, top_k=k)
            retrieved_ids = [r["chunk_id"] for r in bm25_results]

            m = evaluate_query(relevant, retrieved_ids, k=k)
            recalls.append(m["recall_k"])
            hits.append(m["hit_k"])
            mrrs.append(m["mrr"])

        n = max(len(recalls), 1)
        results[f"recall@{k}"] = round(sum(recalls) / n, 4)
        results[f"hit@{k}"] = round(sum(hits) / n, 4)

    results["mrr"] = round(
        sum(mrrs) / max(len(mrrs), 1), 4
    ) if mrrs else 0.0
    return results


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: cleaned data not found at {INPUT_PATH}")
        print("Run preprocessing first: python src/preprocess.py")
        return 1

    records = load_cleaned_records(INPUT_PATH)
    print(f"Loaded {len(records)} cleaned records")

    all_results = []

    for cfg in CONFIGS:
        cs = cfg["chunk_size_words"]
        ov = cfg["overlap_words"]
        print(f"\n{'='*60}")
        print(f"Config: {cs} words / {ov} overlap")
        print(f"{'='*60}")

        # 1. Chunk
        t0 = time.perf_counter()
        chunks = chunk_records(records, cs, ov)
        chunk_time = time.perf_counter() - t0

        # 2. Stats
        stats = compute_stats(chunks)
        print(f"  Chunks: {stats['count']}")
        print(f"  Avg words: {stats['avg_words']}")
        print(f"  Avg chars: {stats['avg_chars']}")
        print(f"  Chunking time: {chunk_time:.2f}s")

        # 3. Evaluate retrieval (BM25 only, no model download needed)
        t1 = time.perf_counter()
        retrieval = evaluate_retrieval(chunks)
        eval_time = time.perf_counter() - t1
        print(f"  Retrieval quality: {json.dumps(retrieval, indent=4)}")
        print(f"  Evaluation time: {eval_time:.2f}s")

        all_results.append({
            "config": cfg,
            "stats": stats,
            "retrieval": retrieval,
            "chunking_time_s": round(chunk_time, 2),
            "evaluation_time_s": round(eval_time, 2),
        })

    # Save results.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "chunking_comparison.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
