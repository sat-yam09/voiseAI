"""End-to-end smoke test of the full Member 2 retrieval pipeline.

Builds indices from existing local chunks, then runs the complete pipeline:
    query → embedding → vector retrieval + BM25 → RRF fusion → reranking → Top-K

Usage:
    python tests/smoke_e2e.py
"""

import json
import os
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.retriever import VectorRetriever
from src.bm25 import BM25Retriever
from src.hybrid_search import HybridSearch
from src.reranker import Reranker


def build_indices(cfg: Config) -> None:
    """Build vector + BM25 indices if they don't already exist."""
    vec_idx = os.path.join(cfg.index_dir, cfg.vector_index_name)
    meta_idx = os.path.join(cfg.index_dir, cfg.metadata_name)
    bm25_idx = os.path.join(cfg.index_dir, "bm25_index.json")

    if os.path.exists(vec_idx) and os.path.exists(meta_idx) and os.path.exists(bm25_idx):
        print("[SKIP] All indices already exist — loading from disk.")
        return

    print("[BUILD] Loading chunks ...")
    chunks = []
    with open(cfg.chunks_path(), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"       Loaded {len(chunks)} chunks.")

    if not (os.path.exists(vec_idx) and os.path.exists(meta_idx)):
        print("[BUILD] Building FAISS vector index (embedding all chunks) ...")
        vr = VectorRetriever(cfg)
        t0 = time.perf_counter()
        n = vr.build_index(chunks)
        elapsed = time.perf_counter() - t0
        print(f"       Vector index: {n} vectors built in {elapsed:.1f}s")
    else:
        print("[SKIP] Vector index already exists.")

    if not os.path.exists(bm25_idx):
        print("[BUILD] Building BM25 index ...")
        bm = BM25Retriever(cfg)
        t0 = time.perf_counter()
        bm.build_index(chunks)
        bm.save()
        elapsed = time.perf_counter() - t0
        print(f"       BM25 index: {bm.size} docs built in {elapsed:.1f}s")
    else:
        print("[SKIP] BM25 index already exists.")


def get_real_query(cfg: Config) -> str:
    """Read the first query from preprocessed.jsonl."""
    with open(cfg.cleaned_jsonl, "r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
    record = json.loads(first_line)
    return record["query"]


def run_pipeline(cfg: Config, query: str, top_k: int = 5) -> None:
    """Run the full hybrid retrieval + reranking pipeline and print results."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"TOP-K: {top_k}")
    print(f"CONFIG: vector_top_k={cfg.vector_top_k}, bm25_top_k={cfg.bm25_top_k}, "
          f"hybrid_top_k={cfg.hybrid_top_k}, rerank_top_n={cfg.rerank_top_n}, "
          f"rrf_k={cfg.rrf_k}")
    print(f"{'='*80}\n")

    # --- Stage 1: Build hybrid search (loads both indices) ---
    print("[LOAD] Loading vector + BM25 indices into HybridSearch ...")
    t_load = time.perf_counter()
    hybrid = HybridSearch(cfg)
    hybrid.load_index()
    print(f"       Indices loaded in {time.perf_counter() - t_load:.2f}s\n")

    # --- Stage 2: Hybrid search (vector + BM25 + RRF) ---
    print("[STEP 1] Hybrid search (vector + BM25 + RRF) ...")
    t0 = time.perf_counter()
    candidates = hybrid.search(query, top_k=cfg.rerank_top_n)
    hybrid_ms = (time.perf_counter() - t0) * 1000
    print(f"         {len(candidates)} candidates in {hybrid_ms:.1f}ms\n")

    # --- Stage 3: Reranking ---
    print("[STEP 2] Cross-encoder reranking ...")
    reranker = Reranker(cfg)
    t1 = time.perf_counter()
    reranked = reranker.rerank(query, candidates, top_n=cfg.rerank_top_n)
    rerank_ms = (time.perf_counter() - t1) * 1000
    print(f"         Reranked {len(reranked)} candidates in {rerank_ms:.1f}ms\n")

    # --- Stage 4: Final Top-K ---
    total_ms = hybrid_ms + rerank_ms
    final = reranked[:top_k]

    print(f"{'='*80}")
    print(f"FINAL TOP-{top_k} RESULTS  (total latency: {total_ms:.1f}ms)")
    print(f"  Hybrid search: {hybrid_ms:.1f}ms")
    print(f"  Reranking:     {rerank_ms:.1f}ms")
    print(f"{'='*80}\n")

    for item in final:
        rank = item["rank"]
        score = item.get("rerank_score", item.get("rrf_score", 0.0))
        chunk_id = item.get("chunk_id", "?")
        text = item.get("chunk_text", "")
        preview = text[:120].replace("\n", " ")
        src = item.get("source_lang", "?")
        tgt = item.get("target_lang", "?")
        vrank = item.get("vector_rank", "-")
        brank = item.get("bm25_rank", "-")
        print(f"  [{rank}] score={score:.4f}  chunk_id={chunk_id}")
        print(f"       lang: {src} → {tgt}  |  vec_rank={vrank}  bm25_rank={brank}")
        print(f"       text: {preview}...")
        print()


def main() -> int:
    cfg = Config()
    top_k = 5

    # Step 0: Build indices if needed
    build_indices(cfg)

    # Step 1: Get a real query
    query = get_real_query(cfg)

    # Step 2: Run the full pipeline
    run_pipeline(cfg, query, top_k=top_k)

    print("\n[DONE] Smoke test complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
