"""Diagnostic: why Assamese retrieval is failing.

Separately measures vector-only, BM25-only, and hybrid retrieval
on a sample of Assamese queries with known ground truth.

No production code is modified.

Usage:
    python experiments/diagnose_retrieval.py
"""

import json
import os
import re
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
from src.retriever import VectorRetriever
from src.bm25 import BM25Retriever, tokenize
from src.hybrid_search import HybridSearch


NUM_QUERIES = 8


def load_ground_truth(chunks_path: str) -> Dict[str, List[Dict]]:
    grouped = {}
    with open(chunks_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            qid = str(c.get("query_id", ""))
            grouped.setdefault(qid, []).append(c)
    return grouped


def compute_metrics(relevant_ids: List[str], retrieved_ids: List[str], k: int = 5):
    rel_set = set(relevant_ids)
    ret_k = retrieved_ids[:k]
    found = sum(1 for cid in ret_k if cid in rel_set)
    recall = found / len(rel_set) if rel_set else 0.0
    hit = 1.0 if any(cid in rel_set for cid in ret_k) else 0.0
    return recall, hit, found


def main():
    cfg = Config()
    chunks_path = cfg.chunks_path()

    # ---------------------------------------------------------------
    # 1. Load ground truth and select Assamese queries
    # ---------------------------------------------------------------
    print("=" * 72)
    print("RETRIEVAL DIAGNOSTIC: Assamese Query Analysis")
    print("=" * 72)

    gt = load_ground_truth(chunks_path)
    eval_pool = {qid: cs for qid, cs in gt.items()
                 if any(c.get("is_selected") for c in cs)}

    # Pick first N queries with ground truth
    sample_qids = list(eval_pool.keys())[:NUM_QUERIES]

    # ---------------------------------------------------------------
    # 2. Load indices
    # ---------------------------------------------------------------
    print("\n[INIT] Loading vector index ...")
    vr = VectorRetriever(cfg)
    t0 = time.perf_counter()
    vr.load_index()
    print(f"       Loaded in {(time.perf_counter()-t0)*1000:.0f}ms  "
          f"({vr.store.size} vectors)")

    print("[INIT] Loading BM25 index ...")
    bm = BM25Retriever(cfg)
    t0 = time.perf_counter()
    bm.load()
    print(f"       Loaded in {(time.perf_counter()-t0)*1000:.0f}ms  "
          f"({bm.size} docs)")

    print("[INIT] Loading hybrid search ...")
    hs = HybridSearch(cfg)
    hs.load_index()
    print("       Ready.\n")

    # ---------------------------------------------------------------
    # 3. BM25 tokenizer analysis on Assamese text
    # ---------------------------------------------------------------
    print("=" * 72)
    print("BM25 TOKENIZER ANALYSIS ON ASSAMESE TEXT")
    print("=" * 72)

    _TOKEN_RE = re.compile(r"\w+", re.UNICODE)

    for qi, qid in enumerate(sample_qids[:3], 1):
        chunks = eval_pool[qid]
        query_text = chunks[0]["query"]
        tokens = _TOKEN_RE.findall(query_text.lower())
        print(f"\n  Query {qi}: {query_text}")
        print(f"  Tokens ({len(tokens)}): {tokens}")

        # Also tokenize a relevant passage
        relevant = [c for c in chunks if c.get("is_selected")]
        if relevant:
            ptext = relevant[0]["chunk_text"][:200]
            ptokens = _TOKEN_RE.findall(ptext.lower())
            print(f"  Relevant passage tokens ({len(ptokens)}): {ptokens[:15]}...")

        # Check BM25 scores for this query
        bm_results = bm.search(query_text, top_k=10)
        bm_retrieved_ids = [r["chunk_id"] for r in bm_results]
        relevant_ids = [c["chunk_id"] for c in chunks if c.get("is_selected")]
        bm_recall, bm_hit, bm_found = compute_metrics(relevant_ids, bm_retrieved_ids)
        print(f"  BM25 top-5 IDs: {bm_retrieved_ids[:5]}")
        print(f"  Relevant IDs:   {relevant_ids}")
        print(f"  BM25 found: {bm_found}/{len(relevant_ids)}  "
              f"Recall@5={bm_recall:.2f}  Hit@5={bm_hit:.0f}")

    # ---------------------------------------------------------------
    # 4. Embedding model inspection
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("EMBEDDING MODEL INSPECTION")
    print("=" * 72)
    print(f"  Model name:           {cfg.embedding_model}")
    print(f"  Embedding dimension:  {cfg.embedding_dim}")
    print(f"  Normalize:            {cfg.normalize_embeddings}")
    print(f"  Batch size:           {cfg.embed_batch_size}")
    print(f"  Same model for query + passage: YES (single SentenceTransformer)")

    # ---------------------------------------------------------------
    # 5. Per-method retrieval diagnostic
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PER-METHOD RETRIEVAL DIAGNOSTIC")
    print("=" * 72)

    vec_recalls, bm25_recalls, hyb_recalls = [], [], []
    vec_hits, bm25_hits, hyb_hits = [], [], []

    for qi, qid in enumerate(sample_qids, 1):
        chunks = eval_pool[qid]
        query_text = chunks[0]["query"]
        query_type = chunks[0].get("query_type", "?")
        src_lang = chunks[0].get("source_lang", "?")
        tgt_lang = chunks[0].get("target_lang", "?")
        relevant_ids = [c["chunk_id"] for c in chunks if c.get("is_selected")]
        relevant_texts = {c["chunk_id"]: c["chunk_text"][:100]
                          for c in chunks if c.get("is_selected")}

        print(f"\n{'─'*72}")
        print(f"Query {qi}/{NUM_QUERIES}: {query_text}")
        print(f"  query_id={qid}  type={query_type}  "
              f"lang={src_lang}→{tgt_lang}")
        print(f"  Relevant chunk_ids: {relevant_ids}")

        # --- Vector only ---
        t0 = time.perf_counter()
        vec_results = vr.search(query_text, top_k=10)
        vec_ms = (time.perf_counter() - t0) * 1000
        vec_ids = [r["chunk_id"] for r in vec_results]
        v_r, v_h, v_f = compute_metrics(relevant_ids, vec_ids)

        print(f"\n  [VECTOR] ({vec_ms:.0f}ms)")
        print(f"    Recall@5={v_r:.2f}  Hit@5={v_h:.0f}  found={v_f}/{len(relevant_ids)}")
        for r in vec_results[:5]:
            mark = " ***" if r["chunk_id"] in set(relevant_ids) else ""
            text_preview = r.get("chunk_text", "")[:80].replace("\n", " ")
            print(f"    [{r['rank']}] score={r['score']:.4f}  "
                  f"id={r['chunk_id']}{mark}")
            print(f"        {text_preview}...")

        # --- BM25 only ---
        t0 = time.perf_counter()
        bm_results = bm.search(query_text, top_k=10)
        bm_ms = (time.perf_counter() - t0) * 1000
        bm_ids = [r["chunk_id"] for r in bm_results]
        b_r, b_h, b_f = compute_metrics(relevant_ids, bm_ids)

        print(f"\n  [BM25] ({bm_ms:.0f}ms)")
        print(f"    Recall@5={b_r:.2f}  Hit@5={b_h:.0f}  found={b_f}/{len(relevant_ids)}")
        if not bm_results:
            print(f"    (no results returned)")
        for r in bm_results[:5]:
            mark = " ***" if r["chunk_id"] in set(relevant_ids) else ""
            text_preview = r.get("chunk_text", "")[:80].replace("\n", " ")
            print(f"    [{r['rank']}] score={r['score']:.4f}  "
                  f"id={r['chunk_id']}{mark}")
            print(f"        {text_preview}...")

        # --- Hybrid (RRF) ---
        t0 = time.perf_counter()
        hyb_results = hs.search(query_text, top_k=10)
        hyb_ms = (time.perf_counter() - t0) * 1000
        hyb_ids = [r["chunk_id"] for r in hyb_results]
        h_r, h_h, h_f = compute_metrics(relevant_ids, hyb_ids)

        print(f"\n  [HYBRID/RRF] ({hyb_ms:.0f}ms)")
        print(f"    Recall@5={h_r:.2f}  Hit@5={h_h:.0f}  found={h_f}/{len(relevant_ids)}")
        for r in hyb_results[:5]:
            mark = " ***" if r["chunk_id"] in set(relevant_ids) else ""
            text_preview = r.get("chunk_text", "")[:80].replace("\n", " ")
            vrk = r.get("vector_rank", "-")
            brk = r.get("bm25_rank", "-")
            print(f"    [{r['rank']}] rrf={r['rrf_score']:.4f}  "
                  f"id={r['chunk_id']}  vec_r={vrk} bm25_r={brk}{mark}")
            print(f"        {text_preview}...")

        vec_recalls.append(v_r)
        bm25_recalls.append(b_r)
        hyb_recalls.append(h_r)
        vec_hits.append(v_h)
        bm25_hits.append(b_h)
        hyb_hits.append(h_h)

    # ---------------------------------------------------------------
    # 6. Aggregate summary
    # ---------------------------------------------------------------
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print("\n" + "=" * 72)
    print("AGGREGATE SUMMARY")
    print("=" * 72)
    print(f"  Queries evaluated: {NUM_QUERIES}")
    print(f"\n  {'Method':<12} {'Recall@5':>10} {'Hit Rate@5':>12}")
    print(f"  {'-'*34}")
    print(f"  {'Vector':<12} {mean(vec_recalls):>10.4f} {mean(vec_hits):>12.4f}")
    print(f"  {'BM25':<12} {mean(bm25_recalls):>10.4f} {mean(bm25_hits):>12.4f}")
    print(f"  {'Hybrid':<12} {mean(hyb_recalls):>10.4f} {mean(hyb_hits):>12.4f}")

    # Check overlap between vector and BM25
    print(f"\n  Vector-BM25 overlap analysis:")
    for qi, qid in enumerate(sample_qids):
        chunks = eval_pool[qid]
        query_text = chunks[0]["query"]
        vec_r = vr.search(query_text, top_k=10)
        bm_r = bm.search(query_text, top_k=10)
        vec_set = set(r["chunk_id"] for r in vec_r[:5])
        bm_set = set(r["chunk_id"] for r in bm_r[:5])
        overlap = vec_set & bm_set
        print(f"    Q{qi+1}: vec_top5={len(vec_set)} bm25_top5={len(bm_set)} "
              f"overlap={len(overlap)} ids={overlap if overlap else 'none'}")

    print("\n" + "=" * 72)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
