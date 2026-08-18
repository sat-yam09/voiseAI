"""Quick comparison: old model (already indexed) vs new E5-base on Assamese queries."""
import json, os, sys, time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.config import Config
from src.embeddings import Embedder
from src.vector_store import VectorStore

NUM_QUERIES = 8

def compute_metrics(relevant_ids, retrieved_ids, k=5):
    rel_set = set(relevant_ids)
    ret_k = retrieved_ids[:k]
    found = sum(1 for cid in ret_k if cid in rel_set)
    recall = found / len(rel_set) if rel_set else 0.0
    hit = 1.0 if found > 0 else 0.0
    return recall, hit, found

def main():
    cfg = Config()
    chunks_path = cfg.chunks_path()

    gt = {}
    with open(chunks_path, "r", encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line.strip())
            qid = str(c.get("query_id", ""))
            gt.setdefault(qid, []).append(c)
    eval_pool = {qid: cs for qid, cs in gt.items()
                 if any(x.get("is_selected") for x in cs)}
    sample_qids = list(eval_pool.keys())[:NUM_QUERIES]

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as fh:
        for line in fh:
            chunks.append(json.loads(line.strip()))
    print("Chunks: {}".format(len(chunks)))

    # ---- Old model evaluation (index already exists) ----
    print("\n" + "=" * 72)
    print("OLD MODEL: paraphrase-multilingual-MiniLM-L12-v2 (no prefix)")
    print("=" * 72)

    old_cfg = Config()
    old_cfg.embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    old_cfg.embedding_dim = 384
    old_cfg.embedding_query_prefix = ""
    old_cfg.embedding_passage_prefix = ""
    old_cfg.index_dir = cfg.index_dir

    old_store = VectorStore(old_cfg)
    old_store.load()
    old_embedder = Embedder(old_cfg)

    old_recalls, old_hits, old_lats = [], [], []
    for qi, qid in enumerate(sample_qids, 1):
        cs = eval_pool[qid]
        query = cs[0]["query"]
        rel_ids = [c["chunk_id"] for c in cs if c.get("is_selected")]
        t0 = time.perf_counter()
        q_vec = old_embedder.encode_query(query)
        results = old_store.search(q_vec, top_k=5)
        ms = (time.perf_counter() - t0) * 1000
        old_lats.append(ms)
        ret_ids = [r["chunk_id"] for r in results]
        r, h, f = compute_metrics(rel_ids, ret_ids)
        old_recalls.append(r)
        old_hits.append(h)
        mark = "OK" if f > 0 else "MISS"
        print("  Q{}: Recall@5={:.2f} Hit@5={:.0f} found={}/{} [{}ms] {}".format(
            qi, r, h, f, len(rel_ids), int(ms), mark))
        print("       {}".format(query[:70]))

    old_mr = sum(old_recalls) / len(old_recalls)
    old_mh = sum(old_hits) / len(old_hits)
    old_ml = sum(old_lats) / len(old_lats)

    # ---- New E5-base model ----
    print("\n" + "=" * 72)
    print("NEW MODEL: intfloat/multilingual-e5-base (query:/passage: prefix)")
    print("=" * 72)

    new_cfg = Config()
    new_cfg.embedding_model = "intfloat/multilingual-e5-base"
    new_cfg.embedding_dim = 1024
    new_cfg.embedding_query_prefix = "query: "
    new_cfg.embedding_passage_prefix = "passage: "
    new_cfg.index_dir = "data/index/e5_base"

    new_embedder = Embedder(new_cfg)

    print("  Encoding all passages ...")
    t0 = time.perf_counter()
    all_vecs = new_embedder.encode_chunks(chunks, show_progress=True)
    encode_s = time.perf_counter() - t0
    print("  Encoded {} vectors (dim={}) in {:.1f}s".format(
        all_vecs.shape[0], all_vecs.shape[1], encode_s))

    os.makedirs("data/index/e5_base", exist_ok=True)
    metadata = [{
        "chunk_id": c["chunk_id"], "chunk_text": c["chunk_text"],
        "query_id": c.get("query_id"), "query": c.get("query"),
        "is_selected": c.get("is_selected", False),
    } for c in chunks]
    store = VectorStore(new_cfg)
    store.build(all_vecs, metadata)
    store.save()
    print("  Index saved: {} vectors\n".format(store.size))

    new_recalls, new_hits, new_lats = [], [], []
    for qi, qid in enumerate(sample_qids, 1):
        cs = eval_pool[qid]
        query = cs[0]["query"]
        rel_ids = [c["chunk_id"] for c in cs if c.get("is_selected")]
        t0 = time.perf_counter()
        q_vec = new_embedder.encode_query(query)
        results = store.search(q_vec, top_k=5)
        ms = (time.perf_counter() - t0) * 1000
        new_lats.append(ms)
        ret_ids = [r["chunk_id"] for r in results]
        r, h, f = compute_metrics(rel_ids, ret_ids)
        new_recalls.append(r)
        new_hits.append(h)
        mark = "OK" if f > 0 else "MISS"
        print("  Q{}: Recall@5={:.2f} Hit@5={:.0f} found={}/{} [{}ms] {}".format(
            qi, r, h, f, len(rel_ids), int(ms), mark))
        print("       {}".format(query[:70]))
        if f > 0:
            rel_set = set(rel_ids)
            for rr in results[:5]:
                m = "***" if rr["chunk_id"] in rel_set else "   "
                print("      [{}] {} score={:.4f} {}".format(
                    rr["rank"], m, rr["score"], rr["chunk_text"][:90]))

    new_mr = sum(new_recalls) / len(new_recalls)
    new_mh = sum(new_hits) / len(new_hits)
    new_ml = sum(new_lats) / len(new_lats)

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("  {:<52} {:>10} {:>8} {:>10}".format("Model", "Recall@5", "Hit@5", "Latency"))
    print("  " + "-" * 80)
    print("  {:<52} {:>10.4f} {:>8.4f} {:>8.0f}ms".format(
        "OLD: paraphrase-multilingual-MiniLM-L12-v2", old_mr, old_mh, old_ml))
    print("  {:<52} {:>10.4f} {:>8.4f} {:>8.0f}ms".format(
        "NEW: intfloat/multilingual-e5-base", new_mr, new_mh, new_ml))
    print()

    if new_mr > old_mr:
        print("  NEW MODEL IMPROVED: Recall@5 +{:.4f}".format(new_mr - old_mr))
    elif new_mr < old_mr:
        print("  NEW MODEL WORSE: Recall@5 -{:.4f}".format(old_mr - new_mr))
    else:
        print("  NO CHANGE in Recall@5")

    should_be_baseline = new_mr >= old_mr and new_mh >= old_mh
    print("  Should become baseline? {}".format("YES" if should_be_baseline else "NO"))

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

if __name__ == "__main__":
    main()
