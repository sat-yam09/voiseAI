"""Compare paraphrase-multilingual-MiniLM-L12-v2 vs multilingual-e5-small.

Uses ONLY the already-cached passage embeddings (no re-encoding).
Evaluates Hindi, English, Bengali, Gujarati separately.
Reports Recall@5, Hit Rate@5, MRR, embedding latency, retrieval latency,
and model/index sizes.  Skips cross-encoder reranker.

Usage:
    python experiments/compare_two_models.py
"""
import gc
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

import numpy as np
from src.config import Config, PROJECT_ROOT
from src.embeddings import Embedder, _loaded_models
from src.vector_store import VectorStore

# ───────────────────────────────────────────────────────────────────────
# Models (only the two with cached passages)
# ───────────────────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "name": "multilingual-e5-small",
        "model_id": "intfloat/multilingual-e5-small",
        "dim": 384,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
]

# ───────────────────────────────────────────────────────────────────────
# Test queries (50 total: 20 en, 10 bn, 10 hi, 10 gu)
# ───────────────────────────────────────────────────────────────────────
TEST_QUERIES = [
    # English (20)
    {"qid": "1102432", "lang": "en", "query": "what is a corporation?"},
    {"qid": "205107",  "lang": "en", "query": "honesty or integrity definition"},
    {"qid": "190327",  "lang": "en", "query": "foods that help with vitamin d"},
    {"qid": "1060361", "lang": "en", "query": "barter system and its problems"},
    {"qid": "373460",  "lang": "en", "query": "how to print an excel spreadsheet"},
    {"qid": "260880",  "lang": "en", "query": "how long does it take for a cantaloupe to ripen"},
    {"qid": "116898",  "lang": "en", "query": "definition of voluntary"},
    {"qid": "271597",  "lang": "en", "query": "how long to microwave a corn kernel"},
    {"qid": "1090352", "lang": "en", "query": "why starve"},
    {"qid": "21860",   "lang": "en", "query": "is corn food and corn flour the same"},
    {"qid": "116095",  "lang": "en", "query": "death penalty definition"},
    {"qid": "330463",  "lang": "en", "query": "how much weight does a stump cover hold"},
    {"qid": "44760",   "lang": "en", "query": "average temperature in caribbean in december"},
    {"qid": "331047",  "lang": "en", "query": "how much do yaky games cost"},
    {"qid": "267380",  "lang": "en", "query": "how many days can you run a car without oil"},
    {"qid": "113570",  "lang": "en", "query": "sociology definition of culture"},
    {"qid": "1060348", "lang": "en", "query": "what is elementary"},
    {"qid": "317450",  "lang": "en", "query": "how much does matt lauer make a year"},
    {"qid": "126172",  "lang": "en", "query": "definition of dinghy"},
    {"qid": "131336",  "lang": "en", "query": "definition of local disk"},

    # Bengali (10)
    {"qid": "1102432", "lang": "bn", "query": "\u0995\u09c0\u09b0\u09cd\u09aa\u09cb\u099f\u09c7\u099a\u09a8 \u0995\u09bf?"},
    {"qid": "205107",  "lang": "bn", "query": "\u09b8\u09a4\u09a4\u09be \u09ac\u09be \u09b8\u09a4\u09a4\u09be\u09b0 \u09b8\u0982\u099c\u09cd\u099e\u09be"},
    {"qid": "190327",  "lang": "bn", "query": "\u09ad\u09bf\u099f\u09be\u09ae\u09bf\u09a8 \u09a1\u09bf-\u0995\u0993 \u09b8\u09b9\u09be\u09af\u09bc \u0995\u09b0\u09be \u0996\u09be\u09a6\u09cd\u09af\u09b8\u09ae\u09c2\u09b9"},
    {"qid": "1060361", "lang": "bn", "query": "\u09ac\u09be\u09b0\u09cd\u099f\u09be\u09b0 \u09ac\u09cd\u09af\u09ac\u09b8\u09cd\u09a5\u09be \u0986\u09b0\u09c1 \u0987\u09af\u09bc\u09be\u09b0 \u09b8\u09ae\u09b8\u09cd\u09af\u09be"},
    {"qid": "373460",  "lang": "bn", "query": "\u098f\u0995\u099f\u09bf \u098f\u0995\u09cd\u09b8\u09c7\u09b2 \u09b6\u09cd\u09ac\u09c0\u099f \u0995\u09c7\u09a8\u09c7\u0995\u09c8 \u09aa\u09cd\u09b0\u09bf\u09a3\u09cd\u099f \u0995\u09b0\u09bf\u09ac"},
    {"qid": "116898",  "lang": "bn", "query": "\u09b8\u09cd\u09ac\u099a\u09cd\u099b\u09be\u09ae\u09c2\u09b2\u0995 \u09a8\u09bf\u09b0\u09cd\u09a7\u09be\u09b0\u0995\u09b0\u09a3 \u09b8\u0982\u099c\u09cd\u099e\u09be"},
    {"qid": "116095",  "lang": "bn", "query": "\u09ae\u09c3\u09a4\u09cd\u09af\u09c1\u09b0 \u09a6\u09a3\u09cd\u09a1\u09c0\u09af\u09bc\u09be\u09b0 \u09b8\u0982\u099c\u09cd\u099e\u09be"},
    {"qid": "267380",  "lang": "bn", "query": "\u0995\u09cd\u09a4\u09a4\u09a8 \u09a6\u09bf\u09a8 \u09a7\u09b0\u09bf \u0995\u09be\u09b0 \u099a\u09b2\u09be\u09ac \u09b2\u09be\u0997\u09c7"},
    {"qid": "1090352", "lang": "bn", "query": "\u0995\u09be\u09b9\u09bf\u09a8\u09c7 \u09b6\u09be\u09b8\u09cd\u09a4\u09be \u09b9\u09ac\u09c7"},
    {"qid": "44760",   "lang": "bn", "query": "\u09a1\u09bf\u09b8\u09c7\u09ae\u09cd\u09ac\u09b0\u09be\u09b0 \u0997\u09a1\u09bc\u09be \u09a4\u09be\u09aa\u09ae\u09be\u09a4\u09cd\u09b0\u09be"},

    # Hindi (10)
    {"qid": "1102432", "lang": "hi", "query": "\u0915\u093e\u0930\u094d\u092a\u094b\u0930\u0947\u0936\u0928 \u0915\u094d\u092f\u093e \u0939\u0948?"},
    {"qid": "205107",  "lang": "hi", "query": "\u0908\u092e\u093e\u0928\u0926\u093e\u0930\u094d\u0924\u093e \u092f\u093e \u0938\u094d\u091a\u094d\u0924\u093e \u0915\u0940 \u092a\u0930\u093f\u092d\u093e\u0937\u093e"},
    {"qid": "190327",  "lang": "hi", "query": "\u0935\u093f\u091f\u093e\u092e\u093f\u0928 \u0921\u0940 \u0915\u0947 \u0932\u093f\u090f \u092e\u0926\u0926 \u0915\u0940 \u0938\u0942\u091a\u0940"},
    {"qid": "1060361", "lang": "hi", "query": "\u092c\u093e\u0930\u094d\u091f\u0930 \u092a\u094d\u0930\u0923\u093e\u0932\u0940 \u0914\u0930 \u0907\u0938\u0915\u0940 \u0938\u092e\u0938\u094d\u092f\u093e\u090f\u0901"},
    {"qid": "373460",  "lang": "hi", "query": "\u090f\u0915\u094d\u0938\u0947\u0932 \u0936\u0940\u091f \u0915\u094b \u0915\u0948\u0938\u0947 \u092a\u094d\u0930\u093f\u0902\u091f \u0915\u0930\u0947\u0902"},
    {"qid": "116898",  "lang": "hi", "query": "\u0938\u094d\u0935\u0947\u091a\u094d\u091f\u093e\u092e\u0942\u0932\u0915 \u092a\u0930\u093f\u092d\u093e\u0937\u093e \u0915\u0940 \u092a\u0930\u093f\u092d\u093e\u0937\u093e"},
    {"qid": "116095",  "lang": "hi", "query": "\u092e\u0943\u0924\u094d\u092f\u0941 \u0926\u0902\u0921 \u0915\u0940 \u092a\u0930\u093f\u092d\u093e\u0937\u093e"},
    {"qid": "267380",  "lang": "hi", "query": "\u092c\u093f\u0928\u093e \u0924\u0947\u0932 \u0915\u093e\u0930 \u0915\u093f\u0924\u0928\u0947 \u091a\u0932\u093e\u0928\u093e \u0939\u0948"},
    {"qid": "1090352", "lang": "hi", "query": "\u0938\u094d\u091f\u093e\u0930 \u0915\u094d\u092f\u094b\u0902 \u0928\u0939\u0940\u0902 \u0915\u0930\u0928\u0947 \u091a\u093e\u0939\u093f\u090f"},
    {"qid": "44760",   "lang": "hi", "query": "\u0921\u093f\u0938\u0947\u0902\u092c\u0930 \u092e\u0947\u0902 \u0926\u0938\u094d\u0924\u093e\u0902\u091c \u0915\u0940 \u0938\u093e\u092e\u093e\u0928\u094d\u0925\u093e\u092f\u093e"},

    # Gujarati (10)
    {"qid": "1102432", "lang": "gu", "query": "\u0a15\u0a3e\u0ab0\u0acd\u0aaa\u0acb\u0ab0\u0ac7\u0b9a\u0aa8 \u0a36\u0ac1\u0ab3 \u0a1b\u0ac7?"},
    {"qid": "205107",  "lang": "gu", "query": "\u0a38\u0a1a\u0ac1\u0a24\u0abe \u0a05\u0a2b\u0a47\u0a35 \u0a38\u0a1a\u0ac1\u0a24\u0abe\u0a28\u0abe \u0a35\u0acd\u0a2f\u0a3e\u0a16\u0acd\u0a2f\u0abe\u0a28"},
    {"qid": "190327",  "lang": "gu", "query": "\u0ab5\u0abf\u0a1f\u0abe\u0aae\u0abf\u0aa8 \u0aa1\u0ac0 \u0aae\u0abe\u0a9f\u0ac7 \u0ab6\u0ac1\u0ab0\u0abe \u0a2e\u0ac2\u0ab8\u0abe\u0ab9\u0abe\u0a30\u0ac0 \u0a24\u0ab9\u0ac7"},
    {"qid": "1060361", "lang": "gu", "query": "\u0ab9\u0abe\u0ab0\u0acd\u0a9f\u0ab0 \u0aae\u0abe\u0ab0\u0acd\u0a9a\u0ab0 \u0a05\u0aa8\u0ac7 \u0a07\u0ab8\u0a28\u0abe \u0ab8\u0aae\u0ab8\u0acd\u0aaf\u0abe"},
    {"qid": "373460",  "lang": "gu", "query": "\u0a0f\u0a15\u0acd\u0ab8\u0ac7\u0ab2 \u0ab6\u0ac0\u0a9f \u0a15\u0ac7\u0ab5\u0abe \u0aaa\u0acd\u0ab0\u0abf\u0a02\u0a9f \u0a15\u0ab0\u0ab5\u0ac1"},
    {"qid": "116898",  "lang": "gu", "query": "\u0ab8\u0acd\u0ab5\u0ac7\u019b\u0ab9\u0acd\u0aae\u0ac2\u0ab2\u0a15 \u0aaa\u0ab0\u0abf\u0aad\u0abe\u0ab7\u0abe \u0aa1\u0abf\u0ab9\u0acd\u0aaf\u0ac7\u0ab6"},
    {"qid": "116095",  "lang": "gu", "query": "\u0aae\u0ac3\u0a24\u0acd\u0a24\u0ac1 \u0aa6\u0a23\u0acd\u0aa1 \u0aa8\u0abe \u0ab5\u0acd\u0a2f\u0abe\u0a16\u0acd\u0a2f\u0abe\u0a28"},
    {"qid": "267380",  "lang": "gu", "query": "\u0a24\u0ac7\u0ab2 \u0ac7\u0ab2\u0ab5\u0abe \u0a15\u0ac7\u0ab0\u0ab5\u0ac1 \u0a15\u0ac7\u0a1f\u0ab2\u0ac0 \u0ac9\u0ab5\u0abe\u0a28\u0ac0 \u0a38\u0a30 \u0a24\u0abe\u0ab2\u0ac7"},
    {"qid": "1090352", "lang": "gu", "query": "\u0ab6\u0abe \u0a15\u0ac7\u0aa8\u0ac7 \u0aa8\u0ab9\u0ac0\u0a02 \u0a15\u0ab0\u0ab5\u0ac1\u0a1a\u0ac0 \u0a10"},
    {"qid": "44760",   "lang": "gu", "query": "\u0a1f\u0a3e\u0a07\u0a2a \u0ac7\u0ab2 \u0a21\u0abf\u0ab8\u0ac7\u0aae\u0acd\u0aac\u0ab0 \u0aae\u0abe\u0a01 \u0ab8\u0ab0\u0abe\u0ab6\u0acd\u0a24\u0abe\u0ab0\u0abf\u0a2f\u0abe\u0aa8"},
]


def percentile(vals, p):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def compute_recall_hit_mrr(relevant_ids, retrieved_ids, k=5):
    rel_set = set(relevant_ids)
    ret_k = retrieved_ids[:k]
    found = sum(1 for cid in ret_k if cid in rel_set)
    recall = found / len(rel_set) if rel_set else 0.0
    hit = 1.0 if found > 0 else 0.0
    mrr = 0.0
    for rank, cid in enumerate(ret_k, start=1):
        if cid in rel_set:
            mrr = 1.0 / rank
            break
    return recall, hit, mrr


def _get_model_size_mb(model_id):
    from pathlib import Path
    cache_home = os.environ.get("HF_HOME", "")
    if not cache_home:
        home = Path.home()
        cache_home = str(home / ".cache" / "huggingface" / "hub")
    model_dir_name = "models--" + model_id.replace("/", "--")
    model_path = Path(cache_home) / model_dir_name
    if model_path.exists():
        total = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        return total / (1024 * 1024)
    return 0.0


def _get_dir_size_mb(dir_path):
    total = 0
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def _cached_passage_path(model_name):
    return os.path.join(
        PROJECT_ROOT, "data", "cache",
        "bench_" + model_name.replace("/", "_"),
        "passages.npy",
    )


def main():
    base_cfg = Config()
    chunks_path = base_cfg.chunks_path()

    # Load chunks for ground truth
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print("Loaded {} chunks from {}".format(len(chunks), chunks_path))

    # Build ground truth: qid -> list of relevant chunk_ids
    gt_by_qid = {}
    for c in chunks:
        qid = str(c.get("query_id", ""))
        if qid and c.get("is_selected", False):
            gt_by_qid.setdefault(qid, []).append(c["chunk_id"])

    # Filter queries with ground truth
    valid_queries = [q for q in TEST_QUERIES if q["qid"] in gt_by_qid]
    print("Test queries with ground truth: {} / {}".format(
        len(valid_queries), len(TEST_QUERIES)))

    by_lang = {}
    for q in valid_queries:
        by_lang.setdefault(q["lang"], []).append(q)
    for lang in sorted(by_lang):
        print("  {}: {} queries".format(lang, len(by_lang[lang])))
    print()

    metadata = [{
        "chunk_id": c["chunk_id"],
        "chunk_text": c["chunk_text"],
        "query_id": c.get("query_id"),
        "query": c.get("query"),
        "is_selected": c.get("is_selected", False),
    } for c in chunks]

    # Verify caches exist
    for model_cfg in MODELS:
        p = _cached_passage_path(model_cfg["name"])
        if not os.path.exists(p):
            print("ERROR: Cached passages not found for {}: {}".format(
                model_cfg["name"], p))
            sys.exit(1)
        vecs = np.load(p)
        print("  Cache OK: {} -> shape={}".format(model_cfg["name"], vecs.shape))
    print()

    results_all = {}

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        print("=" * 72)
        print("MODEL: {}".format(model_name))
        print("=" * 72)
        print("  model_id: {}".format(model_cfg["model_id"]))
        print("  dim: {}  |  query_prefix: {}  |  passage_prefix: {}".format(
            model_cfg["dim"],
            repr(model_cfg["query_prefix"]),
            repr(model_cfg["passage_prefix"])))

        cfg = Config()
        cfg.embedding_model = model_cfg["model_id"]
        cfg.embedding_dim = model_cfg["dim"]
        cfg.embedding_query_prefix = model_cfg["query_prefix"]
        cfg.embedding_passage_prefix = model_cfg["passage_prefix"]
        cfg.embed_batch_size = 64
        cfg.index_dir = os.path.join(
            PROJECT_ROOT, "data", "index",
            "bench_{}".format(model_name.replace("/", "_")))
        os.makedirs(cfg.index_dir, exist_ok=True)

        # Load cached passages (no encoding)
        vecs = np.load(_cached_passage_path(model_name))
        print("  Loaded cached passages: shape={}".format(vecs.shape))

        # Load model for query encoding
        _loaded_models.clear()
        gc.collect()
        embedder = Embedder(cfg)

        # Build FAISS index from cached vectors
        print("  Building FAISS index ...")
        store = VectorStore(cfg)
        t0 = time.perf_counter()
        store.build(vecs, metadata)
        build_ms = (time.perf_counter() - t0) * 1000
        store.save()
        print("  {} vectors  build={:.0f}ms".format(store.size, build_ms))

        # Sizes
        model_size_mb = _get_model_size_mb(model_cfg["model_id"])
        index_size_mb = _get_dir_size_mb(cfg.index_dir)
        print("  Model size (HF cache): {:.1f} MB".format(model_size_mb))
        print("  Index size on disk:    {:.1f} MB".format(index_size_mb))

        # Query evaluation
        print("  Evaluating queries ...")
        lang_results = {}
        all_embed_lats = []
        all_search_lats = []
        all_total_lats = []

        for lang in ["en", "bn", "hi", "gu"]:
            qlist = by_lang.get(lang, [])
            if not qlist:
                continue

            recalls, hits, mrrs = [], [], []
            embed_lats, search_lats, total_lats = [], [], []

            for q in qlist:
                qid = q["qid"]
                query_text = q["query"]
                relevant_ids = gt_by_qid.get(qid, [])

                t0 = time.perf_counter()
                q_vec = embedder.encode_query(query_text)
                embed_ms = (time.perf_counter() - t0) * 1000

                t1 = time.perf_counter()
                results = store.search(q_vec, top_k=5)
                search_ms = (time.perf_counter() - t1) * 1000

                total_ms = embed_ms + search_ms
                ret_ids = [r["chunk_id"] for r in results]
                r, h, m = compute_recall_hit_mrr(relevant_ids, ret_ids, k=5)

                recalls.append(r)
                hits.append(h)
                mrrs.append(m)
                embed_lats.append(embed_ms)
                search_lats.append(search_ms)
                total_lats.append(total_ms)
                all_embed_lats.append(embed_ms)
                all_search_lats.append(search_ms)
                all_total_lats.append(total_ms)

            mean_r = sum(recalls) / len(recalls) if recalls else 0.0
            mean_h = sum(hits) / len(hits) if hits else 0.0
            mean_m = sum(mrrs) / len(mrrs) if mrrs else 0.0
            p50_t = percentile(total_lats, 50)
            p95_t = percentile(total_lats, 95)

            lang_results[lang] = {
                "recall@5": mean_r,
                "hit_rate@5": mean_h,
                "mrr": mean_m,
                "embed_p50": percentile(embed_lats, 50),
                "search_p50": percentile(search_lats, 50),
                "total_p50": p50_t,
                "total_p95": p95_t,
                "count": len(qlist),
            }
            print("    {:>4s}: Recall@5={:.4f}  Hit@5={:.4f}  MRR={:.4f}  "
                  "embed_p50={:.1f}ms  search_p50={:.1f}ms  "
                  "total_p50={:.1f}ms  p95={:.1f}ms  (n={})".format(
                      lang, mean_r, mean_h, mean_m,
                      percentile(embed_lats, 50),
                      percentile(search_lats, 50),
                      p50_t, p95_t, len(qlist)))

        # Aggregate
        total_n = sum(r["count"] for r in lang_results.values())
        agg_recall = sum(r["recall@5"] * r["count"] for r in lang_results.values()) / max(total_n, 1)
        agg_hit = sum(r["hit_rate@5"] * r["count"] for r in lang_results.values()) / max(total_n, 1)
        agg_mrr = sum(r["mrr"] * r["count"] for r in lang_results.values()) / max(total_n, 1)
        agg_total_p50 = percentile(all_total_lats, 50)
        agg_total_p95 = percentile(all_total_lats, 95)
        agg_embed_p50 = percentile(all_embed_lats, 50)
        agg_search_p50 = percentile(all_search_lats, 50)
        target_met = agg_total_p50 <= 200

        print("\n  AGGREGATE: Recall@5={:.4f}  Hit@5={:.4f}  MRR={:.4f}".format(
            agg_recall, agg_hit, agg_mrr))
        print("  Total p50={:.1f}ms  p95={:.1f}ms  <=200ms? {}".format(
            agg_total_p50, agg_total_p95, "YES" if target_met else "NO"))

        results_all[model_name] = {
            "model_id": model_cfg["model_id"],
            "dim": model_cfg["dim"],
            "model_size_mb": round(model_size_mb, 1),
            "index_size_mb": round(index_size_mb, 1),
            "per_lang": lang_results,
            "aggregate": {
                "recall@5": round(agg_recall, 4),
                "hit_rate@5": round(agg_hit, 4),
                "mrr": round(agg_mrr, 4),
                "embed_p50_ms": round(agg_embed_p50, 1),
                "search_p50_ms": round(agg_search_p50, 1),
                "total_p50_ms": round(agg_total_p50, 1),
                "total_p95_ms": round(agg_total_p95, 1),
                "under_200ms": target_met,
            },
        }

        # Cleanup FAISS index (keep passage cache)
        if os.path.exists(cfg.index_dir):
            import shutil
            shutil.rmtree(cfg.index_dir, ignore_errors=True)

        print("\n")

    # ── COMPARISON TABLE ──
    print("=" * 72)
    print("FINAL COMPARISON (cached passages only, no e5-base)")
    print("=" * 72)
    print()
    print("{:<42s} {:>8s} {:>8s} {:>6s} {:>8s} {:>10s} {:>8s}".format(
        "Model", "Recall@5", "Hit@5", "MRR", "Emb p50", "Total p50", "<=200ms?"))
    print("  " + "-" * 98)
    for name, res in results_all.items():
        a = res["aggregate"]
        print("{:<42s} {:>8.4f} {:>8.4f} {:>6.4f} {:>5.0f}ms {:>8.0f}ms {:>8s}".format(
            name, a["recall@5"], a["hit_rate@5"], a["mrr"],
            a["embed_p50_ms"], a["total_p50_ms"],
            "YES" if a["under_200ms"] else "NO"))
    print()

    # Per-language breakdown
    print("Per-language Recall@5 breakdown:")
    lang_header = "{:<42s}".format("Model")
    for lang in ["en", "bn", "hi", "gu"]:
        lang_header += " {:>8s}".format(lang)
    print(lang_header)
    print("  " + "-" * 80)
    for name, res in results_all.items():
        row = "{:<42s}".format(name)
        for lang in ["en", "bn", "hi", "gu"]:
            lr = res["per_lang"].get(lang, {})
            r5 = lr.get("recall@5", float("nan"))
            row += " {:>8.4f}".format(r5)
        print(row)
    print()

    print("Per-language MRR breakdown:")
    lang_header = "{:<42s}".format("Model")
    for lang in ["en", "bn", "hi", "gu"]:
        lang_header += " {:>8s}".format(lang)
    print(lang_header)
    print("  " + "-" * 80)
    for name, res in results_all.items():
        row = "{:<42s}".format(name)
        for lang in ["en", "bn", "hi", "gu"]:
            lr = res["per_lang"].get(lang, {})
            m = lr.get("mrr", float("nan"))
            row += " {:>8.4f}".format(m)
        print(row)
    print()

    # Latency breakdown
    print("Per-language total latency (p50) breakdown:")
    lang_header = "{:<42s}".format("Model")
    for lang in ["en", "bn", "hi", "gu"]:
        lang_header += " {:>10s}".format(lang)
    print(lang_header)
    print("  " + "-" * 88)
    for name, res in results_all.items():
        row = "{:<42s}".format(name)
        for lang in ["en", "bn", "hi", "gu"]:
            lr = res["per_lang"].get(lang, {})
            t = lr.get("total_p50", float("nan"))
            row += " {:>7.0f}ms".format(t)
        print(row)
    print()

    # Model sizes
    print("Model / Index sizes:")
    print("  {:<42s} {:>8s} {:>10s} {:>10s}".format("Model", "Dim", "Model MB", "Index MB"))
    print("  " + "-" * 74)
    for name, res in results_all.items():
        print("  {:<42s} {:>8d} {:>8.1f} {:>10.1f}".format(
            name, res["dim"], res["model_size_mb"], res["index_size_mb"]))
    print()

    # Recommendation
    print("=" * 72)
    print("RECOMMENDATION")
    print("=" * 72)

    # Rank by quality (Recall@5) among models meeting <=200ms
    under_200 = {n: r for n, r in results_all.items() if r["aggregate"]["under_200ms"]}
    if under_200:
        best_name = max(under_200.items(), key=lambda kv: kv[1]["aggregate"]["recall@5"])[0]
        best_res = results_all[best_name]
        print("  Best under 200ms target: {}".format(best_name))
        print("    Recall@5={:.4f}  Hit@5={:.4f}  MRR={:.4f}".format(
            best_res["aggregate"]["recall@5"],
            best_res["aggregate"]["hit_rate@5"],
            best_res["aggregate"]["mrr"]))
        print("    Total p50={:.0f}ms  Model size={:.1f}MB".format(
            best_res["aggregate"]["total_p50_ms"],
            best_res["aggregate"]["model_size_mb"]))
    else:
        best_name = max(results_all.items(), key=lambda kv: kv[1]["aggregate"]["recall@5"])[0]
        best_res = results_all[best_name]
        print("  WARNING: No model meets <=200ms target.")
        print("  Best quality: {} (Recall@5={:.4f}, total_p50={:.0f}ms)".format(
            best_name, best_res["aggregate"]["recall@5"],
            best_res["aggregate"]["total_p50_ms"]))

    print()
    print("=" * 72)
    print("COMPARISON COMPLETE (no config.py changes made)")
    print("=" * 72)

    # Save results
    out_path = os.path.join(PROJECT_ROOT, "experiments", "two_model_comparison.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results_all, fh, indent=2, ensure_ascii=False)
    print("\nResults saved to: {}".format(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
