"""Data preprocessing for ai4bharat/MSMARCO-XI (Member 2: RAG/Retrieval Engineer).

STEP 2 (preprocessing): turn raw MSMARCO-XI records into a clean, structured
JSONL file that is ready for the later chunking stage.

Scope
-----
- Works on a SMALL SAMPLE / SHARD of the dataset (default: first validation
  shard, capped at 500 records). It deliberately does NOT process the full
  ~147 GB dataset yet.
- No chunking, embeddings, vector DB, BM25 or reranking here.

What the script does per record
-------------------------------
1. Normalizes whitespace in query/answer/passage texts: collapses any run of
   whitespace (spaces, tabs, newlines, NBSP, ...) into a single space, strips
   leading/trailing space, maps NBSP to a space and removes zero-width/BOM
   characters. Case is left untouched on purpose so that names, codes and
   other case-sensitive content are preserved.
2. Removes empty passages from the English and Translated lists independently.
   A passage that was flagged ``is_selected`` and turned out empty is treated
   as a broken record -> the whole record is dropped.
3. Detects and removes exact duplicate passages (after normalization) within
   each language list of a single record, keeping the first occurrence.
   Duplicates across DIFFERENT records are NOT removed: in MSMARCO the same
   passage legitimately appears under many queries.
4. Drops a record when: its query is empty, or it has no passages left in
   either language, or one of its selected passages was empty.
5. Keeps useful metadata (query_id, query_type, source_lang, target_lang,
   meta) and the query/answer fields.
6. Validates the output and reports if any cleaned passage is still empty.

Output
------
JSONL (one JSON object per line) to ``data/cleaned/preprocessed.jsonl`` by
default. Each kept record looks like:

    {
      "query_id": "284376",
      "query_type": "wellformed",
      "source_lang": "hi",
      "target_lang": "en",
      "query": "...",
      "Eng_Query": "...",
      "Answer": "...",
      "Eng_Answer": "...",
      "passages": {
        "english":    [{"text": "...", "is_selected": true}, ...],
        "translated": [{"text": "...", "is_selected": false}, ...]
      },
      "meta": {"model_name": "...", "temperature": 0.0, ...}
    }

Passages are stored per language as a list of {"text", "is_selected"} so the
alignment between passage text and its selection flag is unambiguous for the
chunking stage.

Usage
-----
    .venv\\Scripts\\python.exe src\\preprocess.py
    .venv\\Scripts\\python.exe src\\preprocess.py --max-records 1000
    .venv\\Scripts\\python.exe src\\preprocess.py --split train --shards 1
    .venv\\Scripts\\python.exe src\\preprocess.py --output data/cleaned/test.jsonl
"""

import argparse
import collections
import json
import os
import re
import sys

# Silence the "no symlinks on Windows" warning that huggingface_hub prints.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# The data contains non-Latin scripts (Hindi, Tamil, ...). Make sure the console
# can print them on Windows (default cp1252 cannot), replacing unprintable ones.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a real stream or already closed
        pass

# Reuse the proven shard loader from the analysis script (same one-time
# download + huggingface_hub cache). The try/except supports both
# `python src/preprocess.py` and `python -m src.preprocess`.
try:
    from dataset_analysis import load_records
except ImportError:
    from src.dataset_analysis import load_records

# NBSP -> normal space; zero-width space and BOM are meaningless and removed.
NBSP_AND_ZERO_WIDTH = str.maketrans({"\u00a0": " ", "\u200b": "", "\ufeff": ""})
# Matches ANY run of whitespace (spaces, tabs, newlines, etc.).
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text) -> str:
    """Normalize whitespace without changing case or content.

    - collapses tabs/newlines/NBSP/... runs into a single space
    - strips leading/trailing whitespace
    - leaves the actual characters (incl. case and scripts) untouched
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(NBSP_AND_ZERO_WIDTH)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_language(raw_items, selected_flags):
    """Clean + dedup one language's passage list.

    Args:
        raw_items:      raw passage strings (may contain None / "").
        selected_flags: parallel list of bool/None marking selected passages.

    Returns:
        (cleaned, duplicates_removed, empty_removed, empty_selected)
        where ``cleaned`` is a list of {"text", "is_selected"} dicts.
    """
    cleaned = []
    seen = set()
    duplicates_removed = 0
    empty_removed = 0
    empty_selected = 0

    for index, raw in enumerate(raw_items):
        text = normalize_text(raw)
        flag = bool(selected_flags[index]) if index < len(selected_flags) and selected_flags[index] else False

        if not text:
            # Empty passage: drop it. If it was a selected passage the whole
            # record is broken (handled by the caller via ``empty_selected``).
            empty_removed += 1
            if flag:
                empty_selected += 1
            continue

        if text in seen:
            # Exact duplicate (after normalization): keep only the first one.
            duplicates_removed += 1
            continue

        seen.add(text)
        cleaned.append({"text": text, "is_selected": flag})

    return cleaned, duplicates_removed, empty_removed, empty_selected


def clean_passages(passages):
    """Clean, dedup and count removals for a record's passages.

    ``passages`` is the raw value of the record's ``passages`` field, e.g.
    {"English_passages": [...], "Translated_passages": [...], "is_selected": [...]}.

    English and Translated lists are processed independently; each keeps its
    own alignment with ``is_selected``.
    """
    if not isinstance(passages, dict):
        passages = {}
    english_raw = passages.get("English_passages") or []
    translated_raw = passages.get("Translated_passages") or []
    selected = passages.get("is_selected") or []

    english, eng_dup, eng_empty, eng_empty_sel = clean_language(english_raw, selected)
    translated, trl_dup, trl_empty, trl_empty_sel = clean_language(translated_raw, selected)

    return {
        "english": english,
        "translated": translated,
        "duplicates_removed": eng_dup + trl_dup,
        "empty_removed": eng_empty + trl_empty,
        "empty_selected": eng_empty_sel + trl_empty_sel,
    }


def preprocess_record(record):
    """Clean one raw record.

    Returns:
        (cleaned_record, info) if the record is kept, else (None, info).
        ``info`` is a dict with per-record counters and the drop reason.
    """
    # 1. Clean the plain-text fields.
    query = normalize_text(record.get("query"))
    eng_query = normalize_text(record.get("Eng_Query"))
    answer = normalize_text(record.get("Answer"))
    eng_answer = normalize_text(record.get("Eng_Answer"))

    # 2. Clean passages.
    passages = clean_passages(record.get("passages"))

    info = {
        "drop_reason": None,
        "duplicates_removed": passages["duplicates_removed"],
        "empty_removed": passages["empty_removed"],
        "empty_selected": passages["empty_selected"],
    }

    # 3. Decide whether to keep the record (requirement: remove records where
    #    the selected text/passages are empty).
    if not query:
        info["drop_reason"] = "empty query"
    elif not passages["english"] and not passages["translated"]:
        info["drop_reason"] = "no passages"
    elif passages["empty_selected"] > 0:
        info["drop_reason"] = "empty selected passage"

    if info["drop_reason"]:
        return None, info

    # 4. Build the clean structured record (keep useful metadata + original
    #    query/passage information needed for retrieval).
    cleaned = {
        "query_id": record.get("query_id"),
        "query_type": record.get("query_type"),
        "source_lang": record.get("source_lang"),
        "target_lang": record.get("target_lang"),
        "query": query,
        "Eng_Query": eng_query,
        "Answer": answer,
        "Eng_Answer": eng_answer,
        "passages": {
            "english": passages["english"],
            "translated": passages["translated"],
        },
        "meta": record.get("meta"),
    }
    return cleaned, info


def validate_cleaned(records) -> int:
    """Report if any cleaned query/passage is empty. Returns number of problems."""
    problems = 0
    for rec in records:
        if not rec["query"]:
            problems += 1
            print(f"  [validation] empty query in record: {rec.get('query_id')}")
        for lang in ("english", "translated"):
            for passage in rec["passages"][lang]:
                if not passage["text"]:
                    problems += 1
                    print(
                        f"  [validation] empty {lang} passage in record: "
                        f"{rec.get('query_id')}"
                    )
    return problems


def mean_length(lengths) -> float:
    """Simple mean; 0.0 for an empty list."""
    return (sum(lengths) / len(lengths)) if lengths else 0.0


def print_header(title: str) -> None:
    """Print a nicely separated section header."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a small sample of ai4bharat/MSMARCO-XI into a "
                    "clean structured JSONL file (whitespace normalization, "
                    "empty/dedup removal, metadata preserved)."
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="Which split to sample from (default: validation). "
             "WARNING: train shards are ~3.7 GB each.",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="Number of parquet shards to load (default: 1; validation shard "
             "~0.47 GB).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=500,
        help="Maximum number of records to preprocess (default: 500).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "cleaned", "preprocessed.jsonl"),
        help="Where to write the cleaned JSONL file "
             "(default: data/cleaned/preprocessed.jsonl).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Load a small sample/shard (cached by huggingface_hub after first run).
    loaded = load_records(args.split, args.shards, args.max_records)
    records = loaded["records"]

    # Aggregate statistics.
    stats = {
        "records_before": len(records),
        "records_after": 0,
        "empty_records_removed": 0,
        "drop_reasons": collections.Counter(),
        "duplicate_passages_removed": 0,
        "empty_passages_removed": 0,
        "empty_selected_removed": 0,
        "passage_lengths_before": [],
        "passage_lengths_after": [],
    }

    cleaned_records = []
    for record in records:
        # Raw (pre-clean) passage lengths, for the "before" average.
        raw_passages = record.get("passages") or {}
        for lang in ("English_passages", "Translated_passages"):
            for item in raw_passages.get(lang) or []:
                stats["passage_lengths_before"].append(len(str(item)))

        cleaned, info = preprocess_record(record)
        if cleaned is None:
            stats["empty_records_removed"] += 1
            stats["drop_reasons"][info["drop_reason"]] += 1
            continue

        cleaned_records.append(cleaned)
        stats["duplicate_passages_removed"] += info["duplicates_removed"]
        stats["empty_passages_removed"] += info["empty_removed"]
        stats["empty_selected_removed"] += info["empty_selected"]
        for lang in ("english", "translated"):
            for passage in cleaned["passages"][lang]:
                stats["passage_lengths_after"].append(len(passage["text"]))

    stats["records_after"] = len(cleaned_records)

    # Validation: report if any cleaned passage is empty.
    problems = validate_cleaned(cleaned_records)

    # Write the clean structured output.
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for record in cleaned_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print before/after statistics.
    print_header("PREPROCESSING STATISTICS")
    print(f"  Records before             : {stats['records_before']:,}")
    print(f"  Records after              : {stats['records_after']:,}")
    print(f"  Empty records removed      : {stats['empty_records_removed']:,}")
    for reason, count in stats["drop_reasons"].most_common():
        print(f"      - {reason:<28}: {count:,}")
    print(
        f"  Duplicate passages removed : {stats['duplicate_passages_removed']:,}"
    )
    print(
        f"  Empty passages removed     : {stats['empty_passages_removed']:,} "
        f"(of which selected: {stats['empty_selected_removed']:,})"
    )
    before = mean_length(stats["passage_lengths_before"])
    after = mean_length(stats["passage_lengths_after"])
    print(
        f"  Avg passage length before  : {before:,.1f} chars "
        f"(n={len(stats['passage_lengths_before']):,})"
    )
    print(
        f"  Avg passage length after   : {after:,.1f} chars "
        f"(n={len(stats['passage_lengths_after']):,})"
    )
    print(f"  Output file                : {args.output}")
    if problems == 0:
        print("  Validation                 : OK - no empty passages in output")
    else:
        print(f"  Validation                 : {problems} problem(s) found - see above")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
