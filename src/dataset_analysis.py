"""Dataset analysis for ai4bharat/MSMARCO-XI (Member 2: RAG/Retrieval Engineer).

This is dataset analysis ONLY. It does not chunk, embed, index or call any LLM.

What the script does
--------------------
1. Prints the dataset schema/features and the available splits.
2. Prints the column names and classifies each column as:
   - text column  (a plain string field, e.g. query, answer)
   - passage field (a list of strings, e.g. translated passages)
   - metadata field (ids, languages, inference settings, etc.)
3. Loads a SMALL slice of real data (default: first validation shard,
   ~0.47 GB, capped at ``--max-records`` rows) with the ``datasets`` library.
4. Prints a few sample records.
5. Prints text-length statistics, empty-text counts and duplicate-text counts
   computed ONLY from the loaded sample. No statistics are hard-coded.

Why load by direct parquet URL instead of ``load_dataset("ai4bharat/MSMARCO-XI")``?
---------------------------------------------------------------------------------
- The full dataset is ~147 GB (train ~130 GB + validation ~17 GB), so we only
  ever load one small shard.
- The dataset repo's git/REST API returns 401 anonymously, but the actual
  data files (parquet shards) are public and resolvable by URL.
- Every shard is a single parquet row group, so even "streaming" transfers the
  whole shard. Loading one small shard once and reusing the cache is simplest.

Windows notes
-------------
- ``HF_HUB_DISABLE_SYMLINKS_WARNING=1`` silences the noisy symlink warning
  (Developer Mode is not required for this script to work).
- The shard is cached by huggingface_hub, so reruns are fast.
- Only the ``datasets`` library (already installed) and the stdlib are used.

Usage
-----
    python src/dataset_analysis.py
    python src/dataset_analysis.py --split train --shards 1 --max-records 100
    python src/dataset_analysis.py --max-records 200 --samples 3
"""

import argparse
import collections
import json
import os
import statistics
import sys
import urllib.request

# Silence the "no symlinks on Windows" warning that huggingface_hub prints.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# The data contains non-Latin scripts (Hindi, Tamil, ...). Make sure the console
# can print them on Windows (default cp1252 cannot), replacing unprintable ones.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a real stream or already closed
        pass

from datasets import get_dataset_config_info, load_dataset
from datasets.features import Sequence, Value

DATASET_ID = "ai4bharat/MSMARCO-XI"
CONFIG_NAME = "default"

# String fields that are really metadata identifiers/codes, not free text.
# Used only to classify columns for display; statistics are never hard-coded.
METADATA_STRING_FIELDS = {"source_lang", "target_lang", "query_type", "model_name"}

# datasets-server exposes the real parquet file list anonymously.
DATASETS_SERVER_PARQUET = (
    "https://datasets-server.huggingface.co/parquet?"
    "dataset=ai4bharat%2FMSMARCO-XI"
)


def section(title: str) -> None:
    """Print a nicely separated section header."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def shorten(value: str, limit: int = 160) -> str:
    """Truncate long strings for readable console output (display only)."""
    value = str(value)
    return value if len(value) <= limit else value[:limit] + " ... [truncated]"


# ---------------------------------------------------------------------------
# 1. Schema / splits / column classification
# ---------------------------------------------------------------------------

def print_schema_and_splits() -> None:
    """Fetch and print the real dataset features and split sizes from the Hub."""
    info = get_dataset_config_info(DATASET_ID, config_name=CONFIG_NAME)

    section("DATASET SCHEMA / FEATURES")
    print(json.dumps(info.features.to_dict(), indent=2, default=str))

    section("AVAILABLE SPLITS")
    for split_name, split_info in sorted(info.splits.items()):
        mb = split_info.num_bytes / (1024 * 1024)
        print(
            f"  {split_name:<12} examples: {split_info.num_examples:>10,}  "
            f"size: {mb / 1024:,.1f} GB"
        )

    print("\nTop-level columns: " + ", ".join(info.features.keys()))


def classify_fields(features) -> dict:
    """Classify every field as text, passage, or metadata based on its type.

    Returns {"text": [...], "passages": [...], "metadata": [...]} with
    dotted paths for nested fields (e.g. "passages.English_passages").
    """
    result = {"text": [], "passages": [], "metadata": []}

    def walk(path: str, feat) -> None:
        if isinstance(feat, dict):
            # Nested dict: recurse into its children.
            for child_name, child_feat in feat.items():
                walk(f"{path}.{child_name}", child_feat)
        elif isinstance(feat, Sequence):
            # A list: a list of strings is a "passages" field, anything else
            # (e.g. a list of ints) is treated as metadata.
            if isinstance(feat.feature, Value) and feat.feature.dtype == "string":
                result["passages"].append(path)
            else:
                result["metadata"].append(path)
        elif isinstance(feat, Value):
            # `feat` carries no name, so use the last path segment.
            field_name = path.rsplit(".", 1)[-1]
            if feat.dtype == "string" and field_name not in METADATA_STRING_FIELDS:
                result["text"].append(path)
            else:
                result["metadata"].append(path)
        else:
            result["metadata"].append(path)

    for name, feat in features.items():
        walk(name, feat)
    return result


def print_column_roles(features) -> None:
    """Print which columns hold text, passages, or metadata."""
    roles = classify_fields(features)

    section("COLUMN ROLES")
    print("  TEXT columns (plain strings):")
    for name in roles["text"]:
        print(f"    - {name}")
    print("  PASSAGE fields (lists of strings):")
    for name in roles["passages"]:
        print(f"    - {name}")
    print("  METADATA fields (ids, languages, settings, ...):")
    for name in roles["metadata"]:
        print(f"    - {name}")
    return roles


# ---------------------------------------------------------------------------
# 2. Load a small sample of real data
# ---------------------------------------------------------------------------

def fetch_parquet_files(split: str) -> list:
    """Fetch the real parquet shard URLs for a split from datasets-server."""
    request = urllib.request.Request(
        DATASETS_SERVER_PARQUET, headers={"User-Agent": "dataset-analysis"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)

    files = [f for f in payload["parquet_files"] if f["split"] == split]
    # Deterministic ordering by shard filename (0000.parquet, 0001.parquet, ...).
    files.sort(key=lambda f: f["filename"])
    return files


def load_records(split: str, shards: int, max_records: int) -> dict:
    """Download at most ``shards`` shard(s) and return a capped list of records.

    The download is one-time; huggingface_hub caches it for later runs.
    """
    section(f"LOADING DATA (split={split!r})")
    files = fetch_parquet_files(split)
    if not files:
        raise RuntimeError(f"No parquet files found for split {split!r}")

    total_gb = sum(f["size"] for f in files) / (1024 ** 3)
    chosen = files[:shards]
    chosen_gb = sum(f["size"] for f in chosen) / (1024 ** 3)

    print(f"  Split has {len(files)} shards, ~{total_gb:,.2f} GB total.")
    print(f"  Loading the first {len(chosen)} shard(s): ~{chosen_gb:,.2f} GB.")
    print("  This is a one-time download on first run; later runs use the cache.\n")

    urls = [f["url"] for f in chosen]
    # split="train" is the default split name of the parquet builder; it has no
    # special meaning here, it just maps the files we pass in.
    dataset = load_dataset("parquet", data_files=urls, split="train")

    n = min(max_records, dataset.num_rows)
    print(f"  Loaded {dataset.num_rows:,} rows; using a sample of {n:,} for analysis.")
    return {"records": [dataset[i] for i in range(n)], "features": dataset.features}


# ---------------------------------------------------------------------------
# 3. Sample records
# ---------------------------------------------------------------------------

def print_samples(records: list, samples: int) -> None:
    """Pretty-print the first few records with long strings shortened."""
    section("SAMPLE RECORDS")

    def shorten_json(obj):
        if isinstance(obj, dict):
            return {k: shorten_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [shorten_json(v) for v in obj]
        if isinstance(obj, str):
            return shorten(obj)
        return obj

    for i in range(min(samples, len(records))):
        print(f"\n--- record #{i} ---")
        print(json.dumps(shorten_json(records[i]), indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 4. Statistics computed from the loaded sample
# ---------------------------------------------------------------------------

def collect_field_values(records: list, field_path: str) -> list:
    """Extract the raw string values for a text/passage field across records.

    - text field: one string per record (missing/empty -> "")
    - passage field: every string inside the list, flattened across records
    """
    values = []
    for record in records:
        value = record
        for part in field_path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, list):
            values.extend([str(v) for v in value])  # flatten passage lists
        elif value is None:
            values.append("")
        else:
            values.append(str(value))
    return values


def text_stats(values: list) -> dict:
    """Character-length statistics for a list of strings (not empty-safe)."""
    lengths = [len(v) for v in values]
    return {
        "count": len(lengths),
        "empty_text_count": sum(1 for length in lengths if length == 0),
        "min_length": min(lengths),
        "mean_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
        "max_length": max(lengths),
        "stdev_length": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
    }


def duplicate_counts(values: list) -> dict:
    """Count duplicate text values within the loaded sample."""
    counter = collections.Counter(values)
    duplicated_total = sum(count - 1 for count in counter.values() if count > 1)
    duplicated_distinct = sum(1 for count in counter.values() if count > 1)
    return {
        "duplicate_occurrences": duplicated_total,
        "distinct_values_that_duplicate": duplicated_distinct,
        "unique_value_count": len(counter),
    }


def print_statistics(records: list, roles: dict) -> None:
    """Print length/empty/duplicate statistics per text and passage field."""
    section("TEXT STATISTICS (computed from the loaded sample only)")

    fields = roles["text"] + roles["passages"]
    for field in fields:
        values = collect_field_values(records, field)
        if not values:
            print(f"\n  [{field}]  no values in the sample")
            continue
        stats = text_stats(values)
        dups = duplicate_counts(values)
        print(f"\n  [{field}]")
        print(
            f"    values            : {stats['count']:>6,} "
            f"(records: {len(records):,})"
        )
        print(f"    empty text count  : {stats['empty_text_count']:>6,}")
        print(
            f"    char length       : min {stats['min_length']:>6,} | "
            f"mean {stats['mean_length']:>8,.1f} | "
            f"median {stats['median_length']:>6,} | "
            f"max {stats['max_length']:>6,} | "
            f"stdev {stats['stdev_length']:.1f}"
        )
        print(
            f"    duplicate text    : {dups['duplicate_occurrences']:>6,} extra "
            f"occurrences across {dups['distinct_values_that_duplicate']:,} "
            f"distinct value(s) (unique values: {dups['unique_value_count']:,})"
        )


def print_summary(records: list, roles: dict) -> None:
    """Short human-readable wrap-up."""
    section("SUMMARY")
    print(f"  Records analyzed : {len(records):,}")
    print(f"  Text columns     : {', '.join(roles['text'])}")
    print(f"  Passage fields   : {', '.join(roles['passages'])}")
    print(f"  Metadata fields  : {', '.join(roles['metadata'])}")
    print(
        "\n  Note: numbers above cover only the loaded sample, not the full "
        "dataset (~147 GB).\n  The full dataset is public; "
        "increase --shards/--max-records for more coverage."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the ai4bharat/MSMARCO-XI dataset (schema, splits, "
                    "text statistics) on a small sampled portion."
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
        help="Maximum number of records to analyze (default: 500).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="How many sample records to print (default: 2).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        print_schema_and_splits()
        features = get_dataset_config_info(DATASET_ID, config_name=CONFIG_NAME).features
        roles = print_column_roles(features)

        loaded = load_records(args.split, args.shards, args.max_records)
        records = loaded["records"]

        print_samples(records, args.samples)
        print_statistics(records, roles)
        print_summary(records, roles)
    except Exception as exc:  # keep the script safe to run on any machine
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Check your internet connection and that HF is reachable.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
