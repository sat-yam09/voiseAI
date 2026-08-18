"""Word-based chunking for the RAG retrieval corpus (Member 2: RAG/Retrieval Engineer).

STEP 3 (chunking): turn the cleaned JSONL records into retrieval chunks that a
later embedding/indexing stage can consume.

Scope
-----
- Reads ``data/cleaned/preprocessed.jsonl`` (the current cleaned sample) only.
  It deliberately does NOT process the full original MSMARCO-XI dataset.
- This is a BASELINE implementation using WORD-based chunking: passages are
  split on whitespace, and chunks are windows of ``chunk_size`` words with a
  configurable word ``overlap``. We do NOT pretend words are tokens.
- The TRANSLATED passages are used as the primary retrieval corpus because
  this is a multilingual RAG project.
- No embeddings, vector DB, BM25, hybrid search or reranking here.

What the script does
--------------------
1. Reads every record from the input JSONL.
2. For each record, iterates over ``passages["translated"]`` (a list of
   ``{"text": ..., "is_selected": ...}`` objects).
3. Skips passages whose text is empty.
4. Normalizes whitespace (collapse runs of whitespace to a single space,
   strip leading/trailing space) BEFORE chunking.
5. Splits the normalized text into words and slides a window of
   ``chunk_size`` words forward by ``chunk_size - overlap`` words, so the
   tail of each chunk is repeated as the head of the next one. Passages
   shorter than ``chunk_size`` produce exactly one chunk.
6. Validates arguments up front: chunk size must be >= 1, overlap must be
   >= 0 and strictly smaller than chunk size. This guarantees the sliding
   window always moves forward (no infinite loop).
7. Attaches useful metadata to every chunk so downstream stages can trace it
   back to the original record and passage.
8. Writes the chunks as JSONL and prints statistics.

Output
------
JSONL (one JSON object per line). The default output file is
``data/chunks/chunks_256_32.jsonl`` and the filename always reflects the
configuration. Each chunk looks like:

    {
      "chunk_id": "1102432_p3_c1",
      "query_id": 1102432,
      "query": "...",
      "query_type": "DESCRIPTION",
      "source_lang": "eng_Latn",
      "target_lang": "asm_Beng",
      "passage_index": 3,
      "is_selected": false,
      "chunk_index": 1,
      "num_words": 256,
      "chunk_text": "..."
    }

``chunk_id`` is stable and unique within the file: it is derived from the
record's query id, the passage's position in the translated list and the
chunk's position inside that passage.

Usage
-----
    .venv\\Scripts\\python.exe src\\chunking.py
    .venv\\Scripts\\python.exe src\\chunking.py --chunk-size-words 512 --overlap-words 64
    .venv\\Scripts\\python.exe src\\chunking.py --chunk-size-words 1024 --overlap-words 128
"""

import argparse
import json
import os
import re
import sys

# The data contains non-Latin scripts (Assamese, Hindi, ...). Make sure the
# console can print them on Windows (default cp1252 cannot), replacing
# unprintable ones.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a real stream or already closed
        pass

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


def validate_params(chunk_size_words: int, overlap_words: int) -> None:
    """Validate chunking parameters.

    Raises ``argparse.ArgumentTypeError``-style errors (via ``parser.error``
    at call site is done by the caller; here we raise ValueError) when the
    combination could stall or be nonsensical:
      - chunk size must be at least 1 word,
      - overlap must be >= 0,
      - overlap must be STRICTLY smaller than chunk size, otherwise the
        sliding window would never advance and we would loop forever.
    """
    if chunk_size_words < 1:
        raise ValueError("--chunk-size-words must be >= 1")
    if overlap_words < 0:
        raise ValueError("--overlap-words must be >= 0")
    if overlap_words >= chunk_size_words:
        raise ValueError(
            f"--overlap-words ({overlap_words}) must be strictly smaller than "
            f"--chunk-size-words ({chunk_size_words}) to avoid an infinite loop"
        )


def chunk_words(words, chunk_size_words: int, overlap_words: int):
    """Yield word slices as ``(start, end)`` pairs (sliding window).

    Iterates over ``words`` producing windows of ``chunk_size`` words. Each
    window starts ``chunk_size - overlap`` words after the previous one, so
    the trailing ``overlap`` words are shared with the next chunk. When the
    remaining words are fewer than the chunk size, the last window is the
    tail of the list (never empty). The validation guarantees forward
    progress (step >= 1).
    """
    step = chunk_size_words - overlap_words
    total = len(words)

    start = 0
    while start < total:
        end = min(start + chunk_size_words, total)
        yield start, end
        if end == total:
            break  # last window reached the end: stop
        start += step


def chunk_passage(passage_text: str, chunk_size_words: int, overlap_words: int):
    """Chunk a single normalized passage into a list of chunk text strings.

    Args:
        passage_text:   normalized passage text (already whitespace-normalized).
        chunk_size_words: target number of words per chunk.
        overlap_words:   number of words shared between consecutive chunks.

    Returns:
        list[str] of chunk texts. A passage with <= ``chunk_size`` words
        returns a single-element list.
    """
    words = passage_text.split()  # whitespace already normalized -> clean split
    chunks = []
    for start, end in chunk_words(words, chunk_size_words, overlap_words):
        chunks.append(" ".join(words[start:end]))
    return chunks


def build_chunk_record(record, passage_index: int, passage, chunk_index: int,
                       chunk_text: str) -> dict:
    """Build one output chunk record with all required metadata.

    ``chunk_id`` is stable and unique within the output file: it is made of
    the record's query id, the passage's position in the translated list and
    the chunk's position inside that passage.
    """
    return {
        "chunk_id": f"{record['query_id']}_p{passage_index}_c{chunk_index}",
        "query_id": record.get("query_id"),
        "query": record.get("query"),
        "query_type": record.get("query_type"),
        "source_lang": record.get("source_lang"),
        "target_lang": record.get("target_lang"),
        "passage_index": passage_index,
        "is_selected": bool(passage.get("is_selected")),
        "chunk_index": chunk_index,
        "num_words": len(chunk_text.split()),
        "chunk_text": chunk_text,
    }


def mean(values) -> float:
    """Simple mean; 0.0 for an empty list."""
    return (sum(values) / len(values)) if values else 0.0


def print_header(title: str) -> None:
    """Print a nicely separated section header."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk the TRANSLATED passages of the cleaned JSONL sample "
                    "into word-based retrieval chunks with configurable size "
                    "and overlap."
    )
    parser.add_argument(
        "--input",
        default=os.path.join("data", "cleaned", "preprocessed.jsonl"),
        help="Input cleaned JSONL file "
             "(default: data/cleaned/preprocessed.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "chunks"),
        help="Directory to write the chunk JSONL file "
             "(default: data/chunks).",
    )
    parser.add_argument(
        "--chunk-size-words",
        type=int,
        default=256,
        help="Target number of WORDS per chunk (default: 256).",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=32,
        help="Number of WORDS shared between consecutive chunks "
             "(default: 32). Must be strictly smaller than "
             "--chunk-size-words.",
    )
    args = parser.parse_args(argv)

    # Validate up front so we never start work with a broken configuration
    # (e.g. overlap >= chunk size would make the chunking loop infinite).
    try:
        validate_params(args.chunk_size_words, args.overlap_words)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    # Read the cleaned sample (records, one JSON object per line).
    records = []
    with open(args.input, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Aggregate statistics.
    stats = {
        "records_processed": len(records),
        "passages_processed": 0,
        "empty_passages_ignored": 0,
        "chunks_created": 0,
        "chunk_lengths_words": [],
        "chunk_lengths_chars": [],
        "selected_chunks": 0,
    }

    chunks = []
    for record in records:
        translated = record.get("passages", {}).get("translated") or []
        for passage_index, passage in enumerate(translated):
            raw_text = passage.get("text")
            # 1. Skip empty passages (requirement: ignore empty passage text).
            if not isinstance(raw_text, str) or not raw_text.strip():
                stats["empty_passages_ignored"] += 1
                continue

            stats["passages_processed"] += 1

            # 2. Normalize whitespace BEFORE chunking.
            normalized = normalize_text(raw_text)

            # 3. Word-based chunking with configurable size and overlap.
            passage_chunks = chunk_passage(
                normalized, args.chunk_size_words, args.overlap_words
            )

            for chunk_index, chunk_text in enumerate(passage_chunks):
                chunk = build_chunk_record(
                    record, passage_index, passage, chunk_index, chunk_text
                )
                chunks.append(chunk)
                stats["chunks_created"] += 1
                stats["chunk_lengths_words"].append(chunk["num_words"])
                stats["chunk_lengths_chars"].append(len(chunk_text))
                if chunk["is_selected"]:
                    stats["selected_chunks"] += 1

    # Write chunks to a filename that reflects the configuration.
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"chunks_{args.chunk_size_words}_{args.overlap_words}.jsonl",
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Print statistics.
    lengths = stats["chunk_lengths_words"]
    print_header("CHUNKING STATISTICS")
    print(f"  Configuration          : {args.chunk_size_words} words per chunk, "
          f"{args.overlap_words} words overlap")
    print(f"  Records processed      : {stats['records_processed']:,}")
    print(f"  Passages processed     : {stats['passages_processed']:,}")
    print(f"  Empty passages ignored : {stats['empty_passages_ignored']:,}")
    print(f"  Chunks created         : {stats['chunks_created']:,}")
    print(f"  Chunk length (words)   : "
          f"avg {mean(lengths):,.1f} | "
          f"min {min(lengths):,} | max {max(lengths):,}"
          if lengths else
          "  Chunk length (words)   : n/a (no chunks created)")
    if stats["chunk_lengths_chars"]:
        chars = stats["chunk_lengths_chars"]
        print(f"  Chunk length (chars)   : "
              f"avg {mean(chars):,.1f} | "
              f"min {min(chars):,} | max {max(chars):,}")
    print(f"  Selected chunks        : {stats['selected_chunks']:,}")
    print(f"  Output file            : {output_path}")

    return 0 if chunks else 1


if __name__ == "__main__":
    sys.exit(main())
