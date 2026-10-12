"""
stats.py — chains iter_documents() and tokenize() into a single-pass
pipeline that computes corpus statistics in (roughly) constant memory.

Usage:
    python -m findex.stats data/                  # lazy (default)
    python -m findex.stats data/ --limit 5         # first 5 docs only
    python -m findex.stats data/ --eager           # deliberately greedy version,
                                                    # for the M4 measurement

Both modes print the same numbers: document count, total tokens,
vocabulary size, top-50 terms, elapsed time, and peak memory
(tracemalloc). Only *how* they get there differs.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def run_lazy(root: Path, limit: int | None) -> Counter:
    """The point of this lab: a pipeline of generators, nothing
    materialized except the Counter itself.

    Memory in flight at any moment: one Document, the tokens of its
    current sentence-ish chunk (finditer yields these one at a time
    too), and the running Counter. Never the whole corpus.
    """
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)  # still lazy: only pulls `limit` documents

    token_stream = (tok for doc in docs for tok in tokenize(doc.text))
    return Counter(token_stream)


def run_eager(root: Path, limit: int | None) -> Counter:
    """The deliberately "wrong" version for the M4 comparison: read
    every document into a list, tokenize every document into a list
    of lists, THEN count. Every arrow here is a full materialization.
    """
    docs = list(iter_documents(root))          # (1) whole corpus in memory
    if limit is not None:
        docs = docs[:limit]

    all_tokens: list[list[str]] = []
    for doc in docs:
        all_tokens.append(list(tokenize(doc.text)))  # (2) every doc's tokens as a list

    flat: list[str] = []
    for doc_tokens in all_tokens:
        flat.extend(doc_tokens)                 # (3) one giant flat list of all tokens

    return Counter(flat)                        # (4) counted only at the very end


def _doc_count(root: Path, limit: int | None) -> int:
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)
    return sum(1 for _ in docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="findex corpus statistics")
    parser.add_argument("root", type=Path, help="corpus directory or .jsonl file")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only process the first N documents (uses itertools.islice)",
    )
    parser.add_argument(
        "--eager", action="store_true",
        help="use the deliberately greedy, list-based version (for M4 comparison)",
    )
    parser.add_argument(
        "--top", type=int, default=50, help="how many top terms to print",
    )
    args = parser.parse_args()

    tracemalloc.start()
    start = time.perf_counter()

    counts = run_eager(args.root, args.limit) if args.eager else run_lazy(args.root, args.limit)

    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_docs = _doc_count(args.root, args.limit)
    n_tokens = sum(counts.values())
    vocab_size = len(counts)

    mode = "eager (lists)" if args.eager else "lazy (generators)"
    print(f"mode:              {mode}")
    print(f"documents:         {n_docs}")
    print(f"total tokens:      {n_tokens}")
    print(f"vocabulary size:   {vocab_size}")
    print(f"elapsed:           {elapsed:.4f} s")
    print(f"peak memory:       {peak / 1024:.1f} KiB  ({peak} bytes)")
    print()
    print(f"top {args.top} terms:")
    for term, count in counts.most_common(args.top):
        print(f"  {term:<20} {count}")


if __name__ == "__main__":
    main()
