"""
index.py — builds the inverted index in ONE pass over the lazy pipeline
from Lab 1, and saves it.

    python -m findex.index data/ --out index.bin
    python -m findex.index data/ --out index.json.bin --format binary
    python -m findex.index data/ --out index.bin --positions
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.model import DocMeta, Index, PlainPosting, Posting
from findex.tokenize import tokenize
from findex.utils import timed

STORAGES = ("dataclass", "slots", "array")


def _title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return ""


@timed
def build_index(root: Path, storage: str = "slots", positions: bool = False,
                limit: int | None = None) -> Index:
    """Stream documents one by one; only the index itself grows."""
    if storage not in STORAGES:
        raise ValueError(f"storage must be one of {STORAGES}")
    if storage == "array" and positions:
        raise ValueError("array storage does not support positions")

    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)

    # defaultdict: no "if term not in postings" needed.
    if storage == "array":
        postings = defaultdict(lambda: (array("I"), array("I")))
    else:
        postings = defaultdict(list)
    make = Posting if storage == "slots" else PlainPosting

    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(docs):
        counts: Counter[str] = Counter()          # current document only
        offsets = defaultdict(list) if positions else None
        n = 0
        for i, tok in enumerate(tokenize(doc.text)):
            counts[tok] += 1
            n = i + 1
            if offsets is not None:
                offsets[tok].append(i)

        # flush this document's counter into the postings.
        # doc_ids arrive in increasing order => every list is already sorted.
        for term, tf in counts.items():
            if storage == "array":
                ids, tfs = postings[term]
                ids.append(doc_id)
                tfs.append(tf)
            else:
                pos = tuple(offsets[term]) if offsets is not None else ()
                postings[term].append(make(doc_id, tf, pos))

        doc_lengths[doc_id] = n
        doc_meta[doc_id] = DocMeta(str(doc.path), _title(doc.text))

    # dict(...) drops the defaultdict factory (a lambda cannot be pickled)
    return Index(storage, positions, dict(postings), doc_lengths, doc_meta)


def main() -> None:
    from findex import store

    p = argparse.ArgumentParser(description="build and save the inverted index")
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--format", choices=("pickle", "binary"), default="pickle")
    p.add_argument("--storage", choices=STORAGES, default="slots")
    p.add_argument("--positions", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    storage = a.storage
    if a.format == "binary" and storage != "array":
        storage = "array"   # binary format is array-based by design
        if a.positions:
            p.error("--positions is not supported by the binary format")

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(a.root, storage, a.positions, a.limit)
    build_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.perf_counter()
    store.save(index, a.out, a.format)
    save_time = time.perf_counter() - t0

    print(f"documents:    {len(index.doc_meta)}")
    print(f"terms:        {len(index.postings)}")
    print(f"storage:      {index.storage}   positions: {index.has_positions}")
    print(f"build time:   {build_time:.3f} s")
    print(f"peak memory:  {peak / 1024:.1f} KiB (build)")
    print(f"saved:        {a.out} ({a.format}, {a.out.stat().st_size} bytes, {save_time:.3f} s)")


if __name__ == "__main__":
    main()
