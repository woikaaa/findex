"""
search.py — Boolean search over a saved index.

    python -m findex.search index.bin "alice rabbit"          # implicit AND
    python -m findex.search index.bin "queen OR duchess"
    python -m findex.search index.bin "alice NOT rabbit" --engine set

Query grammar (evaluated left to right, no parentheses — Lab 3 adds a parser):
    term term        AND
    term AND term    AND
    term OR term     OR
    term NOT term    AND NOT
Operators must be UPPERCASE so that the ordinary word "or" is still searchable.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

from findex import store
from findex.model import Index
from findex.tokenize import tokenize


# --------------------------------------------------- engine 1: two-pointer merge
def and_merge(a, b) -> list[int]:
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def or_merge(a, b) -> list[int]:
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def andnot_merge(a, b) -> list[int]:
    """a AND NOT b"""
    i = j = 0
    out = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i]); i += 1
        elif a[i] == b[j]:
            i += 1; j += 1
        else:
            j += 1
    return out


# --------------------------------------------------- engine 2: Python sets
def and_set(a, b) -> list[int]:
    return sorted(set(a) & set(b))


def or_set(a, b) -> list[int]:
    return sorted(set(a) | set(b))


def andnot_set(a, b) -> list[int]:
    return sorted(set(a) - set(b))


ENGINES = {
    "merge": (and_merge, or_merge, andnot_merge),
    "set": (and_set, or_set, andnot_set),
}


def evaluate(index: Index, query: str, engine: str = "merge") -> list[int]:
    """Left-to-right Boolean evaluation. Returns sorted doc_ids."""
    f_and, f_or, f_andnot = ENGINES[engine]
    result = None
    op = "AND"
    negate = False

    for word in query.split():
        if word in ("AND", "OR"):
            op = word
            continue
        if word == "NOT":
            negate = True
            continue
        for term in tokenize(word):
            ids = index.doc_ids(term)
            if result is None:
                result = f_andnot(index.all_doc_ids(), ids) if negate else list(ids)
            elif op == "AND":
                result = f_andnot(result, ids) if negate else f_and(result, ids)
            else:  # OR
                if negate:
                    ids = f_andnot(index.all_doc_ids(), ids)
                result = f_or(result, ids)
            op, negate = "AND", False
    return result or []


def main() -> None:
    p = argparse.ArgumentParser(description="search a saved findex index")
    p.add_argument("index", type=Path)
    p.add_argument("query")
    p.add_argument("--engine", choices=tuple(ENGINES), default="merge")
    p.add_argument("--max", type=int, default=10, help="max results to print")
    a = p.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()
    index = store.load(a.index)
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    hits = evaluate(index, a.query, a.engine)
    search_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"{len(hits)} result(s) for {a.query!r}  [engine={a.engine}]")
    for doc_id in hits[: a.max]:
        m = index.doc_meta[doc_id]
        print(f"  #{doc_id:<4} {m.path}  —  {m.title}")
    if len(hits) > a.max:
        print(f"  ... and {len(hits) - a.max} more")
    print(f"load: {load_time:.4f} s   search: {search_time * 1000:.3f} ms   "
          f"peak memory: {peak / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
