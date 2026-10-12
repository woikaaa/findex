"""
bench.py — merge vs set on the two most common and two rarest terms.

    python -m findex.bench index.bin
"""

from __future__ import annotations

import argparse
import timeit
from pathlib import Path

from findex import store
from findex.search import ENGINES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("index", type=Path)
    p.add_argument("--number", type=int, default=200)
    a = p.parse_args()

    index = store.load(a.index)
    by_df = sorted(index.postings, key=lambda t: (-index.df(t), t))
    common = by_df[:2]
    rare = by_df[-2:]
    cases = [
        ("common AND common", common[0], common[1]),
        ("rare AND rare", rare[0], rare[1]),
        ("common AND rare", common[0], rare[0]),
    ]

    print(f"common: {[(t, index.df(t)) for t in common]}")
    print(f"rare:   {[(t, index.df(t)) for t in rare]}\n")
    print("| Case | terms (df) | merge | set |")
    print("| ---- | ---------- | ----- | --- |")
    for name, t1, t2 in cases:
        A, B = list(index.doc_ids(t1)), list(index.doc_ids(t2))
        row = []
        for engine in ("merge", "set"):
            f_and = ENGINES[engine][0]
            t = timeit.timeit(lambda f=f_and, x=A, y=B: f(x, y), number=a.number) / a.number
            row.append(f"{t * 1e6:.1f} µs")
        print(f"| {name} | {t1} ({len(A)}), {t2} ({len(B)}) | {row[0]} | {row[1]} |")


if __name__ == "__main__":
    main()
