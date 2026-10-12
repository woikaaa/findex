"""
bench.py — порівняння merge і set на двох найчастіших та двох найрідших словах.
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
    p.add_argument("--number", type=int, default=200)  # скільки разів повторити вимір
    a = p.parse_args()

    # завантажую індекс
    index = store.load(a.index)
    # сортуємо слова від найчастіших (більше документів) до найрідших
    by_df = sorted(index.postings, key=lambda t: (-index.df(t), t))
    common = by_df[:2]    # два найчастіші
    rare = by_df[-2:]     # два найрідші
    cases = [
        ("common AND common", common[0], common[1]),
        ("rare AND rare", rare[0], rare[1]),
        ("common AND rare", common[0], rare[0]),
    ]

    print(f"common: {[(t, index.df(t)) for t in common]}")
    print(f"rare:   {[(t, index.df(t)) for t in rare]}\n")
    print("| Case | terms (df) | merge | set |")
    print("| ---- | ---------- | ----- | --- |")
    # перевіряю кожну пару слів
    for name, t1, t2 in cases:
        # списки документів двох слів
        A, B = list(index.doc_ids(t1)), list(index.doc_ids(t2))
        row = []
        # міряю обидва способи
        for engine in ("merge", "set"):
            # нам потрібна лише функція AND
            f_and = ENGINES[engine][0]
            # timeit.timeit запускає функцію багато разів і повертає загальний час
            t = timeit.timeit(lambda f=f_and, x=A, y=B: f(x, y), number=a.number) / a.number
            row.append(f"{t * 1e6:.1f} µs")
        print(f"| {name} | {t1} ({len(A)}), {t2} ({len(B)}) | {row[0]} | {row[1]} |")


if __name__ == "__main__":
    main()
