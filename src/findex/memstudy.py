"""
memstudy.py — the M4 table: three ways to store postings, same corpus.

    python -m findex.memstudy data/
    python -m findex.memstudy data/ --positions      # (skips the array row)
"""

from __future__ import annotations

import argparse
import gc
import tempfile
import time
import tracemalloc
from pathlib import Path

from findex import store
from findex.index import build_index

ROWS = [
    ("list[Posting], plain @dataclass", "dataclass"),
    ("list[Posting], slots=True", "slots"),
    ("array('I') pairs (no objects)", "array"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--positions", action="store_true")
    a = p.parse_args()

    print("| Postings representation | Peak memory (build) | Index memory after build "
          "| Pickle size | Load time |")
    print("| ----------------------- | ------------------- | ------------------------ "
          "| ----------- | --------- |")
    for label, storage in ROWS:
        if storage == "array" and a.positions:
            continue
        gc.collect()
        tracemalloc.start()
        index = build_index(a.root, storage, a.positions)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "idx.pkl"
            store.save_pickle(index, path)
            size = path.stat().st_size
            del index
            gc.collect()
            t0 = time.perf_counter()
            store.load_pickle(path)
            load = time.perf_counter() - t0
        print(f"| {label} | {peak / 1e6:.1f} MB | {current / 1e6:.1f} MB "
              f"| {size / 1e6:.2f} MB | {load:.3f} s |")


if __name__ == "__main__":
    main()
