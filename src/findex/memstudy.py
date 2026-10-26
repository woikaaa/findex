"""
таблиця порівняння пам'яті: три способи зберігати записи на одному й тому ж корпусі.

    python -m findex.memstudy data/
    python -m findex.memstudy data/ --positions      # (пропускає рядок array)
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
from findex.model import Storage

# (підпис у таблиці, назва способу зберігання)
ROWS: list[tuple[str, Storage]] = [
    ("list[Posting], plain @dataclass", "dataclass"),
    ("list[Posting], slots=True", "slots"),
    ("array('I') pairs (no objects)", "array"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--positions", action="store_true")
    a = p.parse_args()

    print(
        "| Postings representation | Peak memory (build) | Index memory after build "
        "| Pickle size | Load time |"
    )
    print(
        "| ----------------------- | ------------------- | ------------------------ "
        "| ----------- | --------- |"
    )
    for label, storage in ROWS:
        if storage == "array" and a.positions:
            continue  # array не вміє зберігати позиції
        gc.collect()  # чистимо пам'ять перед кожним виміром
        tracemalloc.start()
        # будую індекс цим способом зберігання
        index = build_index(a.root, storage, a.positions)
        current, peak = tracemalloc.get_traced_memory()  # зараз і пік
        tracemalloc.stop()

        # зберігаємо у тимчасову папку, міряємо розмір файлу і час завантаження
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "idx.pkl"
            store.save_pickle(index, path)
            # розмір файлу в байтах
            size = path.stat().st_size
            # видаляю індекс, щоб міряти чисте завантаження
            del index
            gc.collect()
            t0 = time.perf_counter()
            # завантажую назад і міряю час
            store.load_pickle(path)
            load = time.perf_counter() - t0
        print(
            f"| {label} | {peak / 1e6:.1f} MB | {current / 1e6:.1f} MB "
            f"| {size / 1e6:.2f} MB | {load:.3f} s |"
        )


if __name__ == "__main__":
    main()
