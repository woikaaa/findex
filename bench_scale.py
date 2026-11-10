"""
bench_scale.py: пам'ять і швидкість на великому індексі (лаба 8, замість семантичного пошуку).

    findex index data\\wiki10x -o data\\wiki10x.bin --storage array
    python bench_scale.py data\\wiki10x.bin

Порівнюю три способи тримати індекс. Кожен спосіб запускаю в ОКРЕМОМУ процесі, бо тільки
так видно чесну пам'ять (RSS: скільки оперативної пам'яті процес реально займає):
  lab3    - store.load: індекс у звичайних структурах Python (лаби 2-3);
  numpy   - NpIndex.from_file(mmap=False): масиви int32 читаю в пам'ять;
  memmap  - NpIndex.from_file(mmap=True): масиви лишаються у файлі, ОС підвантажує потрібні
            сторінки сама.
Для кожного: час завантаження, пам'ять після завантаження, час пошуку трьох запитів
(частотне слово, рідкісне, три слова) і пам'ять після пошуків.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import psutil

from findex import store
from findex.bench_cases import pick_cases
from findex.npsearch import NpIndex, NpSearcher
from findex.ranking import Searcher


def rss_mb() -> float:
    """Скільки МБ оперативної пам'яті займає цей процес."""
    return psutil.Process().memory_info().rss / 1e6


def child(mode: str, path: Path, cases: list[tuple[str, str]], repeat: int) -> None:
    """Один спосіб, один процес. Результат друкую як JSON в один рядок."""
    base = rss_mb()
    t0 = time.perf_counter()
    search: Any
    if mode == "lab3":
        search = Searcher(store.load(path)).search
    else:
        search = NpSearcher(NpIndex.from_file(path, mmap=(mode == "memmap"))).search
    load_ms = (time.perf_counter() - t0) * 1000
    after_load = rss_mb()
    ms: dict[str, float] = {}
    for name, q in cases:
        search(q, 10)  # прогрів (кеш запиту)
        runs: list[float] = []
        for _ in range(repeat):
            t = time.perf_counter()
            search(q, 10)
            runs.append((time.perf_counter() - t) * 1000)
        ms[name] = statistics.median(runs)
    print(
        json.dumps(
            {
                "mode": mode,
                "load_ms": load_ms,
                "mb_after_load": after_load - base,
                "mb_after_search": rss_mb() - base,
                "ms": ms,
            }
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("index", type=Path)
    p.add_argument("--repeat", type=int, default=200)
    p.add_argument("--child", choices=["lab3", "numpy", "memmap"])
    p.add_argument("--cases", help="(службове) запити для дочірніх процесів, JSON")
    a = p.parse_args()
    if a.child:
        child(a.child, a.index, [tuple(c) for c in json.loads(a.cases)], a.repeat)
        return

    index = store.load(a.index)
    cases = pick_cases(index)
    size_mb = a.index.stat().st_size / 1e6
    print(f"Індекс: {index!r}, файл {size_mb:.1f} МБ")
    for name, q in cases:
        print(f"  {name:9} {q}")
    del index
    rows: list[dict[str, Any]] = []
    for mode in ("lab3", "numpy", "memmap"):
        cmd = [sys.executable, __file__, str(a.index), "--child", mode, "--repeat", str(a.repeat)]
        cmd += ["--cases", json.dumps(cases)]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        rows.append(cast("dict[str, Any]", json.loads(out.strip().splitlines()[-1])))
    print("\n| спосіб | завантаження, мс | після завантаження, МБ | після пошуків, МБ |")
    print("| ------ | ---------------- | ---------------------- | ----------------- |")
    for r in rows:
        print(
            f"| {r['mode']} | {r['load_ms']:.0f} | {r['mb_after_load']:.1f} "
            f"| {r['mb_after_search']:.1f} |"
        )
    print("\n| спосіб | частотне слово, мс | рідкісне, мс | три слова, мс |")
    print("| ------ | ------------------ | ------------ | ------------- |")
    for r in rows:
        m = r["ms"]
        print(f"| {r['mode']} | {m['frequent']:.4f} | {m['rare']:.4f} | {m['three']:.4f} |")


if __name__ == "__main__":
    main()
