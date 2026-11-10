"""
profile_run.py: профілювання пошуку і збірки індексу (лаба 8, крок 1: ДО змін коду).

    python profile_run.py search data/index.bin --repeat 300
    python profile_run.py page   data/index.bin --repeat 300
    python profile_run.py build  data/wiki
    python profile_run.py search data/index.bin --repeat 3000 --plain     # для py-spy / Scalene
    python profile_run.py search data/index.bin --engine numpy            # ПІСЛЯ змін (NumPy)

search: сам пошук (розбір запиту, відбір, бали, топ-10) для трьох запитів:
        частотне слово, рідкісне слово, три слова.
page:   те саме + уривки для топ-10, як на сайті (результат сторінки).
build:  будую індекс з папки з .txt.
Результат cProfile пишу в docs/ (текст і файл .prof).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

from findex import store
from findex.bench_cases import pick_cases
from findex.index import build_index
from findex.model import Index
from findex.npsearch import NpIndex, NpSearcher
from findex.ranking import Searcher
from findex.snippets import snippet_for


def load_or_build(source: Path) -> Index:
    """Файл індексу завантажую, папку з .txt будую."""
    if source.is_dir():
        return build_index(source)
    return store.load(source)


def run_queries(
    searcher: Searcher | NpSearcher, index: Index, queries: list[str], repeat: int, snippets: bool
) -> int:
    """Повторює кожен запит `repeat` разів. Повертає кількість запусків.

    snippets=True додає уривки для топ-10 (як на сайті)."""
    runs = 0
    for _ in range(repeat):
        for q in queries:
            # скільки всього підходить (кеш запиту прогріється після першого разу)
            searcher.count(q)
            top = searcher.search(q, k=10)
            terms = searcher.terms(q)
            if snippets:
                # як на сайті: до кожного результату з топ-10 беру уривок
                for r in top:
                    snippet_for(index, r.doc_id, terms)
            runs += 1
    return runs


def report(profiler: cProfile.Profile, title: str, out: Path, prof_out: Path, top: int) -> None:
    """Записує дві таблиці (за власним часом і за сумарним) у текстовий файл."""
    out.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(prof_out))
    buf = io.StringIO()
    buf.write(f"# {title}\n\n")
    sections = (("tottime", "за власним часом функції"), ("cumulative", "за сумарним часом"))
    for key, label in sections:
        buf.write(f"## Топ-{top} {label}\n\n")
        stats = pstats.Stats(profiler, stream=buf)
        stats.strip_dirs().sort_stats(key).print_stats(top)
    out.write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())
    print(f"Записано: {out} і {prof_out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["search", "page", "build"])
    p.add_argument("source", type=Path)
    p.add_argument("--repeat", type=int, default=300)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--engine", choices=["lab3", "numpy"], default="lab3", help="чим шукати")
    p.add_argument("--plain", action="store_true", help="без cProfile (для py-spy і Scalene)")
    p.add_argument("--out-dir", type=Path, default=Path("docs"))
    a = p.parse_args()

    if a.mode == "build":
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        if a.plain:
            index = build_index(a.source)
        else:
            index = pr.runcall(build_index, a.source)
        print(f"{index!r}\nзбірка: {time.perf_counter() - t0:.2f} с")
        if not a.plain:
            report(
                pr,
                f"cProfile: build_index({a.source})",
                a.out_dir / "profile_build_before.txt",
                a.out_dir / "build_before.prof",
                a.top,
            )
        return

    index = load_or_build(a.source)
    searcher: Searcher | NpSearcher
    searcher = NpSearcher(NpIndex.from_index(index)) if a.engine == "numpy" else Searcher(index)
    phase = "after" if a.engine == "numpy" else "before"  # у назвах файлів: до/після змін
    cases = pick_cases(index)
    for name, q in cases:
        print(f"{name:9} df={index.df(q) if ' ' not in q else '-':>5}  запит: {q}")
    queries = [q for _, q in cases]
    with_snippets = a.mode == "page"
    run_queries(searcher, index, queries, 1, with_snippets)  # прогрів: перший запуск не міряю
    t0 = time.perf_counter()
    if a.plain:
        runs = run_queries(searcher, index, queries, a.repeat, with_snippets)
    else:
        pr = cProfile.Profile()
        runs = pr.runcall(run_queries, searcher, index, queries, a.repeat, with_snippets)
    dt = time.perf_counter() - t0
    print(f"\n{runs} пошуків за {dt:.2f} с = {dt / runs * 1000:.2f} мс на пошук")
    if not a.plain:
        report(
            pr,
            f"cProfile: {a.mode}, {a.engine} ({a.source}), {runs} пошуків",
            a.out_dir / f"profile_{a.mode}_{phase}.txt",
            a.out_dir / f"{a.mode}_{phase}.prof",
            a.top,
        )


if __name__ == "__main__":
    main()
