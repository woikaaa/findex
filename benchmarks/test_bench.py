"""Бенчмарки pytest-benchmark: пошук (лаба 3 проти NumPy) і збірка індексу.

    pytest benchmarks --benchmark-only
    pytest benchmarks --benchmark-only --corpus data\\wiki --benchmark-json=docs\\bench_wiki.json
    python benchmarks\\summary.py docs\\bench_wiki.json

Кеш запитів прогрітий (перший виклик у calibration), тому міряється саме підрахунок балів і
вибір топ-k, а не розбір запиту. Однаково для обох версій.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.index import build_index
from findex.model import Index
from findex.npsearch import NpIndex, NpSearcher
from findex.ranking import Searcher
from findex.snippets import make_snippet


@pytest.mark.parametrize("case", ["frequent", "rare", "three"])
def test_search_lab3(
    benchmark: BenchmarkFixture, lab3_searcher: Searcher, cases: dict[str, str], case: str
) -> None:
    benchmark.group = f"search-{case}"
    benchmark.extra_info["query"] = cases[case]
    benchmark(lab3_searcher.search, cases[case], 10)


@pytest.mark.parametrize("case", ["frequent", "rare", "three"])
def test_search_numpy(
    benchmark: BenchmarkFixture, numpy_searcher: NpSearcher, cases: dict[str, str], case: str
) -> None:
    benchmark.group = f"search-{case}"
    benchmark.extra_info["query"] = cases[case]
    benchmark(numpy_searcher.search, cases[case], 10)


def test_build_index(benchmark: BenchmarkFixture, corpus_dir: Path) -> None:
    benchmark.group = "build"
    benchmark.pedantic(
        build_index, args=(corpus_dir,), kwargs={"storage": "array"}, rounds=3, iterations=1
    )


def test_build_numpy_index(benchmark: BenchmarkFixture, index: Index) -> None:
    """Скільки коштує переклад готового індексу в масиви NumPy."""
    benchmark.group = "build"
    benchmark.pedantic(NpIndex.from_index, args=(index,), rounds=5, iterations=1)


def test_snippet(benchmark: BenchmarkFixture, corpus_dir: Path, cases: dict[str, str]) -> None:
    """Уривок із найбільшого файла для частотного слова."""
    biggest = max(corpus_dir.rglob("*.txt"), key=lambda p: p.stat().st_size)
    text = biggest.read_text(encoding="utf-8")
    benchmark.group = "snippet"
    benchmark(make_snippet, text, {cases["frequent"]}, 90)
