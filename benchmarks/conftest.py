"""Спільне для бенчмарків (лаба 8).

За замовчуванням бенчмарки працюють на синтетичному корпусі (слова за законом Ципфа), щоб
у CI результат не залежав від скачаних текстів. Свій корпус: --corpus data\\wiki
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from findex.bench_cases import pick_cases
from findex.index import build_index
from findex.model import Index
from findex.npsearch import NpIndex, NpSearcher
from findex.ranking import Searcher


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--corpus", default=None, help="папка з .txt (замість синтетичного корпусу)")
    parser.addoption("--bench-docs", type=int, default=500, help="скільки документів синтетичних")


def _write_synthetic(root: Path, n_docs: int, words: int = 300, vocab: int = 5000) -> None:
    """Пише n_docs текстів, слова вибираю з вагами 1/номер (закон Ципфа, як у справжніх текстах)."""
    rnd = random.Random(7)
    pool = [f"слово{i}" for i in range(vocab)]
    weights = [1 / (i + 1) for i in range(vocab)]
    for i in range(n_docs):
        text = " ".join(rnd.choices(pool, weights, k=words))
        (root / f"doc{i:05d}.txt").write_text(f"Документ {i}\n{text}\n", encoding="utf-8")


@pytest.fixture(scope="session")
def corpus_dir(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    given = request.config.getoption("--corpus")
    if given:
        return Path(str(given))
    root = tmp_path_factory.mktemp("synthetic_corpus")
    _write_synthetic(root, int(request.config.getoption("--bench-docs")))
    return root


@pytest.fixture(scope="session")
def index(corpus_dir: Path) -> Index:
    # array - так індекс зберігає сервіс і бінарний файл
    return build_index(corpus_dir, storage="array")


@pytest.fixture(scope="session")
def np_index(index: Index) -> NpIndex:
    return NpIndex.from_index(index)


@pytest.fixture(scope="session")
def cases(index: Index) -> dict[str, str]:
    """Назва випадку -> запит (частотне слово, рідкісне слово, три слова)."""
    return dict(pick_cases(index))


@pytest.fixture
def lab3_searcher(index: Index) -> Searcher:
    return Searcher(index)


@pytest.fixture
def numpy_searcher(np_index: NpIndex) -> NpSearcher:
    return NpSearcher(np_index)
