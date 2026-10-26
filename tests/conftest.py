"""спільні фікстури для всіх тестів."""

from __future__ import annotations

from pathlib import Path

import pytest

from findex.index import build_index
from findex.model import Index

# маленький набір текстів, що лежить у репозиторії (13 файлів)
DATA = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    """Папка з текстами для тестів."""
    return DATA


@pytest.fixture(scope="session")
def index(corpus_dir: Path) -> Index:
    """Індекс з позиціями, збудований один раз на всю сесію (тести його лише читають)."""
    return build_index(corpus_dir, "slots", positions=True)


@pytest.fixture
def saved_index(index: Index, tmp_path: Path) -> Path:
    """Файл з індексом у тимчасовій папці (для тестів завантаження і CLI)."""
    from findex.store import save

    path = tmp_path / "i.pkl"
    save(index, path, "pickle")
    return path
