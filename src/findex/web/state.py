"""Те, що сервіс тримає в пам'яті: завантажений індекс, тексти документів, оцінювачі."""

from __future__ import annotations

import logging
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import cast

from findex.model import DocMeta, Index
from findex.ranking import Searcher
from findex.scoring import SCORER_NAMES, ScorerName, make_scorer
from findex.store import load

log = logging.getLogger("findex.web")


class DocTexts:
    """Дістає текст документа з корпусу: або з папки з .txt, або з .jsonl від краулера."""

    def __init__(self, root: Path | None) -> None:
        # ім'я файлу -> повний шлях (для папки з .txt)
        self._files: dict[str, Path] = {}
        # id запису (адреса) -> текст (для .jsonl)
        self._records: dict[str, str] = {}
        self.available = False
        if root is None or not root.exists():
            return  # корпусу немає: працюю без уривків
        if root.is_dir():
            # запам'ятовую файли за іменем: в індексі шлях міг бути записаний на іншій машині
            self._files = {_key(p.name): p for p in root.rglob("*.txt")}
        elif root.suffix == ".jsonl":
            self._records = _read_jsonl(root)
        self.available = bool(self._files or self._records)

    def text(self, meta: DocMeta) -> str | None:
        """Текст документа або None, якщо його немає.

        Читання файла блокувальне, тому викликати з маршруту `def`."""
        if meta.path in self._records:
            return self._records[meta.path]
        found = self._files.get(_key(_file_name(meta.path)))
        if found is None:
            return None
        return found.read_text(encoding="utf-8", errors="replace")


def _file_name(path: str) -> str:
    """Ім'я файла з рядка-шляху, записаного і на Windows (a\\b\\c.txt), і на Linux (a/b/c.txt).

    Індекс могли побудувати на Windows, а сервіс запустити в Linux-контейнері. Звичайний
    Path там не вважає зворотну скісну риску роздільником і віддав би весь рядок цілком.
    PureWindowsPath розуміє обидва види."""
    return PureWindowsPath(path).name


def _key(name: str) -> str:
    """Ключ для порівняння імен: Юнікод в єдиній формі (на macOS «й» і «ї» бувають розкладені)."""
    return unicodedata.normalize("NFC", name)


def _read_jsonl(path: Path) -> dict[str, str]:
    """Читає JSONL від findex crawl: рядок за рядком, id -> text."""
    import json

    out: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec: object = json.loads(line)
            except json.JSONDecodeError:
                continue  # зіпсований рядок пропускаю
            if isinstance(rec, dict):
                d = cast("dict[str, object]", rec)
                key, text = d.get("id"), d.get("text")
                if isinstance(key, str) and isinstance(text, str):
                    out[key] = text
    return out


@dataclass
class AppState:
    """Усе, що живе, поки працює сервіс. Створюється один раз у lifespan."""

    index: Index
    texts: DocTexts
    snippet_width: int = 90
    started: float = field(default_factory=time.monotonic)
    # по одному Searcher на кожну формулу (у кожного свій кеш запитів)
    searchers: dict[ScorerName, Searcher] = field(init=False)

    def __post_init__(self) -> None:
        # створюю оцінювачі наперед, щоб не робити це під час запиту
        self.searchers = {name: Searcher(self.index, make_scorer(name)) for name in SCORER_NAMES}

    @property
    def uptime(self) -> float:
        """Скільки секунд сервіс працює."""
        return time.monotonic() - self.started


def load_state(index_path: Path, corpus_path: Path | None, snippet_width: int) -> AppState:
    """Завантажує індекс із файла і готує тексти (блокувальна робота: викликати в потоці)."""
    t0 = time.perf_counter()
    index = load(index_path)
    texts = DocTexts(corpus_path)
    log.info(
        "індекс завантажено: %s (%d документів, %d слів) за %.2f с; тексти: %s",
        index_path,
        index.num_docs,
        len(index),
        time.perf_counter() - t0,
        "є" if texts.available else "немає",
    )
    return AppState(index, texts, snippet_width)
