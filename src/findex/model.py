"""
типи даних індексу.

`python -m findex.index` цей модуль називався б `__main__`, і потім
`python -m findex.search` не зміг би прочитати файл.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True, slots=True)
class Posting:
    """Пара (документ, скільки разів слово в ньому).
    frozen - можна хешувати, slots- займає менше пам'яті."""
    doc_id: int
    tf: int                            # частота слова в документі
    positions: tuple[int, ...] = ()    # де саме слово стоїть (необов'язково)


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Те, що треба, щоб показати результат користувачу."""
    path: str
    title: str


@dataclass
class PlainPosting:
    """Те саме, що Posting, але звичайний dataclass (має __dict__).
    Існує ЛИШЕ для порівняння пам'яті в memstudy."""
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


class Index(Mapping):
    def __init__(self, storage: str, has_positions: bool, postings: dict,
                 doc_lengths: dict[int, int], doc_meta: dict[int, DocMeta]) -> None:
        # тут я просто запам'ятовую все, що передали в клас
        self.storage = storage
        self.has_positions = has_positions
        self.postings = postings        # слово -> список записів
        self.doc_lengths = doc_lengths  # doc_id -> скільки слів у документі
        self.doc_meta = doc_meta        # doc_id -> шлях і заголовок
        self.closed = False

    # ---- протокол Mapping (що Python викликає за лаштунками для len(), `in`, [], for ...)
    def __getitem__(self, term: str):
        # беру запис слова зі словника
        entry = self.postings[term]            # KeyError, якщо слова немає
        if self.storage == "array":
            # розкладаю пару масивів на дві частини
            ids, tfs = entry
            return [Posting(i, t) for i, t in zip(ids, tfs)]
        return entry

    def __iter__(self):
        # перебираю слова індексу
        return iter(self.postings)

    def __len__(self) -> int:
        # скільки різних слів в індексі
        return len(self.postings)

    def __contains__(self, term: object) -> bool:   # швидше, ніж стандартний у Mapping
        # просто питаю в словника, чи є таке слово
        return term in self.postings

    def __repr__(self) -> str:
        return (f"Index(terms={len(self):_}, docs={self.num_docs:_}, "
                f"storage={self.storage!r}, positions={self.has_positions})")

    # ---- обчислювані значення
    @property
    def num_docs(self) -> int:
        """Скільки всього документів."""
        # скільки документів, стільки й довжин
        return len(self.doc_lengths)

    @cached_property
    def avg_doc_length(self) -> float:
        """Рахується один раз і запам'ятовується (BM25 потрібна вона для кожного документа)."""
        return sum(self.doc_lengths.values()) / self.num_docs if self.doc_lengths else 0.0

    def doc_length(self, doc_id: int) -> int:
        """Скільки слів у документі."""
        # довжина конкретного документа
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        """У скількох документах зустрічається слово."""
        # get не падає, якщо слова немає
        entry = self.postings.get(term)
        if entry is None:
            return 0
        return len(entry[0]) if self.storage == "array" else len(entry)

    # ---- допоміжні функції з лаби 2
    def doc_ids(self, term: str):
        """Відсортовані doc_id, де є слово `term` (порожньо, якщо слова немає)."""
        entry = self.postings.get(term)
        if entry is None:
            return []
        if self.storage == "array":
            return entry[0]
        return [p.doc_id for p in entry]

    def all_doc_ids(self) -> list[int]:
        """Номери всіх документів."""
        # документи йдуть по порядку
        return sorted(self.doc_meta)

    def close(self) -> None:
        """Звільняє великі структури (викликається в open_index, навіть при помилці)."""
        # порожніми словниками звільняю пам'ять
        self.postings = {}
        self.doc_lengths = {}
        self.doc_meta = {}
        self.__dict__.pop("avg_doc_length", None)  # прибираємо закешоване значення
        self.closed = True
