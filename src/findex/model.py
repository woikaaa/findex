"""
типи даних індексу.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Protocol, TypeAlias, get_args

# три способи зберігати записи; Literal не дає написати помилку в назві
Storage = Literal["dataclass", "slots", "array"]
STORAGES: tuple[Storage, ...] = get_args(Storage)


@dataclass(frozen=True, slots=True)
class Posting:
    """Пара (документ, скільки разів слово в ньому).
    frozen - можна хешувати, slots- займає менше пам'яті."""

    doc_id: int
    tf: int  # частота слова в документі
    positions: tuple[int, ...] = ()  # де саме слово стоїть (необов'язково)


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Те, що треба, щоб показати результат користувачу."""

    path: str
    title: str


@dataclass
class PlainPosting:
    """Те саме, що Posting, але звичайний dataclass (потрібен для порівняння пам'яті)."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


class PostingLike(Protocol):
    """Усе, що має doc_id, tf і positions: і Posting, і PlainPosting підходять без наслідування."""

    @property
    def doc_id(self) -> int: ...
    @property
    def tf(self) -> int: ...
    @property
    def positions(self) -> tuple[int, ...]: ...


# що лежить у словнику індексу для одного слова: список записів або пара масивів
Entry: TypeAlias = "list[Posting] | list[PlainPosting] | tuple[array[int], array[int]]"


class Index(Mapping[str, Sequence[PostingLike]]):
    def __init__(
        self,
        storage: Storage,
        has_positions: bool,
        postings: dict[str, Entry],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
    ) -> None:
        # тут я просто запам'ятовую все, що передали в клас
        self.storage = storage
        self.has_positions = has_positions
        self.postings = postings  # слово -> список записів
        self.doc_lengths = doc_lengths  # doc_id -> скільки слів у документі
        self.doc_meta = doc_meta  # doc_id -> шлях і заголовок
        self.closed = False

    # ---- протокол Mapping (що Python викликає за лаштунками для len(), `in`, [], for ...)
    def __getitem__(self, term: str) -> Sequence[PostingLike]:
        # беру запис слова зі словника
        entry = self.postings[term]  # KeyError, якщо слова немає
        if isinstance(entry, tuple):
            # розкладаю пару масивів на дві частини
            ids, tfs = entry
            return [Posting(i, t) for i, t in zip(ids, tfs, strict=True)]
        return entry

    def __iter__(self) -> Iterator[str]:
        # перебираю слова індексу
        return iter(self.postings)

    def __len__(self) -> int:
        # скільки різних слів в індексі
        return len(self.postings)

    def __contains__(self, term: object) -> bool:  # швидше, ніж стандартний у Mapping
        # просто питаю в словника, чи є таке слово
        return term in self.postings

    def __repr__(self) -> str:
        return (
            f"Index(terms={len(self):_}, docs={self.num_docs:_}, "
            f"storage={self.storage!r}, positions={self.has_positions})"
        )

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
        return len(entry[0]) if isinstance(entry, tuple) else len(entry)

    # ---- допоміжні функції з лаби 2
    def doc_ids(self, term: str) -> Sequence[int]:
        """Відсортовані doc_id, де є слово `term` (порожньо, якщо слова немає)."""
        entry = self.postings.get(term)
        if entry is None:
            return []
        if isinstance(entry, tuple):
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
