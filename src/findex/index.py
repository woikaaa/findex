"""
будує інвертований індекс за ОДИН прохід по ледачому конвеєру з лаби 1.

Індекс збирається з частин: build_partial будує частину індексу з кількох файлів,
а merge склеює частини в один Index. Так одну й ту саму роботу можна віддати
одному ядру (лаба 4) або кільком (лаба 5).
"""

from __future__ import annotations

import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from findex.corpus import Document, iter_documents, list_corpus_files, read_document
from findex.model import STORAGES, DocMeta, Entry, Index, PlainPosting, Posting, Storage
from findex.tokenize import tokenize
from findex.utils import timed

if TYPE_CHECKING:
    from findex.parallel import Executors


def _title(text: str) -> str:
    """Заголовок документа: перший непорожній рядок (до 80 символів)."""
    # шукаю перший рядок, де щось написано
    for line in text.splitlines():
        line = line.strip()
        if line:
            # обрізаю заголовок до 80 символів
            return line[:80]
    return ""


@dataclass(slots=True)
class Partial:
    """Частина індексу, зібрана з якогось шматка корпусу (її можна передати між процесами)."""

    storage: Storage
    has_positions: bool
    postings: dict[str, Entry]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    cpu_seconds: float = field(default=0.0)  # скільки процесорного часу пішло на цю частину


class _PartialBuilder:
    """Додає документи по одному й збирає з них Partial."""

    def __init__(self, storage: Storage, positions: bool) -> None:
        # перевіряю, що вибрано існуючий спосіб
        if storage not in STORAGES:
            raise ValueError(f"storage must be one of {STORAGES}")
        if storage == "array" and positions:
            raise ValueError("array storage does not support positions")
        self.storage: Storage = storage
        self.positions = positions
        # окремий словник для кожного способу зберігання, щоб типи були зрозумілі
        self._slots: dict[str, list[Posting]] = {}
        self._plain: dict[str, list[PlainPosting]] = {}
        self._arrays: dict[str, tuple[array[int], array[int]]] = {}
        self.doc_lengths: dict[int, int] = {}  # скільки слів у кожному документі
        self.doc_meta: dict[int, DocMeta] = {}  # шлях і заголовок кожного документа

    def add(self, doc_id: int, doc: Document) -> None:
        """Рахує слова одного документа й переносить їх в індекс."""
        counts: Counter[str] = Counter()  # слова лише поточного документа
        # позиції слів (якщо треба)
        offsets: defaultdict[str, list[int]] | None = defaultdict(list) if self.positions else None
        n = 0
        # іду по словах документа, i це номер слова
        for i, tok in enumerate(tokenize(doc.text)):
            # додаю одиницю до лічильника цього слова
            counts[tok] += 1
            n = i + 1  # скільки слів уже прочитали
            if offsets is not None:
                offsets[tok].append(i)

        # переношу підрахунок цього документа в індекс.
        # doc_id ростуть по порядку => кожен список уже відсортований.
        for term, tf in counts.items():
            if self.storage == "array":
                pair = self._arrays.get(term)
                if pair is None:
                    pair = (array("I"), array("I"))
                    self._arrays[term] = pair
                pair[0].append(doc_id)
                pair[1].append(tf)
                continue
            pos = tuple(offsets[term]) if offsets is not None else ()
            if self.storage == "slots":
                self._slots.setdefault(term, []).append(Posting(doc_id, tf, pos))
            else:
                self._plain.setdefault(term, []).append(PlainPosting(doc_id, tf, pos))

        # запам'ятовую довжину документа
        self.doc_lengths[doc_id] = n
        # запам'ятовую шлях і заголовок
        self.doc_meta[doc_id] = DocMeta(doc.location, _title(doc.text))

    def finish(self) -> Partial:
        postings: dict[str, Entry]
        if self.storage == "array":
            postings = dict(self._arrays)
        elif self.storage == "slots":
            postings = dict(self._slots)
        else:
            postings = dict(self._plain)
        return Partial(self.storage, self.positions, postings, self.doc_lengths, self.doc_meta)


def build_partial(
    paths: Sequence[Path],
    first_doc_id: int,
    storage: Storage = "slots",
    positions: bool = False,
    on_doc: Callable[[], None] | None = None,
) -> Partial:
    """Будує частину індексу з файлів `paths`.

    Це функція рівня модуля: її можна віддати іншому процесу. У воркер їдуть лише шляхи,
    файли він читає сам. doc_id файла = first_doc_id + його номер у списку, тому
    діапазони різних частин не перетинаються."""
    builder = _PartialBuilder(storage, positions)
    t0 = time.thread_time()  # процесорний час саме цього потоку
    for offset, path in enumerate(paths):
        doc = read_document(path)
        # зіпсований файл пропускаю, але його номер лишається зайнятим
        if doc is not None:
            builder.add(first_doc_id + offset, doc)
        if on_doc is not None:
            on_doc()
    part = builder.finish()
    part.cpu_seconds = time.thread_time() - t0
    return part


def _extend(mine: Entry, extra: Entry) -> None:
    """Дописує в кінець `mine` записи з `extra`."""
    if isinstance(mine, tuple) and isinstance(extra, tuple):
        mine[0].extend(extra[0])
        mine[1].extend(extra[1])
    elif isinstance(mine, list) and isinstance(extra, list):
        # списки одного типу, тому extend безпечний
        cast("list[Any]", mine).extend(extra)
    else:
        raise ValueError("cannot merge partials with different storage")


def merge(partials: Sequence[Partial]) -> Index:
    """Склеює частини в один Index. Частини мають іти в порядку зростання doc_id."""
    if not partials:
        raise ValueError("nothing to merge")
    first = partials[0]
    postings: dict[str, Entry] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    last_id = -1  # найбільший doc_id серед уже склеєних частин
    for part in partials:
        if part.storage != first.storage or part.has_positions != first.has_positions:
            raise ValueError("cannot merge partials built with different settings")
        # діапазони не повинні перетинатися, інакше списки перестануть бути відсортованими
        if part.doc_meta and min(part.doc_meta) <= last_id:
            raise ValueError("partials must be given in doc_id order without overlap")
        for term, entry in part.postings.items():
            mine = postings.get(term)
            if mine is None:
                postings[term] = entry  # перший раз бачу слово, беру список як є
            else:
                _extend(mine, entry)
        doc_lengths.update(part.doc_lengths)
        doc_meta.update(part.doc_meta)
        if part.doc_meta:
            last_id = max(part.doc_meta)
    return Index(first.storage, first.has_positions, postings, doc_lengths, doc_meta)


def build_index_from_docs(
    docs: Iterable[Document], storage: Storage = "slots", positions: bool = False
) -> Index:
    """Будує індекс із будь-якого потоку документів (у пам'яті росте лише сам індекс).

    Це найпростіший послідовний варіант: на ньому я перевіряю, що частинна збірка
    дає той самий індекс."""
    builder = _PartialBuilder(storage, positions)
    # номер документа це просто його порядковий номер
    for doc_id, doc in enumerate(docs):
        builder.add(doc_id, doc)
    return merge([builder.finish()])


@timed
def build_index(
    root: Path,
    storage: Storage = "slots",
    positions: bool = False,
    limit: int | None = None,
    on_doc: Callable[[], None] | None = None,
    workers: int = 1,
    executor: Executors = "serial",
) -> Index:
    """Будує індекс з папки з .txt або з файлу .jsonl.

    workers і executor (serial, threads, processes) працюють лише для папки з .txt."""
    root = Path(root)
    if root.is_dir():
        paths = list_corpus_files(root)
        if limit is not None:
            paths = paths[:limit]  # беремо лише перші `limit` файлів
        if workers == 1 and executor == "serial":
            return merge([build_partial(paths, 0, storage, positions, on_doc)])
        # імпорт тут, щоб не було циклу: parallel сам імпортує index
        from findex.parallel import EXECUTORS, build_parallel

        if executor not in EXECUTORS:
            raise ValueError(f"unknown executor: {executor}")
        return build_parallel(paths, workers, executor, storage, positions, on_doc).index
    # беру документи по одному, як у лабі 1 (так читається й .jsonl)
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)  # беремо лише перші `limit` документів
    return build_index_from_docs(docs, storage, positions)
