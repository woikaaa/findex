"""
будує інвертований індекс за ОДИН прохід по ледачому конвеєру з лаби 1 і зберігає його у файл.

    python -m findex.index data/ --out index.bin
    python -m findex.index data/ --out index.json.bin --format binary
    python -m findex.index data/ --out index.bin --positions
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.model import DocMeta, Index, PlainPosting, Posting
from findex.tokenize import tokenize
from findex.utils import timed

# три способи зберігати записи, які ми порівнюємо за пам'яттю
STORAGES = ("dataclass", "slots", "array")


def _title(text: str) -> str:
    """Заголовок документа: перший непорожній рядок (до 80 символів)."""
    # шукаю перший рядок, де щось написано
    for line in text.splitlines():
        line = line.strip()
        if line:
            # обрізаю заголовок до 80 символів
            return line[:80]
    return ""


@timed
def build_index(root: Path, storage: str = "slots", positions: bool = False,
                limit: int | None = None) -> Index:
    """Читає документи по одному, у пам'яті росте лише сам індекс."""
    # перевіряю, що вибрано існуючий спосіб
    if storage not in STORAGES:
        raise ValueError(f"storage must be one of {STORAGES}")
    if storage == "array" and positions:
        raise ValueError("array storage does not support positions")

    # беру документи по одному, як у лабі 1
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)  # беремо лише перші `limit` документів

    # defaultdict сам створює порожній запис для нового слова,
    # тому не треба перевіряти "if term not in postings".
    if storage == "array":
        postings = defaultdict(lambda: (array("I"), array("I")))
    else:
        postings = defaultdict(list)
    # який клас використовувати для запису: зі slots чи звичайний
    make = Posting if storage == "slots" else PlainPosting

    doc_lengths: dict[int, int] = {}   # скільки слів у кожному документі
    doc_meta: dict[int, DocMeta] = {}  # шлях і заголовок кожного документа

    # номер документа це просто його порядковий номер
    for doc_id, doc in enumerate(docs):
        counts: Counter[str] = Counter()          # слова лише поточного документа
        offsets = defaultdict(list) if positions else None  # позиції слів (якщо треба)
        n = 0
        # іду по словах документа, i це номер слова
        for i, tok in enumerate(tokenize(doc.text)):
            # додаю одиницю до лічильника цього слова
            counts[tok] += 1
            n = i + 1  # скільки слів уже прочитали
            if offsets is not None:
                offsets[tok].append(i)

        # переносимо підрахунок цього документа в індекс.
        # doc_id ростуть по порядку => кожен список уже відсортований.
        for term, tf in counts.items():
            if storage == "array":
                ids, tfs = postings[term]
                ids.append(doc_id)
                tfs.append(tf)
            else:
                pos = tuple(offsets[term]) if offsets is not None else ()
                postings[term].append(make(doc_id, tf, pos))

        # запам'ятовую довжину документа
        doc_lengths[doc_id] = n
        # запам'ятовую шлях і заголовок
        doc_meta[doc_id] = DocMeta(str(doc.path), _title(doc.text))

    # dict(...) прибирає defaultdict (lambda не можна зберегти через pickle)
    return Index(storage, positions, dict(postings), doc_lengths, doc_meta)


def main() -> None:
    # імпортую тут, щоб не було циклічного імпорту
    from findex import store

    # аргументи командного рядка
    p = argparse.ArgumentParser(description="build and save the inverted index")
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--format", choices=("pickle", "binary"), default="pickle")
    p.add_argument("--storage", choices=STORAGES, default="slots")
    p.add_argument("--positions", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    # зчитую прапорці з терміналу
    a = p.parse_args()

    storage = a.storage
    if a.format == "binary" and storage != "array":
        storage = "array"   # бінарний формат працює лише з array
        if a.positions:
            p.error("--positions is not supported by the binary format")

    # вимірюємо час і піковий обсяг пам'яті при побудові
    tracemalloc.start()
    t0 = time.perf_counter()
    # будую індекс
    index = build_index(a.root, storage, a.positions, a.limit)
    build_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # зберігаємо індекс у файл і міряємо час збереження
    t0 = time.perf_counter()
    # зберігаю індекс у файл
    store.save(index, a.out, a.format)
    save_time = time.perf_counter() - t0

    # друкую підсумок
    print(f"documents:    {len(index.doc_meta)}")
    print(f"terms:        {len(index.postings)}")
    print(f"storage:      {index.storage}   positions: {index.has_positions}")
    print(f"build time:   {build_time:.3f} s")
    print(f"peak memory:  {peak / 1024:.1f} KiB (build)")
    print(f"saved:        {a.out} ({a.format}, {a.out.stat().st_size} bytes, {save_time:.3f} s)")


if __name__ == "__main__":
    main()
