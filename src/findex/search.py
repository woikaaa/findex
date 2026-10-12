from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

from findex import store
from findex.model import Index
from findex.tokenize import tokenize


# --------------------------------------------------- спосіб 1: злиття двома вказівниками
def and_merge(a, b) -> list[int]:
    """Спільні номери двох відсортованих списків (AND)."""
    # ставлю обидва вказівники на початок
    i = j = 0  # два вказівники: по одному на кожен список
    # тут збиратиму результат
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1  # однакові: беремо й рухаємо обидва
        elif a[i] < b[j]:
            i += 1  # у першому менше: рухаємо перший
        else:
            j += 1  # у другому менше: рухаємо другий
    # готовий список
    return out


def or_merge(a, b) -> list[int]:
    """Усі номери з обох списків без повторів (OR)."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    # один зі списків закінчився, решту іншого додаємо як є
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def andnot_merge(a, b) -> list[int]:
    """a AND NOT b: номери з a, яких немає в b."""
    i = j = 0
    out = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i]); i += 1
        elif a[i] == b[j]:
            i += 1; j += 1  # є в обох: пропускаємо
        else:
            j += 1
    return out


# --------------------------------------------------- спосіб 2: множини Python
def and_set(a, b) -> list[int]:
    """AND через перетин множин."""
    return sorted(set(a) & set(b))


def or_set(a, b) -> list[int]:
    """OR через об'єднання множин."""
    return sorted(set(a) | set(b))


def andnot_set(a, b) -> list[int]:
    """AND NOT через різницю множин."""
    return sorted(set(a) - set(b))


# назва способу -> три його функції (AND, OR, AND NOT)
ENGINES = {
    "merge": (and_merge, or_merge, andnot_merge),
    "set": (and_set, or_set, andnot_set),
}


def evaluate(index: Index, query: str, engine: str = "merge") -> list[int]:
    """Обчислює запит зліва направо. Повертає відсортовані doc_id."""
    # беру три функції вибраного способу
    f_and, f_or, f_andnot = ENGINES[engine]
    result = None      # поточний результат
    op = "AND"         # який оператор зараз діє (за замовчуванням AND)
    negate = False     # чи стояв перед словом NOT

    # ділю запит на слова по пробілах
    for word in query.split():
        # якщо це оператор, запам'ятовую його
        if word in ("AND", "OR"):
            op = word
            continue
        # NOT діє на наступне слово
        if word == "NOT":
            negate = True
            continue
        # слово проходить ту саму нормалізацію, що й при побудові індексу
        for term in tokenize(word):
            # дізнаюсь, у яких документах є це слово
            ids = index.doc_ids(term)
            if result is None:
                # перше слово запиту
                result = f_andnot(index.all_doc_ids(), ids) if negate else list(ids)
            elif op == "AND":
                result = f_andnot(result, ids) if negate else f_and(result, ids)
            else:  # OR
                if negate:
                    ids = f_andnot(index.all_doc_ids(), ids)
                result = f_or(result, ids)
            op, negate = "AND", False  # після слова повертаємо стандартні значення
    # якщо нічого не знайшлось, повертаю порожній список
    return result or []


def main() -> None:
    # аргументи командного рядка
    p = argparse.ArgumentParser(description="search a saved findex index")
    p.add_argument("index", type=Path)
    p.add_argument("query")
    p.add_argument("--engine", choices=tuple(ENGINES), default="merge")
    p.add_argument("--max", type=int, default=10, help="max results to print")
    a = p.parse_args()

    # вимірюємо завантаження, пошук і пам'ять окремо
    tracemalloc.start()
    t0 = time.perf_counter()
    # завантажую збережений індекс
    index = store.load(a.index)
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    # виконую сам пошук
    hits = evaluate(index, a.query, a.engine)
    search_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"{len(hits)} result(s) for {a.query!r}  [engine={a.engine}]")
    # показую лише перші результати
    for doc_id in hits[: a.max]:
        m = index.doc_meta[doc_id]
        print(f"  #{doc_id:<4} {m.path}  —  {m.title}")
    if len(hits) > a.max:
        print(f"  ... and {len(hits) - a.max} more")
    print(f"load: {load_time:.4f} s   search: {search_time * 1000:.3f} ms   "
          f"peak memory: {peak / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
