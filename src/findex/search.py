"""
булевий пошук: AND, OR, NOT двома способами (злиття двома вказівниками і множини).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from findex.model import Index
from findex.tokenize import tokenize

# який спосіб використати; Literal не дає помилитися в назві
Engine = Literal["merge", "set"]

# операція над двома відсортованими списками номерів документів
SetOp = Callable[[Sequence[int], Sequence[int]], list[int]]


# --------------------------------------------------- спосіб 1: злиття двома вказівниками
def and_merge(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Спільні номери двох відсортованих списків (AND)."""
    # ставлю обидва вказівники на початок
    i = j = 0  # два вказівники: по одному на кожен список
    # тут збиратиму результат
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1  # однакові: беремо й рухаємо обидва
        elif a[i] < b[j]:
            i += 1  # у першому менше: рухаємо перший
        else:
            j += 1  # у другому менше: рухаємо другий
    # готовий список
    return out


def or_merge(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Усі номери з обох списків без повторів (OR)."""
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    # один зі списків закінчився, решту іншого додаємо як є
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def andnot_merge(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """a AND NOT b: номери з a, яких немає в b."""
    i = j = 0
    out: list[int] = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1  # є в обох: пропускаємо
        else:
            j += 1
    return out


# --------------------------------------------------- спосіб 2: множини Python
def and_set(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """AND через перетин множин."""
    return sorted(set(a) & set(b))


def or_set(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """OR через об'єднання множин."""
    return sorted(set(a) | set(b))


def andnot_set(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """AND NOT через різницю множин."""
    return sorted(set(a) - set(b))


# назва способу -> три його функції (AND, OR, AND NOT)
ENGINES: dict[Engine, tuple[SetOp, SetOp, SetOp]] = {
    "merge": (and_merge, or_merge, andnot_merge),
    "set": (and_set, or_set, andnot_set),
}


def evaluate(index: Index, query: str, engine: Engine = "merge") -> list[int]:
    """Обчислює запит зліва направо. Повертає відсортовані doc_id."""
    # беру три функції вибраного способу
    f_and, f_or, f_andnot = ENGINES[engine]
    result: list[int] | None = None  # поточний результат
    op = "AND"  # який оператор зараз діє (за замовчуванням AND)
    negate = False  # чи стояв перед словом NOT

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
