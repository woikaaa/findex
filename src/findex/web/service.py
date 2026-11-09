"""Сам пошук для веб-частини: одна функція, яку викликають і JSON-маршрут, і сторінка."""

from __future__ import annotations

import math
from dataclasses import dataclass

from findex.scoring import ScorerName
from findex.snippets import make_snippet
from findex.tokenize import TOKEN_RE
from findex.web.state import AppState


@dataclass(frozen=True)
class HitData:
    """Один результат (внутрішній dataclass; у відповідь його перетворює Pydantic)."""

    doc_id: int
    title: str
    score: float
    snippet: str


@dataclass(frozen=True)
class PageData:
    """Одна сторінка результатів."""

    total: int  # усього документів під запит
    pages: int  # скільки це сторінок
    hits: list[HitData]


def run_search(state: AppState, q: str, scorer: ScorerName, page: int, size: int) -> PageData:
    """Шукає, відбирає потрібну сторінку і робить уривки.

    Це звичайна блокувальна функція (чистий Python, процесор): тому маршрути, які її
    викликають, написані як `def`, а не `async def`. FastAPI запускає їх у пулі потоків,
    і цикл подій не зупиняється. QueryError (поганий запит) іде нагору до маршруту."""
    searcher = state.searchers[scorer]
    # скільки документів підходить усього (з кешу, якщо запит уже був)
    total = searcher.count(q)
    # беру топ до кінця потрібної сторінки, а потім відрізаю її початок
    top = searcher.search(q, k=page * size)
    chosen = top[(page - 1) * size : page * size]
    # слова запиту, які треба підсвітити
    terms = searcher.terms(q)

    hits: list[HitData] = []
    for r in chosen:
        text = state.texts.text(state.index.doc_meta[r.doc_id])
        # уривок є лише тоді, коли текст доступний
        snippet = make_snippet(text, terms, state.snippet_width) if text else ""
        hits.append(HitData(r.doc_id, r.title, r.score, snippet))
    return PageData(total=total, pages=math.ceil(total / size), hits=hits)


def mark_text(text: str, terms: frozenset[str]) -> str:
    """Виділяє слова запиту в усьому тексті (у зірочках **слово**), як в уривках."""
    out: list[str] = []
    pos = 0
    for m in TOKEN_RE.finditer(text):
        # беру тільки ті слова, що є в запиті (без урахування регістру)
        if m.group(0).casefold() in terms:
            out.append(text[pos : m.start()])
            out.append("**" + m.group(0) + "**")
            pos = m.end()
    out.append(text[pos:])
    return "".join(out)
