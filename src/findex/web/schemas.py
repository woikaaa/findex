"""Моделі Pydantic. Вони стоять лише на межі: перевіряють вхід і описують вихід.

Усередині програми лишаються звичайні dataclass-и з лаб 2–3 (SearchResult, Index)."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from findex.scoring import ScorerName


class SearchParams(BaseModel):
    """Параметри GET /search. Pydantic сам перевіряє типи й межі та дає відповідь 422."""

    # зайві параметри в адресі вважаю помилкою, щоб одруківка не пройшла непоміченою
    model_config = ConfigDict(extra="forbid")

    q: Annotated[str, Field(min_length=1, max_length=200, description="Запит")]
    # k це розмір сторінки (скільки результатів показати), як у CLI
    k: Annotated[int, Field(ge=1, le=50, description="Результатів на сторінці")] = 10
    scorer: Annotated[ScorerName, Field(description="Формула ранжування")] = "bm25"
    page: Annotated[int, Field(ge=1, le=200, description="Номер сторінки, з 1")] = 1


class Hit(BaseModel):
    """Один результат пошуку."""

    doc_id: int
    title: str
    score: float
    # уривок із виділеними словами у зірочках (**слово**); порожній, якщо тексту немає
    snippet: str


class SearchResponse(BaseModel):
    """Відповідь GET /search."""

    query: str
    scorer: ScorerName
    page: int
    page_size: int
    # скільки документів підійшло під запит усього (не лише на цій сторінці)
    total: int
    pages: int
    results: list[Hit]


class DocResponse(BaseModel):
    """Відповідь GET /docs/{doc_id}."""

    doc_id: int
    title: str
    length: int  # скільки слів у документі
    text: str | None  # повний текст (None, якщо корпус недоступний)


class StatsResponse(BaseModel):
    """Відповідь GET /stats."""

    documents: int
    terms: int
    storage: str
    has_positions: bool
    avg_doc_length: float
    uptime_s: float


class HealthResponse(BaseModel):
    """Відповідь GET /health."""

    status: str
    documents: int | None = None


class ErrorResponse(BaseModel):
    """Тіло помилки. Для 500 тут є request_id, за яким у логах знайдуть стек."""

    detail: str
    request_id: str | None = None
