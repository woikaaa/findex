from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex import __version__
from findex.query import QueryError, parse_cached
from findex.scoring import ScorerName
from findex.web.deps import get_settings, get_state, get_state_or_none
from findex.web.schemas import (
    DocResponse,
    ErrorResponse,
    HealthResponse,
    Hit,
    SearchParams,
    SearchResponse,
    StatsResponse,
)
from findex.web.service import mark_text, run_search
from findex.web.settings import Settings
from findex.web.state import AppState, load_state

log = logging.getLogger("findex.web")

# id поточного запиту: потрапляє в кожен рядок логу цього запиту
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
# id від клієнта беру лише якщо він короткий і безпечний (його пишу в лог і заголовок)
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")


class _RequestIdFilter(logging.Filter):
    """Додає до кожного запису логу поле request_id."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _StderrHandler(logging.Handler):
    """Пише лог у поточний sys.stderr (а не в той, що був при створенні обробника).

    У тестах pytest підміняє stderr і потім закриває його: звичайний StreamHandler, який
    запам'ятав старий потік, видавав би помилку «I/O operation on closed file»."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
        except Exception:  # лог не має валити сервіс
            self.handleError(record)


def setup_logging(level: str) -> None:
    """Налаштовую логи findex один раз: у stderr, з id запиту в кожному рядку."""
    logger = logging.getLogger("findex")
    logger.setLevel(level.upper())
    if any(getattr(h, "_findex_web", False) for h in logger.handlers):
        return  # уже налаштовано (create_app викликають кілька разів у тестах)
    handler = _StderrHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
    )
    handler.addFilter(_RequestIdFilter())
    handler._findex_web = True  # type: ignore[attr-defined]  # мітка, щоб не додати обробник вдруге
    logger.addHandler(handler)


def highlight(snippet: str) -> Markup:
    """Перетворює **слово** на <mark>слово</mark>, решту екранує (захист від HTML у тексті)."""
    parts = snippet.split("**")
    out = [f"<mark>{escape(p)}</mark>" if i % 2 else str(escape(p)) for i, p in enumerate(parts)]
    return Markup("".join(out))


TEMPLATES.env.filters["highlight"] = highlight  # type: ignore[no-untyped-call]  # фільтри Jinja2 не типізовані


async def _load_in_background(app: FastAPI, settings: Settings) -> None:
    """Завантажує індекс у потоці, щоб сервер одразу відповідав.

    Поки індекс вантажиться, /health віддає 503."""
    try:
        app.state.findex = await asyncio.to_thread(
            load_state, settings.index_path, settings.corpus_path, settings.snippet_width
        )
    except Exception:
        # якщо файла немає або він зламаний, сервіс лишається живим, а /health віддає 503
        log.exception("не вдалося завантажити індекс з %s", settings.index_path)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Збирає застосунок. Налаштування можна передати (у тестах), інакше беруться з оточення."""
    settings = settings or Settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # lifespan: код до yield виконується один раз при старті, після yield при зупинці
        app.state.settings = settings
        app.state.findex = None
        task = asyncio.create_task(_load_in_background(app, settings))
        try:
            yield
        finally:
            task.cancel()
            log.info("сервіс зупинено")

    app = FastAPI(
        title="findex",
        version=__version__,
        description="Пошук по корпусу текстів: булеві запити, BM25 і TF-IDF.",
        lifespan=lifespan,
        # власні адреси для автодокументації, щоб /docs/{doc_id} не плутався зі Swagger
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    # на випадок, якщо lifespan ще не запускався (наприклад, TestClient без `with`)
    app.state.settings = settings
    app.state.findex = None

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Дає запиту id, міряє час і ловить необроблені винятки (відповідь 500 без стека)."""
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                # стек іде лише в лог (разом з id запиту), а клієнт бачить коротку відповідь
                log.exception("необроблений виняток: %s %s", request.method, request.url.path)
                body = ErrorResponse(detail="Внутрішня помилка сервера", request_id=rid)
                response = JSONResponse(status_code=500, content=body.model_dump())
            response.headers["X-Request-ID"] = rid
            log.info(
                "%s %s -> %d за %.1f мс",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - t0) * 1000,
            )
            return response
        finally:
            request_id_var.reset(token)

    # ------------------------------------------------------------ JSON API
    @app.get(
        "/search",
        response_model=SearchResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        summary="Пошук з ранжуванням і сторінками",
    )
    def search(  # def, а не async def: пошук це робота процесора, див. service.run_search
        params: Annotated[SearchParams, Query()],
        state: Annotated[AppState, Depends(get_state)],
    ) -> SearchResponse:
        try:
            page = run_search(state, params.q, params.scorer, params.page, params.k)
        except QueryError as exc:
            # очікувана помилка користувача (наприклад, не закрита дужка): 422, а не 500
            raise HTTPException(status_code=422, detail=f"Поганий запит: {exc}") from exc
        return SearchResponse(
            query=params.q,
            scorer=params.scorer,
            page=params.page,
            page_size=params.k,
            total=page.total,
            pages=page.pages,
            results=[Hit(doc_id=h.doc_id, title=h.title, score=h.score, snippet=h.snippet)
                     for h in page.hits],
        )  # fmt: skip

    @app.get(
        "/docs/{doc_id}",
        response_model=DocResponse,
        responses={404: {"model": ErrorResponse}},
        summary="Один документ",
    )
    def get_doc(doc_id: int, state: Annotated[AppState, Depends(get_state)]) -> DocResponse:
        meta = state.index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Документа {doc_id} немає")
        return DocResponse(
            doc_id=doc_id,
            title=meta.title,
            length=state.index.doc_length(doc_id),
            text=state.texts.text(meta),
        )

    @app.get("/stats", response_model=StatsResponse, summary="Статистика індексу")
    def stats(state: Annotated[AppState, Depends(get_state)]) -> StatsResponse:
        idx = state.index
        return StatsResponse(
            documents=idx.num_docs,
            terms=len(idx),
            storage=idx.storage,
            has_positions=idx.has_positions,
            avg_doc_length=round(idx.avg_doc_length, 2),
            uptime_s=round(state.uptime, 1),
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        summary="Перевірка для платформи (503, поки індекс не завантажений)",
    )
    async def health(
        response: Response, state: Annotated[AppState | None, Depends(get_state_or_none)]
    ) -> HealthResponse:
        # тут немає блокувальної роботи, тому async def безпечний
        if state is None:
            response.status_code = 503
            return HealthResponse(status="loading")
        return HealthResponse(status="ok", documents=state.index.num_docs)

    # ------------------------------------------------------------ сторінки (Jinja2)
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
        state: Annotated[AppState | None, Depends(get_state_or_none)],
        q: Annotated[str, Query(max_length=200)] = "",
        scorer: ScorerName = "bm25",
        page: Annotated[int, Query(ge=1, le=200)] = 1,
    ) -> Response:
        """Сторінка з полем пошуку, результатами, уривками з підсвіткою і сторінками."""
        ctx: dict[str, object] = {
            "q": q,
            "scorer": scorer,
            "page": page,
            "result": None,
            "error": None,
            "settings": settings,
            "version": __version__,
        }
        status = 200
        q = q.strip()
        if state is None:
            ctx["error"] = "Індекс ще завантажується. Оновіть сторінку за кілька секунд."
            status = 503
        elif q:
            try:
                size = settings.default_page_size
                ctx["result"] = run_search(state, q, scorer, page, size)
                ctx["size"] = size
                ctx["docs_total"] = state.index.num_docs
            except QueryError as exc:
                ctx["error"] = f"Не вдалося розібрати запит: {exc}"
                status = 422
        else:
            ctx["docs_total"] = state.index.num_docs

        def page_url(n: int) -> str:
            # посилання на іншу сторінку результатів з тим самим запитом
            return "/?" + urlencode({"q": q, "scorer": scorer, "page": n})

        ctx["page_url"] = page_url
        return TEMPLATES.TemplateResponse(request, "index.html", ctx, status_code=status)

    @app.get("/view/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def view_doc(
        request: Request,
        doc_id: int,
        settings: Annotated[Settings, Depends(get_settings)],
        state: Annotated[AppState, Depends(get_state)],
        q: Annotated[str, Query(max_length=200)] = "",
    ) -> Response:
        """Сторінка з повним текстом документа; слова запиту підсвічені."""
        meta = state.index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Документа {doc_id} немає")
        text = state.texts.text(meta)
        if text is not None and q.strip():
            try:
                text = mark_text(text, parse_cached(q).positive_terms())
            except QueryError:
                pass  # запит поганий: показую текст без підсвітки
        paragraphs = [p for p in (text or "").split("\n") if p.strip()]
        ctx: dict[str, object] = {
            "doc_id": doc_id,
            "title": meta.title,
            "paragraphs": paragraphs,
            "has_text": text is not None,
            "q": q,
            "settings": settings,
            "version": __version__,
        }
        return TEMPLATES.TemplateResponse(request, "doc.html", ctx)

    return app
