"""Тести веб-частини (лаба 7).

Усі тести працюють через TestClient і app.dependency_overrides: справжнього файлу індексу
для них не треба (його підміняє індекс із conftest). Виняток: тест lifespan, там перевіряю,
що застосунок справді завантажує файл за Settings.index_path."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app as cli_app
from findex.model import Index
from findex.web import app as web_app_module
from findex.web.app import create_app, highlight
from findex.web.deps import get_state, get_state_or_none
from findex.web.settings import Settings
from findex.web.state import AppState, DocTexts


@pytest.fixture(autouse=True)
def _quiet_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Прибирає з кореневого логера чужі потокові обробники.

    Команди CLI з попередніх тестів (CliRunner) через logging.basicConfig додають обробник
    на свій тимчасовий stderr. Він або вже закритий, або засмічує вивід pytest очікуваними
    помилками (наприклад, «не вдалося завантажити індекс»). Обробник pytest (caplog) лишається.
    Сам сервіс у тестах мовчить (LOG_LEVEL=CRITICAL); тест 500 вмикає рівень через caplog."""
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
    root = logging.getLogger()
    for h in list(root.handlers):
        if type(h) is logging.StreamHandler:  # саме звичайний, не підклас pytest
            root.removeHandler(h)


@pytest.fixture
def state(index: Index, corpus_dir: Path) -> AppState:
    """Готовий стан сервісу з маленьким індексом (без жодного файлу індексу)."""
    return AppState(index, DocTexts(corpus_dir), snippet_width=60)


@pytest.fixture
def client(state: AppState) -> Iterator[TestClient]:
    """Клієнт із підміненою залежністю: сервіс ніби вже завантажив індекс."""
    app = create_app(Settings(index_path=Path("немає.bin"), corpus_path=None))
    app.dependency_overrides[get_state_or_none] = lambda: state
    # get_state бере результат get_state_or_none, тому окремо його підміняти не треба
    with TestClient(app) as c:
        yield c


@pytest.fixture
def empty_client() -> Iterator[TestClient]:
    """Клієнт без жодної підміни: файлу індексу немає, тож індекс не завантажений."""
    app = create_app(Settings(index_path=Path("немає.bin"), corpus_path=None))
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- GET /search
def test_search_ok(client: TestClient) -> None:
    res = client.get("/search", params={"q": "rabbit"})
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "rabbit" and data["scorer"] == "bm25" and data["page"] == 1
    assert data["total"] > 0 and data["results"]
    first = data["results"][0]
    assert {"doc_id", "title", "score", "snippet"} <= set(first)
    # уривок із підсвіткою слова запиту (**слово**)
    assert "**" in first["snippet"]


def test_search_results_sorted_by_score(client: TestClient) -> None:
    scores = [
        r["score"] for r in client.get("/search", params={"q": "alice OR rabbit"}).json()["results"]
    ]
    assert scores == sorted(scores, reverse=True)


def test_search_pagination_pages_do_not_overlap(client: TestClient) -> None:
    p1 = client.get("/search", params={"q": "alice", "k": 2, "page": 1}).json()
    p2 = client.get("/search", params={"q": "alice", "k": 2, "page": 2}).json()
    ids1 = {r["doc_id"] for r in p1["results"]}
    ids2 = {r["doc_id"] for r in p2["results"]}
    assert len(p1["results"]) == 2 and ids1.isdisjoint(ids2)
    # усього сторінок: стеля від ділення total на розмір сторінки
    assert p1["pages"] == -(-p1["total"] // 2)


def test_search_page_beyond_end_is_empty_not_error(client: TestClient) -> None:
    data = client.get("/search", params={"q": "alice", "page": 200}).json()
    assert data["results"] == [] and data["total"] > 0


def test_search_tfidf_scorer(client: TestClient) -> None:
    res = client.get("/search", params={"q": "alice OR rabbit", "scorer": "tfidf"})
    assert res.status_code == 200 and res.json()["scorer"] == "tfidf"


@pytest.mark.parametrize(
    "params",
    [
        {},  # немає q
        {"q": ""},  # порожній q
        {"q": "x" * 201},  # задовгий q
        {"q": "alice", "k": 0},
        {"q": "alice", "k": 51},
        {"q": "alice", "page": 0},
        {"q": "alice", "scorer": "magic"},
        {"q": "alice", "bogus": "1"},  # зайвий параметр
    ],
)
def test_search_validation_422(client: TestClient, params: dict[str, object]) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_search_bad_query_syntax_is_422_not_500(client: TestClient) -> None:
    res = client.get("/search", params={"q": "alice AND"})
    assert res.status_code == 422 and "Поганий запит" in res.json()["detail"]


# ---------------------------------------------------------------- GET /docs/{id}
def test_get_doc_ok(client: TestClient) -> None:
    hit = client.get("/search", params={"q": "rabbit"}).json()["results"][0]
    res = client.get(f"/docs/{hit['doc_id']}")
    assert res.status_code == 200
    data = res.json()
    assert data["title"] == hit["title"] and data["length"] > 0 and data["text"]


def test_get_doc_unknown_is_404(client: TestClient) -> None:
    res = client.get("/docs/999999")
    assert res.status_code == 404 and "detail" in res.json()


# ---------------------------------------------------------------- /stats і /health
def test_stats(client: TestClient, index: Index) -> None:
    data = client.get("/stats").json()
    assert data["documents"] == index.num_docs and data["terms"] == len(index)
    assert data["storage"] == "slots" and data["has_positions"] is True


def test_health_ok_when_index_loaded(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200 and res.json()["status"] == "ok"


def test_health_503_when_index_not_loaded(empty_client: TestClient) -> None:
    res = empty_client.get("/health")
    assert res.status_code == 503 and res.json()["status"] == "loading"


def test_search_503_when_index_not_loaded(empty_client: TestClient) -> None:
    assert empty_client.get("/search", params={"q": "alice"}).status_code == 503


# ---------------------------------------------------------------- необроблений виняток
def test_unexpected_error_gives_500_with_request_id_and_no_stack(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("секретна внутрішня деталь")

    monkeypatch.setattr("findex.web.app.run_search", boom)
    with caplog.at_level(logging.ERROR, logger="findex.web"):
        res = client.get("/search", params={"q": "alice"}, headers={"X-Request-ID": "test-id-42"})
    body = res.json()
    assert res.status_code == 500
    assert body["request_id"] == "test-id-42" and res.headers["X-Request-ID"] == "test-id-42"
    # назовні немає ні стека, ні тексту винятку
    assert "секретна" not in res.text and "Traceback" not in res.text
    # у лозі стек є, і в рядку видно той самий id
    assert any("секретна" in (r.exc_text or "") for r in caplog.records)
    assert any(getattr(r, "request_id", "") == "test-id-42" for r in caplog.records)


def test_request_id_is_generated_and_unsafe_one_is_replaced(client: TestClient) -> None:
    generated = client.get("/health").headers["X-Request-ID"]
    assert generated and generated != "-"
    replaced = client.get("/health", headers={"X-Request-ID": "<script>"}).headers["X-Request-ID"]
    assert replaced != "<script>"


# ---------------------------------------------------------------- сторінка
def test_home_without_query(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200 and "<form" in res.text and "Спробуйте" in res.text


def test_home_with_query_shows_results_and_highlight(client: TestClient) -> None:
    res = client.get("/", params={"q": "rabbit"})
    assert res.status_code == 200 and "<mark>" in res.text and "/view/" in res.text


def test_home_no_results_message(client: TestClient) -> None:
    res = client.get("/", params={"q": "zzzzqqq"})
    assert res.status_code == 200 and "Нічого не знайдено" in res.text


def test_home_bad_query_shows_error_with_422(client: TestClient) -> None:
    res = client.get("/", params={"q": "alice AND"})
    assert res.status_code == 422 and 'role="alert"' in res.text


def test_home_pagination_links(client: TestClient, state: AppState) -> None:
    state_size = 1
    app = create_app(Settings(default_page_size=state_size, corpus_path=None))
    app.dependency_overrides[get_state_or_none] = lambda: state
    with TestClient(app) as c:
        res = c.get("/", params={"q": "alice", "page": 2})
    assert 'rel="prev"' in res.text and 'rel="next"' in res.text


def test_home_503_while_loading(empty_client: TestClient) -> None:
    assert empty_client.get("/").status_code == 503


def test_view_doc_page(client: TestClient) -> None:
    hit = client.get("/search", params={"q": "rabbit"}).json()["results"][0]
    res = client.get(f"/view/{hit['doc_id']}", params={"q": "rabbit"})
    assert res.status_code == 200 and hit["title"] in res.text and "<mark>" in res.text


def test_view_unknown_doc_is_404(client: TestClient) -> None:
    assert client.get("/view/999999").status_code == 404


def test_highlight_escapes_html() -> None:
    # текст із тегами не має потрапити на сторінку як розмітка
    out = str(highlight("<b>небезпечно</b> **слово**"))
    assert "<b>" not in out and "&lt;b&gt;" in out and "<mark>слово</mark>" in out


def test_doc_text_found_for_windows_style_path(corpus_dir: Path) -> None:
    """Індекс побудовано на Windows (шлях зі зворотними рисками), а сервіс працює на Linux."""
    from findex.model import DocMeta

    texts = DocTexts(corpus_dir)
    name = sorted(p.name for p in corpus_dir.rglob("*.txt"))[0]
    meta = DocMeta(path=f"demo\\corpus\\{name}", title="x")
    assert texts.text(meta)


# ---------------------------------------------------------------- lifespan і налаштування
def test_lifespan_loads_index_from_settings(saved_index: Path, corpus_dir: Path) -> None:
    """Справжній старт: індекс читається з Settings.index_path один раз, у фоні."""
    app = create_app(Settings(index_path=saved_index, corpus_path=corpus_dir))
    with TestClient(app) as c:
        for _ in range(100):  # чекаю, поки фонове завантаження закінчиться (до ~5 с)
            if c.get("/health").status_code == 200:
                break
            time.sleep(0.05)
        assert c.get("/health").status_code == 200
        assert c.get("/search", params={"q": "rabbit"}).json()["total"] > 0


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDEX_PATH", "/data/my.bin")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    s = Settings()
    assert s.index_path == Path("/data/my.bin") and s.log_level == "WARNING"


def test_missing_index_does_not_crash_service(empty_client: TestClient) -> None:
    # файла індексу немає, але сервіс живий і чесно каже 503
    assert empty_client.get("/openapi.json").status_code in (200, 404)
    assert empty_client.get("/health").status_code == 503


def test_search_routes_are_plain_def_not_async() -> None:
    """Пошук це робота процесора: async def зупинив би цикл подій (див. README, замір)."""
    for route in create_app(Settings(corpus_path=None)).routes:
        path = getattr(route, "path", "")
        endpoint = getattr(route, "endpoint", None)
        if path in {"/search", "/docs/{doc_id}", "/stats", "/"}:
            assert endpoint is not None and not inspect.iscoroutinefunction(endpoint), path


def test_get_state_dependency_raises_503() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as err:
        get_state(None)
    assert err.value.status_code == 503


# ---------------------------------------------------------------- CLI: findex serve
def test_serve_passes_options_to_uvicorn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import uvicorn

    calls: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.update(args=a, kwargs=kw))
    res = CliRunner().invoke(
        cli_app, ["serve", "--port", "9001", "--workers", "3", "--index", str(tmp_path / "i.bin")]
    )
    assert res.exit_code == 0, res.output
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["port"] == 9001 and kwargs["workers"] == 3 and kwargs["factory"] is True
    assert calls["args"] == ("findex.web.app:create_app",)


def test_serve_port_comes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    calls: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.update(kwargs=kw))
    monkeypatch.setenv("PORT", "10000")
    res = CliRunner().invoke(cli_app, ["serve", "--host", "0.0.0.0"])
    assert res.exit_code == 0, res.output
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict) and kwargs["port"] == 10000 and kwargs["host"] == "0.0.0.0"


def test_module_exports() -> None:
    # щоб uvicorn міг імпортувати "findex.web.app:create_app"
    assert callable(web_app_module.create_app)
