"""Перевірки краулера. Усе через httpx.MockTransport: жива мережа не потрібна."""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from findex.crawler import (
    CrawlStats,
    HostGate,
    RobotsCache,
    backoff_delay,
    canonicalize,
    crawl,
    fetch,
    parse_html,
    parse_retry_after,
)

UA = "findex-crawler/test (contact: test@example.org)"
Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


def client_for(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


class Sleeps:
    """Підмінена asyncio.sleep: нічого не чекає, лише запам'ятовує, скільки просили."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


# ------------------------------------------------------------ canonicalize
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("HTTP://Example.COM/A", "http://example.com/A"),  # схема і хост у нижньому регістрі
        ("http://example.com:80/x", "http://example.com/x"),  # порт за замовчуванням
        ("https://example.com:443", "https://example.com/"),
        ("http://example.com:8080/x", "http://example.com:8080/x"),  # інший порт лишається
        ("http://example.com/a#frag", "http://example.com/a"),  # фрагмент відкидаємо
        ("http://example.com/a/", "http://example.com/a"),  # кінцевий слеш
        ("http://example.com//a///b", "http://example.com/a/b"),  # подвійні слеші
        ("http://example.com/p?b=2&a=1", "http://example.com/p?a=1&b=2"),  # порядок query
        ("http://user:pw@example.com/", "http://example.com/"),  # логін відкидаємо
        ("mailto:a@b.c", None),
        ("javascript:void(0)", None),
        ("ftp://example.com/", None),
        ("/relative", None),  # без бази це не адреса
    ],
)
def test_canonicalize(url: str, expected: str | None) -> None:
    assert canonicalize(url) == expected


def test_canonicalize_relative_with_base() -> None:
    assert canonicalize("../b?y=1&x=2#z", base="http://e.com/a/c/") == "http://e.com/a/b?x=2&y=1"


def test_canonical_forms_are_equal() -> None:
    a = canonicalize("HTTP://E.com:80/a/?b=2&a=1#x")
    b = canonicalize("http://e.com/a?a=1&b=2")
    assert a == b


# ------------------------------------------------------------ parse_html
def test_parse_html_title_text_links() -> None:
    html = (
        "<html><head><title> Мій  заголовок </title><style>p{}</style></head>"
        "<body><script>var x=1;</script><p>Привіт <b>світ</b></p>"
        '<a href="/a">A</a><a href="/b" rel="nofollow">B</a></body></html>'
    )
    page = parse_html(html)
    assert page.title == "Мій заголовок"
    assert page.text.split() == ["Привіт", "світ", "A", "B"]  # script і style не потрапили
    assert page.links == ["/a"]  # nofollow пропущено


def test_parse_html_survives_broken_markup() -> None:
    assert parse_html("<p>не закрито <a href='/x'>x").links == ["/x"]


# ------------------------------------------------------------ Retry-After і backoff
def test_retry_after_seconds_and_date_and_garbage() -> None:
    assert parse_retry_after("7") == 7.0
    now = datetime(2026, 1, 1, tzinfo=UTC)
    later = (now + timedelta(seconds=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(later, now) == pytest.approx(30.0)
    assert parse_retry_after("не дата") is None
    assert parse_retry_after(None) is None


def test_backoff_grows_has_jitter_and_cap() -> None:
    rng = random.Random(1)
    d1 = [backoff_delay(1, 1.0, 100, rng) for _ in range(50)]
    d3 = [backoff_delay(3, 1.0, 100, rng) for _ in range(50)]
    assert all(0.5 <= d <= 1.0 for d in d1)  # половина гарантована + jitter
    assert all(2.0 <= d <= 4.0 for d in d3)  # 2^(3-1) = 4
    assert len(set(d3)) > 1  # справді випадкові
    assert backoff_delay(20, 1.0, 5.0, rng) <= 5.0  # стеля


# ------------------------------------------------------------ fetch
def test_fetch_ok_records_status_bytes_time() -> None:
    async def run() -> None:
        async with client_for(lambda r: httpx.Response(200, html="<p>hi</p>")) as c:
            res = await fetch(c, "http://e.com/")
        assert res.ok and res.status == 200 and res.nbytes == 9 and res.attempts == 1
        assert res.seconds >= 0 and res.is_html

    asyncio.run(run())


def test_fetch_404_is_not_retried() -> None:
    calls: list[str] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(str(r.url))
        return httpx.Response(404)

    async def run() -> None:
        async with client_for(handler) as c:
            res = await fetch(c, "http://e.com/x", sleep=Sleeps())
        assert not res.ok and res.status == 404 and res.attempts == 1

    asyncio.run(run())
    assert len(calls) == 1


def test_fetch_retries_503_then_succeeds() -> None:
    codes = iter([503, 503, 200])
    sleeps = Sleeps()

    async def run() -> None:
        async with client_for(lambda r: httpx.Response(next(codes), html="ok")) as c:
            res = await fetch(c, "http://e.com/", sleep=sleeps, rng=random.Random(1))
        assert res.ok and res.attempts == 3

    asyncio.run(run())
    assert len(sleeps.delays) == 2 and sleeps.delays[1] > sleeps.delays[0] * 0.9


def test_fetch_gives_up_after_three_retries() -> None:
    calls: list[int] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500)

    async def run() -> None:
        async with client_for(handler) as c:
            res = await fetch(c, "http://e.com/", sleep=Sleeps())
        assert not res.ok and res.attempts == 4 and "gave up" in (res.error or "")

    asyncio.run(run())
    assert len(calls) == 4  # 1 запит + щонайбільше 3 повтори


def test_fetch_honours_retry_after() -> None:
    answers = iter(
        [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, html="x")]
    )
    sleeps = Sleeps()

    async def run() -> None:
        async with client_for(lambda r: next(answers)) as c:
            res = await fetch(c, "http://e.com/", sleep=sleeps, backoff_base=0.1)
        assert res.ok

    asyncio.run(run())
    assert sleeps.delays == [7.0]  # сервер просив 7 с, а власний backoff був би менший


def test_fetch_retries_timeout() -> None:
    n = 0

    async def handler(r: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        if n == 1:
            await asyncio.sleep(5)  # перша відповідь надто повільна
        return httpx.Response(200, html="ok")

    async def run() -> None:
        async with client_for(handler) as c:
            res = await fetch(c, "http://e.com/", timeout=0.05, sleep=Sleeps())
        assert res.ok and res.attempts == 2

    asyncio.run(run())


def test_fetch_redirect_loop_is_error_without_retry() -> None:
    calls: list[str] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(r.url.path)
        other = "/b" if r.url.path == "/a" else "/a"
        return httpx.Response(302, headers={"Location": other})

    async def run() -> None:
        async with client_for(handler) as c:
            res = await fetch(c, "http://e.com/a", sleep=Sleeps())
        assert not res.ok and "redirect loop" in (res.error or "") and res.attempts == 1

    asyncio.run(run())
    assert calls == ["/a", "/b"]


def test_fetch_follows_redirect_and_blocks_foreign_host() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/old":
            return httpx.Response(301, headers={"Location": "/new"})
        if r.url.path == "/out":
            return httpx.Response(301, headers={"Location": "http://evil.com/"})
        return httpx.Response(200, html="new")

    async def run() -> None:
        only_e = lambda u: "e.com" in u  # noqa: E731
        async with client_for(handler) as c:
            ok = await fetch(c, "http://e.com/old", accept_redirect=only_e)
            bad = await fetch(c, "http://e.com/out", accept_redirect=only_e)
        assert ok.ok and ok.final_url == "http://e.com/new"
        assert not bad.ok and "disallowed" in (bad.error or "")

    asyncio.run(run())


def test_fetch_connection_error_not_retried() -> None:
    calls: list[int] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("boom")

    async def run() -> None:
        async with client_for(handler) as c:
            res = await fetch(c, "http://e.com/", sleep=Sleeps())
        assert not res.ok and res.status is None and "ConnectError" in (res.error or "")

    asyncio.run(run())
    assert len(calls) == 1


# ------------------------------------------------------------ robots.txt
ROBOTS = "User-agent: *\nDisallow: /private/\nCrawl-delay: 2\n"


def test_robots_per_host_cached_and_obeyed() -> None:
    hits: Counter[str] = Counter()

    def handler(r: httpx.Request) -> httpx.Response:
        hits[f"{r.url.host}{r.url.path}"] += 1
        if r.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS if r.url.host == "a.com" else "")
        return httpx.Response(200, html="x")

    async def run() -> None:
        async with client_for(handler) as c:
            rc = RobotsCache(c, "findex-crawler")
            assert not await rc.allowed("http://a.com/private/x")
            assert await rc.allowed("http://a.com/public")
            assert await rc.allowed("http://a.com/other")
            assert await rc.allowed("http://b.com/private/x")  # у b.com свій (порожній) robots
            assert await rc.crawl_delay("http://a.com/") == 2.0
        assert rc.requests == 2

    asyncio.run(run())
    assert hits["a.com/robots.txt"] == 1 and hits["b.com/robots.txt"] == 1  # кеш працює


def test_robots_parallel_requests_load_file_once() -> None:
    hits: list[int] = []

    async def handler(r: httpx.Request) -> httpx.Response:
        hits.append(1)
        await asyncio.sleep(0.02)
        return httpx.Response(200, text=ROBOTS)

    async def run() -> None:
        async with client_for(handler) as c:
            rc = RobotsCache(c, "findex-crawler")
            await asyncio.gather(*(rc.allowed(f"http://a.com/p{i}") for i in range(10)))

    asyncio.run(run())
    assert len(hits) == 1


def test_robots_404_allows_all_and_5xx_disallows_all() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        return httpx.Response(404 if r.url.host == "ok.com" else 503)

    async def run() -> None:
        async with client_for(handler) as c:
            rc = RobotsCache(c, "findex-crawler", sleep=Sleeps())
            assert await rc.allowed("http://ok.com/anything")
            assert not await rc.allowed("http://down.com/anything")

    asyncio.run(run())


# ------------------------------------------------------------ HostGate
def test_host_gate_enforces_min_delay() -> None:
    async def run() -> float:
        gate = HostGate(per_host=5, min_delay=0.05)

        async def one() -> None:
            async with gate.slot("h"):
                pass

        t0 = asyncio.get_running_loop().time()
        await asyncio.gather(*(one() for _ in range(4)))
        return asyncio.get_running_loop().time() - t0

    assert asyncio.run(run()) >= 0.14  # 4 запити з паузою 0.05: перший без очікування


def test_host_gate_limits_parallelism_per_host() -> None:
    async def run() -> int:
        gate = HostGate(per_host=2, min_delay=0)
        now = peak = 0

        async def one() -> None:
            nonlocal now, peak
            async with gate.slot("h"):
                now += 1
                peak = max(peak, now)
                await asyncio.sleep(0.01)
                now -= 1

        await asyncio.gather(*(one() for _ in range(10)))
        return peak

    assert asyncio.run(run()) == 2


# ------------------------------------------------------------ crawl
def site(pages: dict[str, str], robots: str = "") -> Callable[[httpx.Request], httpx.Response]:
    """Маленький сайт у пам'яті: шлях -> html."""
    log = Counter[str]()

    def handler(r: httpx.Request) -> httpx.Response:
        query = r.url.query.decode()
        key = f"{r.url.host}{r.url.path}" + (f"?{query}" if query else "")
        log[key] += 1
        if r.url.path == "/robots.txt":
            return httpx.Response(200, text=robots, headers={"content-type": "text/plain"})
        if key in pages:
            return httpx.Response(200, html=pages[key])
        return httpx.Response(404)

    handler.log = log  # type: ignore[attr-defined]  # додаю лічильник як атрибут функції, щоб тест міг його прочитати
    return handler


def collect(
    handler: Callable[[httpx.Request], httpx.Response], urls: list[str], **kw: object
) -> tuple[list[str], CrawlStats]:
    stats = CrawlStats()

    async def run() -> list[str]:
        out: list[str] = []
        transport = httpx.MockTransport(handler)
        async with contextlib.aclosing(
            crawl(urls, user_agent=UA, transport=transport, stats=stats, sleep=Sleeps(), **kw)  # type: ignore[arg-type]  # kw збирається як словник object
        ) as pages:
            async for p in pages:
                out.append(p.url)
        return out

    return asyncio.run(run()), stats


def link(*paths: str) -> str:
    return "<title>t</title>" + "".join(f'<a href="{p}">x</a>' for p in paths)


def test_crawl_visits_each_page_once_and_stays_on_host() -> None:
    h = site(
        {
            "e.com/": link("/a", "/b", "http://other.com/z"),
            "e.com/a": link("/", "/b/", "/a#top"),  # циклічні й повторні посилання
            "e.com/b": link("/a?y=1&x=2", "/a?x=2&y=1"),
            "e.com/a?x=2&y=1": link(),
        }
    )
    urls, stats = collect(h, ["http://E.com"], max_pages=50, delay=0)
    assert sorted(urls) == [
        "http://e.com/",
        "http://e.com/a",
        "http://e.com/a?x=2&y=1",
        "http://e.com/b",
    ]
    log: Counter[str] = h.log  # type: ignore[attr-defined]  # атрибут log додано в site() вручну
    assert all(v == 1 for k, v in log.items()), log  # жодної адреси двічі
    assert "other.com/z" not in log  # чужий хост не чіпаємо
    assert stats.pages == 4


def test_crawl_respects_max_pages() -> None:
    pages = {f"e.com/p{i}": link(*(f"/p{j}" for j in range(i + 1, i + 4))) for i in range(40)}
    pages["e.com/"] = link("/p0")
    urls, stats = collect(site(pages), ["http://e.com/"], max_pages=7, concurrency=4, delay=0)
    assert len(urls) == 7 and stats.pages == 7


def test_crawl_respects_max_depth() -> None:
    pages = {"e.com/": link("/a"), "e.com/a": link("/b"), "e.com/b": link("/c"), "e.com/c": link()}
    urls, _ = collect(site(pages), ["http://e.com/"], max_depth=1, delay=0)
    assert sorted(urls) == ["http://e.com/", "http://e.com/a"]


def test_crawl_obeys_robots_and_never_requests_forbidden_url() -> None:
    h = site(
        {"e.com/": link("/ok", "/private/x"), "e.com/ok": link(), "e.com/private/x": link()},
        robots="User-agent: *\nDisallow: /private/\n",
    )
    urls, stats = collect(h, ["http://e.com/"], delay=0)
    assert sorted(urls) == ["http://e.com/", "http://e.com/ok"]
    assert "e.com/private/x" not in h.log  # type: ignore[attr-defined]  # атрибут log додано в site() вручну
    assert stats.skipped == 1


def test_crawl_counts_errors_and_continues() -> None:
    h = site({"e.com/": link("/missing", "/ok"), "e.com/ok": link()})
    urls, stats = collect(h, ["http://e.com/"], delay=0)
    assert sorted(urls) == ["http://e.com/", "http://e.com/ok"] and stats.errors == 1


def test_crawl_request_concurrency_is_limited() -> None:
    state = {"now": 0, "peak": 0}
    pages = {f"e.com/p{i}": link() for i in range(30)}
    pages["e.com/"] = link(*(f"/p{i}" for i in range(30)))

    async def handler(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/robots.txt":
            return httpx.Response(404)
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        return httpx.Response(200, html=pages[f"e.com{r.url.path}"])

    stats = CrawlStats()

    async def run() -> int:
        n = 0
        t = httpx.MockTransport(handler)
        gen = crawl(
            ["http://e.com/"], user_agent=UA, transport=t, stats=stats,
            max_pages=100, concurrency=8, per_host=3, delay=0,
        )  # fmt: skip
        async with contextlib.aclosing(gen) as pages_iter:
            async for _ in pages_iter:
                n += 1
        return n

    assert asyncio.run(run()) == 31
    assert state["peak"] == 3  # ліміт на хост (3) строгіший за загальний (8)


def test_crawl_closes_client_and_leaves_no_tasks_on_early_break() -> None:
    closed: list[bool] = []

    class Tracking(httpx.MockTransport):
        async def aclose(self) -> None:
            closed.append(True)

    pages = {f"e.com/p{i}": link(f"/p{i + 1}", f"/p{i + 2}") for i in range(200)}
    pages["e.com/"] = link("/p0")

    async def run() -> int:
        t = Tracking(site(pages))
        got = 0
        async with contextlib.aclosing(
            crawl(
                ["http://e.com/"], user_agent=UA, transport=t, max_pages=100, concurrency=5, delay=0
            )
        ) as gen:
            async for _ in gen:
                got += 1
                if got == 3:
                    break  # раніше, ніж набралось max_pages
        await asyncio.sleep(0)
        return len(asyncio.all_tasks()) - 1  # без самої цієї задачі

    assert asyncio.run(run()) == 0  # жодна задача воркера не лишилась
    assert closed == [True]  # клієнт закрито


def test_crawl_rejects_bad_start_url() -> None:
    async def run() -> None:
        async with contextlib.aclosing(crawl(["mailto:a@b.c"], user_agent=UA)) as g:
            async for _ in g:
                pass

    with pytest.raises(ValueError, match="start URLs"):
        asyncio.run(run())


def test_crawl_sends_honest_user_agent() -> None:
    seen: set[str] = set()

    def handler(r: httpx.Request) -> httpx.Response:
        seen.add(r.headers["user-agent"])
        return (
            httpx.Response(200, html=link()) if r.url.path != "/robots.txt" else httpx.Response(404)
        )

    collect(handler, ["http://e.com/"], delay=0)
    assert seen == {UA}
