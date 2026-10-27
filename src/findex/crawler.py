"""
асинхронний краулер (лаба 6).

Краулер майже весь час чекає на мережу, тому один потік з циклом подій (asyncio)
тримає десятки запитів одночасно. Головне правило: у корутині не можна робити
довгу блокувальну роботу (час.sleep, важкий розбір, індексація), бо тоді стоїть
весь цикл і всі запити разом з ним.

Складові:
  canonicalize   — приводить URL до одного вигляду (щоб не качати двічі одне й те саме);
  parse_html     — дістає заголовок, текст і посилання;
  fetch          — один URL: таймаут, повтори з backoff і jitter, Retry-After;
  RobotsCache    — robots.txt окремо для кожного хоста, з кешем;
  HostGate       — ліміт одночасних запитів і мінімальна пауза на хост;
  crawl          — асинхронний генератор сторінок (черга + воркери у TaskGroup).
"""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import logging
import random
import re
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

log = logging.getLogger("findex.crawl")

# коди, після яких має сенс повторити запит: "забагато запитів" і помилки сервера
RETRY_STATUS_MIN_5XX = 500
REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
DEFAULT_PORTS = {"http": 80, "https": 443}


# ---------------------------------------------------------------- адреси
def canonicalize(url: str, base: str | None = None) -> str | None:
    """Приводить URL до одного вигляду. Повертає None, якщо це не http(s)-адреса.

    Що робимо: схема й хост у нижньому регістрі, прибираємо порт за замовчуванням,
    відкидаємо #фрагмент і логін, зливаємо подвійні слеші, прибираємо кінцевий
    слеш (крім кореня) і сортуємо параметри запиту."""
    try:
        parts = urlsplit(urljoin(base, url.strip()) if base else url.strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in DEFAULT_PORTS or not host:
        return None
    if ":" in host:
        host = f"[{host}]"  # IPv6 пишеться в дужках
    netloc = host if port in (None, DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path) or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def host_of(url: str) -> str:
    """Хост з портом (так ми розрізняємо сайти на localhost з різними портами)."""
    return urlsplit(url).netloc.lower()


# ---------------------------------------------------------------- розбір HTML
@dataclass(frozen=True)
class ParsedPage:
    title: str
    text: str
    links: list[str]


class _PageParser(HTMLParser):
    """Збирає текст, заголовок і посилання зі сторінки (без зовнішніх бібліотек)."""

    _SKIP = frozenset({"script", "style", "noscript", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._skip = 0  # всередині script/style текст не збираємо
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "a":
            d = dict(attrs)
            href = d.get("href")
            rel = (d.get("rel") or "").lower()
            if href and "nofollow" not in rel:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        (self.title_parts if self._in_title else self.text_parts).append(data)


def parse_html(html: str) -> ParsedPage:
    """Розбирає HTML. Це CPU-робота, тому з корутин її віддаю в потік (asyncio.to_thread)."""
    parser = _PageParser()
    parser.feed(html)
    parser.close()
    title = " ".join(" ".join(parser.title_parts).split())
    text = " ".join(" ".join(parser.text_parts).split())
    return ParsedPage(title, text, parser.links)


# ---------------------------------------------------------------- завантаження
@dataclass
class FetchResult:
    """Що вийшло із завантаження однієї адреси."""

    url: str
    status: int | None = None  # None, якщо відповіді взагалі не було
    nbytes: int = 0
    seconds: float = 0.0
    attempts: int = 0
    body: str = ""
    is_html: bool = False
    final_url: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is not None and 200 <= self.status < 300


class SlotFactory(Protocol):
    """Функція, що дає "місце" для одного запиту до url (семафори й пауза між запитами)."""

    def __call__(self, url: str) -> AbstractAsyncContextManager[None]: ...


def _no_slot(url: str) -> AbstractAsyncContextManager[None]:
    return contextlib.nullcontext()


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Retry-After: або число секунд, або дата HTTP. Повертає секунди чи None."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def backoff_delay(attempt: int, base: float, cap: float, rng: random.Random) -> float:
    """Пауза перед повтором: base * 2^(спроба-1), але не більше cap, плюс jitter.

    Jitter (випадкова добавка) потрібен, щоб багато воркерів, які впали разом,
    не повторювали запит у той самий момент. Беремо половину гарантовану і
    половину випадкову."""
    ceiling = min(cap, base * 2 ** (attempt - 1))
    return ceiling / 2 + rng.uniform(0, ceiling / 2)


class _RedirectError(Exception):
    """Редірект зациклився, занадто довгий або веде на заборонений хост."""


@dataclass
class _Raw:
    status: int
    headers: httpx.Headers
    body: str
    nbytes: int
    is_html: bool
    final_url: str


async def _get_once(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    max_redirects: int,
    slot: SlotFactory,
    accept_redirect: Callable[[str], bool],
) -> _Raw:
    """Одна спроба: іду за редіректами вручну (кожен крок — окремий запит зі слотом)."""
    current = url
    visited = {url}
    for _ in range(max_redirects + 1):
        async with slot(current), asyncio.timeout(timeout), client.stream("GET", current) as resp:
            if resp.status_code in REDIRECT_CODES and "location" in resp.headers:
                nxt = canonicalize(resp.headers["location"], base=current)
                if nxt is None:
                    raise _RedirectError("redirect to a non-http address")
                if nxt in visited:
                    raise _RedirectError("redirect loop")
                if not accept_redirect(nxt):
                    raise _RedirectError("redirect to a disallowed host")
                visited.add(nxt)
                current = nxt
                continue
            ctype = resp.headers.get("content-type", "").lower()
            is_html = "html" in ctype or ctype == ""
            chunks: list[bytes] = []
            n = 0
            # тіло читаю лише для 200 і лише для текстових типів (html, robots.txt)
            if resp.status_code == 200 and (is_html or ctype.startswith("text/")):
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                    n += len(chunk)
                    if n > max_bytes:
                        break  # більші за ліміт сторінки не читаємо до кінця
            raw = b"".join(chunks)[:max_bytes]
            body = raw.decode(resp.encoding or "utf-8", errors="replace")
            return _Raw(resp.status_code, resp.headers, body, len(raw), is_html, current)
    raise _RedirectError("too many redirects")


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    backoff_cap: float = 30.0,
    max_bytes: int = 2_000_000,
    max_redirects: int = 5,
    slot: SlotFactory = _no_slot,
    accept_redirect: Callable[[str], bool] = lambda _u: True,
    rng: random.Random | None = None,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    on_retry: Callable[[], None] | None = None,
) -> FetchResult:
    """Завантажує url і записує статус, байти й час.

    Повторюю ТІЛЬКИ 429, 5xx і таймаут, щонайбільше max_retries разів, з exponential
    backoff і jitter; заголовок Retry-After теж враховую. Інші 4xx, помилки
    з'єднання і зациклені редіректи не повторюю."""
    rng = rng or random.Random()
    start = time.perf_counter()
    attempt = 0
    while True:
        attempt += 1
        retry_after: float | None = None
        status: int | None = None
        try:
            raw = await _get_once(
                client,
                url,
                timeout=timeout,
                max_bytes=max_bytes,
                max_redirects=max_redirects,
                slot=slot,
                accept_redirect=accept_redirect,
            )
        except TimeoutError:
            reason = f"timeout after {timeout:g}s"
        except _RedirectError as exc:
            return _failed(url, start, attempt, None, str(exc))
        except httpx.HTTPError as exc:
            return _failed(url, start, attempt, None, f"{type(exc).__name__}: {exc}")
        else:
            status = raw.status
            if status != 429 and status < RETRY_STATUS_MIN_5XX:
                return FetchResult(
                    url=url,
                    status=status,
                    nbytes=raw.nbytes,
                    seconds=time.perf_counter() - start,
                    attempts=attempt,
                    body=raw.body,
                    is_html=raw.is_html,
                    final_url=raw.final_url,
                    # 404 та інші 4xx: не повторюємо, але це не успіх
                    error=None if 200 <= status < 300 else f"HTTP {status}",
                )
            reason = f"HTTP {status}"
            retry_after = parse_retry_after(raw.headers.get("retry-after"))

        # сюди доходимо лише коли помилку можна повторити
        if attempt > max_retries:
            return _failed(
                url, start, attempt, status, f"gave up after {attempt} attempts: {reason}"
            )
        delay = backoff_delay(attempt, backoff_base, backoff_cap, rng)
        if retry_after is not None:
            delay = max(delay, min(retry_after, backoff_cap))  # сервер просить чекати довше
        log.warning("RETRY %s attempt=%d reason=%s wait=%.2fs", url, attempt, reason, delay)
        if on_retry is not None:
            on_retry()
        await sleep(delay)


def _failed(url: str, start: float, attempts: int, status: int | None, error: str) -> FetchResult:
    return FetchResult(
        url=url,
        status=status,
        seconds=time.perf_counter() - start,
        attempts=attempts,
        final_url=url,
        error=error,
    )


# ---------------------------------------------------------------- robots.txt
class RobotsCache:
    """robots.txt окремо для кожного хоста; відповідь кешується.

    Правила RFC 9309: файл є (200) — слухаємось; файлу немає (4xx) — можна все;
    сервер недоступний (5xx, помилка мережі) — вважаємо, що заборонено все."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent_token: str,
        *,
        timeout: float = 10.0,
        slot: SlotFactory = _no_slot,
        rng: random.Random | None = None,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._token = user_agent_token
        self._timeout = timeout
        self._slot = slot
        self._rng = rng
        self._sleep = sleep
        self._rules: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}  # щоб один хост не вантажили двічі разом
        self.requests = 0  # скільки разів справді ходили по robots.txt

    async def _rules_for(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin in self._rules:
                return self._rules[origin]
            self.requests += 1
            robots_url = f"{origin}/robots.txt"
            res = await fetch(
                self._client,
                robots_url,
                timeout=self._timeout,
                slot=self._slot,
                accept_redirect=lambda u: urlsplit(u).netloc == parts.netloc,
                rng=self._rng,
                sleep=self._sleep,
            )
            rp = RobotFileParser(robots_url)
            if res.ok:
                rp.parse(res.body.splitlines())
            elif res.status is not None and 400 <= res.status < 500:
                rp.parse([])  # порожні правила: дозволено все
            else:
                log.warning(
                    "ROBOTS %s unavailable (%s): treating as disallow-all", origin, res.error
                )
                rp.parse(["User-agent: *", "Disallow: /"])  # заборонено все
            self._rules[origin] = rp
            return rp

    async def allowed(self, url: str) -> bool:
        rp = await self._rules_for(url)
        return rp.can_fetch(self._token, url)

    async def crawl_delay(self, url: str) -> float:
        """Crawl-delay з robots.txt (секунди), або 0."""
        rp = await self._rules_for(url)
        value = rp.crawl_delay(self._token)
        return float(value) if value is not None else 0.0


# ---------------------------------------------------------------- ліміти на хост
class HostGate:
    """Не більше per_host одночасних запитів до одного хоста і мінімальна пауза між ними."""

    def __init__(self, per_host: int, min_delay: float) -> None:
        self._per_host = per_host
        self._delay = min_delay
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._next_time: dict[str, float] = {}
        self.extra_delay: dict[str, float] = {}  # Crawl-delay з robots.txt для хоста

    @contextlib.asynccontextmanager
    async def slot(self, host: str) -> AsyncGenerator[None]:
        sem = self._sems.setdefault(host, asyncio.Semaphore(self._per_host))
        async with sem:
            await self._pace(host)
            yield

    async def _pace(self, host: str) -> None:
        """Резервує момент, коли можна стартувати запит до хоста, і чекає до нього.

        Між читанням і записом _next_time немає await, тому два воркери не отримають
        один і той самий момент."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        delay = max(self._delay, self.extra_delay.get(host, 0.0))
        start = max(now, self._next_time.get(host, 0.0))
        self._next_time[host] = start + delay
        if start > now:
            await asyncio.sleep(start - now)


# ---------------------------------------------------------------- краулер
@dataclass
class CrawlStats:
    """Лічильники для екрана rich. Їх читає інша задача, а пишуть воркери."""

    started: float = field(default_factory=time.perf_counter)
    pages: int = 0  # зібрані сторінки
    errors: int = 0  # адреси, що не вдалось завантажити
    retries: int = 0
    skipped: int = 0  # заборонено robots.txt або не html
    in_flight: int = 0  # запити, що зараз чекають відповіді
    queue_depth: int = 0  # скільки адрес чекає в черзі
    discovered: int = 0  # скільки різних адрес бачили
    nbytes: int = 0
    finished: float | None = None

    @property
    def elapsed(self) -> float:
        return (self.finished or time.perf_counter()) - self.started

    @property
    def pages_per_second(self) -> float:
        return self.pages / self.elapsed if self.elapsed > 0 else 0.0


@dataclass(frozen=True)
class Page:
    url: str
    depth: int
    status: int
    title: str
    text: str
    nbytes: int
    seconds: float
    # коли сторінку завантажено (час UTC у форматі ISO)
    fetched_at: str = ""


async def crawl(
    start_urls: list[str],
    *,
    user_agent: str,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host: int = 2,
    delay: float = 0.0,
    max_depth: int = 3,
    timeout: float = 10.0,
    allowed_hosts: set[str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    rng: random.Random | None = None,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> AsyncGenerator[Page]:
    """Асинхронний генератор: віддає сторінки по одній, у міру надходження.

    Усередині: черга адрес, `concurrency` воркерів у TaskGroup, один семафор на всі
    запити і окремий (плюс пауза) на кожен хост. Коли набрано max_pages, воркери
    скасовуються, а клієнт закривається. Викликайте через contextlib.aclosing,
    щоб при break усе прибралось одразу."""
    stats = stats or CrawlStats()
    seeds: list[str] = []
    for raw in start_urls:
        c = canonicalize(raw)
        if c is not None and c not in seeds:
            seeds.append(c)
    if not seeds:
        raise ValueError("no valid http(s) start URLs")
    hosts = allowed_hosts if allowed_hosts is not None else {host_of(u) for u in seeds}
    token = user_agent.split("/")[0].split()[0]  # "findex-crawler" для robots.txt

    frontier: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    out: asyncio.Queue[Page | None] = asyncio.Queue(maxsize=max(2, concurrency))
    seen: set[str] = set()  # адреси, які вже стояли в черзі: другий раз не качаємо
    max_seen = max_pages * 50  # стеля, щоб seen не росла без меж

    global_sem = asyncio.Semaphore(concurrency)  # один семафор на ВСІ запити
    gate = HostGate(per_host, delay)
    # коли набрано max_pages, ставлю True: нові запити не починаю, але ті, що вже йдуть,
    # дозволяю довести до кінця (скасування посеред запиту на Windows іноді зависає)
    stopping = False

    @contextlib.asynccontextmanager
    async def slot(url: str) -> AsyncGenerator[None]:
        # спочатку місце на хості (там і пауза), потім загальне місце
        async with gate.slot(host_of(url)), global_sem:
            stats.in_flight += 1
            try:
                yield
            finally:
                stats.in_flight -= 1

    def enqueue(url: str, depth: int) -> None:
        if url in seen or len(seen) >= max_seen:
            return
        seen.add(url)
        stats.discovered += 1
        frontier.put_nowait((url, depth))

    for s in seeds:
        enqueue(s, 0)

    def count_retry() -> None:
        stats.retries += 1

    # Клієнт створюю в потоці: httpx при створенні вантажить сертифікати (SSL), це
    # блокувальна робота на ~0.1 с, і в debug-режимі asyncio це видно як зупинку циклу.
    client = await asyncio.to_thread(
        lambda: httpx.AsyncClient(
            transport=transport,
            headers={"User-Agent": user_agent},
            follow_redirects=False,  # редіректи веду сам, щоб перевіряти хост кожного кроку
            timeout=None,  # таймаут ставить fetch через asyncio.timeout
            limits=httpx.Limits(max_connections=max(concurrency, 1) * 2),
        )
    )
    async with client:
        robots = RobotsCache(client, token, timeout=timeout, slot=slot, rng=rng, sleep=sleep)

        async def process(url: str, depth: int) -> None:
            if not await robots.allowed(url):
                stats.skipped += 1
                log.info("SKIP robots %s", url)
                return
            # поважаю Crawl-delay, якщо сайт його просить
            gate.extra_delay[host_of(url)] = await robots.crawl_delay(url)
            if stopping:
                return  # уже зупиняємось: нових запитів не починаю
            res = await fetch(
                client,
                url,
                timeout=timeout,
                slot=slot,
                accept_redirect=lambda u: host_of(u) in hosts,
                rng=rng,
                sleep=sleep,
                on_retry=count_retry,
            )
            ms = res.seconds * 1000
            if not res.ok:
                stats.errors += 1
                log.info(
                    "ERR %s status=%s bytes=%d ms=%.0f attempts=%d error=%s",
                    url, res.status, res.nbytes, ms, res.attempts, res.error,
                )  # fmt: skip
                return
            stats.nbytes += res.nbytes
            if not res.is_html:
                stats.skipped += 1
                log.info("SKIP non-html %s", url)
                return
            # розбір HTML віддаю в потік, щоб не зупиняти цикл подій
            parsed = await asyncio.to_thread(parse_html, res.body)
            if stopping:
                return  # ліміт набрано поки чекали: сторінку не віддаю і посилань не додаю
            log.info(
                "OK %s status=%d bytes=%d ms=%.0f attempts=%d depth=%d",
                url, res.status or 0, res.nbytes, ms, res.attempts, depth,
            )  # fmt: skip
            if depth < max_depth:
                for href in parsed.links:
                    link = canonicalize(href, base=res.final_url or url)
                    if link is not None and host_of(link) in hosts:
                        enqueue(link, depth + 1)
            if stats.pages >= max_pages:
                return  # ліміт уже набрано: цю сторінку не віддаємо
            stats.pages += 1
            page = Page(
                url,
                depth,
                res.status or 0,
                parsed.title,
                parsed.text,
                res.nbytes,
                res.seconds,
                datetime.now(UTC).isoformat(timespec="seconds"),
            )
            await out.put(page)

        async def worker() -> None:
            while True:
                url, depth = await frontier.get()
                try:
                    stats.queue_depth = frontier.qsize()
                    await process(url, depth)
                except Exception:  # одна погана адреса не повинна валити всіх
                    stats.errors += 1
                    log.exception("CRASH %s", url)
                finally:
                    frontier.task_done()
                    stats.queue_depth = frontier.qsize()

        async def closer() -> None:
            await frontier.join()  # черга порожня і всі воркери вільні: обхід закінчено
            await out.put(None)

        tasks: list[asyncio.Task[None]] = []
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(worker()) for _ in range(concurrency)]
                tasks.append(tg.create_task(closer()))
                try:
                    given = 0
                    while given < max_pages:
                        page = await out.get()
                        if page is None:
                            break
                        given += 1
                        yield page
                except GeneratorExit:
                    # споживач вийшов із циклу раніше (break) і закрив генератор.
                    # Це не помилка: без цього рядка TaskGroup загорнув би GeneratorExit
                    # у ExceptionGroup, і aclose() зламався б.
                    pass
                finally:
                    # набрали max_pages, закінчили або нас закрили: зупиняємось м'яко
                    stopping = True

                    async def discard() -> None:
                        # викидаю сторінки, які ще дійдуть, щоб воркери не застрягли на out.put
                        while True:
                            await out.get()

                    sink = asyncio.create_task(discard())
                    try:
                        # даю до 5 с, щоб запити, які вже йдуть, завершилися самі
                        async with asyncio.timeout(5):
                            await frontier.join()
                    except TimeoutError:
                        pass  # не встигли: далі скасую примусово
                    finally:
                        # лишилися тільки воркери, що чекають нову адресу: їх скасовувати безпечно
                        sink.cancel()
                        for t in tasks:
                            t.cancel()
        finally:
            stats.finished = time.perf_counter()
            stats.queue_depth = 0
