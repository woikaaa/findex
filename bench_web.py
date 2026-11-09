"""Замір для лаби 7: 20 одночасних запитів на пошук у варіантах `def` і `async def`.

Що робить скрипт:
  1. будує синтетичний корпус (слова за законом Ципфа, як у справжньому тексті) і індекс;
  2. піднімає два маршрути з тією самою роботою: один `def`, другий `async def`;
  3. шле 20 одночасних запитів і міряє: за скільки готові всі, медіану готовності і затримку
     легкого запиту /ping, надісланого посеред них (так видно, чи цикл подій живий).

Запуск:  uv run python bench_web.py --docs 3000 --rounds 5
Навантаження на повний сервіс (findex serve) міряємо окремо через oha (див. README)."""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx
from fastapi import FastAPI

from findex.corpus import Document
from findex.index import build_index_from_docs
from findex.web.service import run_search
from findex.web.state import AppState, DocTexts

CONCURRENCY = 20  # стільки запитів одночасно (за умовою лаби)


def make_corpus(n_docs: int, words_per_doc: int, vocab: int, seed: int = 7) -> list[Document]:
    """Синтетичні документи: слова w0..wN, частих мало, рідкісних багато (Ципф)."""
    rnd = random.Random(seed)
    words = [f"w{i}" for i in range(vocab)]
    weights = [1 / (i + 1) for i in range(vocab)]
    docs: list[Document] = []
    for i in range(n_docs):
        text = " ".join(rnd.choices(words, weights, k=words_per_doc))
        docs.append(Document(doc_id=f"d{i}", path=Path(f"d{i}.txt"), text=text))
    return docs


def make_app(state: AppState) -> FastAPI:
    """Мінімальний застосунок: ті самі обчислення, лише різна форма маршруту."""
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/def")
    def search_def(q: str) -> dict[str, int]:
        return {"total": run_search(state, q, "bm25", 1, 10).total}

    @app.get("/async")
    async def search_async(q: str) -> dict[str, int]:
        # блокувальна робота прямо в async def: тримає цикл подій зайнятим
        return {"total": run_search(state, q, "bm25", 1, 10).total}

    return app


async def one_round(
    client: httpx.AsyncClient, path: str, queries: list[str]
) -> tuple[float, list[float], float]:
    """20 запитів одночасно + один легкий запит /ping, надісланий посеред них.

    Повертає: час, коли готовий останній пошук; час готовності кожного пошуку (від спільного
    старту); затримку /ping. /ping нічого не рахує, тож його затримка показує, чи живий цикл."""
    t0 = time.perf_counter()

    async def call(q: str) -> float:
        r = await client.get(path, params={"q": q})
        r.raise_for_status()
        return time.perf_counter() - t0

    async def probe() -> float:
        await asyncio.sleep(0.01)  # 10 мс після старту пошуків
        r = await client.get("/ping")
        r.raise_for_status()
        # рахую від моменту, коли запит МАВ піти (t0 + 10 мс): якщо цикл зайнятий, він спізниться
        return time.perf_counter() - (t0 + 0.01)

    *done, ping = await asyncio.gather(*(call(q) for q in queries), probe())
    return max(done), list(done), ping


PORT = 8123


def serve(docs: int, words: int, vocab: int) -> None:
    """Режим сервера: будує індекс і віддає два маршрути через uvicorn."""
    import uvicorn

    index = build_index_from_docs(make_corpus(docs, words, vocab))
    state = AppState(index, DocTexts(Path("немає")), snippet_width=90)
    uvicorn.run(make_app(state), port=PORT, log_level="warning")


async def main(docs: int, words: int, vocab: int, rounds: int) -> None:
    print(f"будую корпус ({docs} документів по {words} слів) і запускаю сервер ...")
    cmd = [sys.executable, __file__, "--serve", "--docs", str(docs), "--words", str(words)]
    server = subprocess.Popen([*cmd, "--vocab", str(vocab)])
    rnd = random.Random(1)

    # різні запити, щоб кеш Searcher не підміняв реальну роботу; частота слів різна
    def queries() -> list[str]:
        return [
            " OR ".join(f"w{rnd.randrange(0, 40)}" for _ in range(3)) for _ in range(CONCURRENCY)
        ]

    try:
        await _wait_ready()
        await _run(rounds=rounds, queries=queries)
    finally:
        server.terminate()
        server.wait()


async def _wait_ready() -> None:
    """Чекає, поки сервер збудує індекс і почне відповідати."""
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}") as c:
        for _ in range(300):
            try:
                await c.get("/ping")
                return
            except httpx.TransportError:
                await asyncio.sleep(0.2)
    raise RuntimeError("сервер не запустився")


async def _run(rounds: int, queries: Callable[[], list[str]]) -> None:
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=60) as client:
        await client.get("/def", params={"q": "w1"})  # прогрів
        rows: dict[str, tuple[float, float, float, float]] = {}
        for path in ("/def", "/async"):
            totals: list[float] = []
            medians: list[float] = []
            pings: list[float] = []
            for _ in range(rounds):
                t, per, pg = await one_round(client, path, queries())
                totals.append(t)
                medians.append(statistics.median(per))
                pings.append(pg)
            rows[path] = (
                statistics.median(totals),
                statistics.median(medians),
                statistics.median(pings),
                max(pings),
            )
    print(f"\n{CONCURRENCY} одночасних запитів, медіана за {rounds} раундів (мс)")
    head = f"{'варіант':<11}{'усі 20':>9}{'p50 пошуку':>12}{'/ping, мед.':>13}{'/ping, макс.':>14}"
    print(head)
    for path, (tot, med, ping_med, ping_max) in rows.items():
        name = "def" if path == "/def" else "async def"
        print(
            f"{name:<11}{tot * 1000:>9.1f}{med * 1000:>12.1f}"
            f"{ping_med * 1000:>13.1f}{ping_max * 1000:>14.1f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=3000)
    ap.add_argument("--words", type=int, default=300)
    ap.add_argument("--vocab", type=int, default=5000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--serve", action="store_true", help="службовий режим: лише сервер")
    a = ap.parse_args()
    if a.serve:
        serve(a.docs, a.words, a.vocab)
    else:
        asyncio.run(main(a.docs, a.words, a.vocab, a.rounds))
