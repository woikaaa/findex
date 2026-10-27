"""
командний рядок findex: три команди — index, search, stats.

Усі команди друкують результат у stdout, а повідомлення про хід роботи (логи,
прогрес) — у stderr. Так вивід можна передати в інший інструмент через канал.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import IO, Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from findex.corpus import CorpusError, list_corpus_files
from findex.crawler import CrawlStats, canonicalize, crawl, host_of
from findex.index import build_index
from findex.model import Storage
from findex.parallel import Executors
from findex.query import QueryError
from findex.ranking import Searcher
from findex.scoring import ScorerName, make_scorer
from findex.snippets import snippet_for
from findex.stats import collect_stats
from findex.store import Fmt, IndexLoadError, open_index, save
from findex.sysinfo import peak_rss_mb

# окрема консоль для stderr: прогрес і помилки не псують stdout
err = Console(stderr=True)
out = Console()

app = typer.Typer(
    help="findex — пошук по текстових корпусах.",
    no_args_is_help=True,
    add_completion=False,
)


def _setup_logging(verbose: int) -> None:
    """Налаштовую логування один раз, у stderr: -v це INFO, -vv це DEBUG."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _fail(message: str) -> typer.Exit:
    """Друкує помилку одним рядком (без стека) і готує код виходу 2."""
    err.print(f"[red]помилка:[/red] {message}")
    return typer.Exit(code=2)


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="-v: INFO, -vv: DEBUG (у stderr)."),
    ] = 0,
) -> None:
    _setup_logging(verbose)


@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="Папка з .txt або файл .jsonl.")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="Куди зберегти індекс.")] = Path(
        "index.bin"
    ),
    storage: Annotated[Storage, typer.Option(help="Як зберігати постінги.")] = "slots",
    fmt: Annotated[
        Fmt | None,
        typer.Option(
            "--format", help="Формат файлу (за замовчуванням: pickle з --positions, інакше binary)."
        ),
    ] = None,
    positions: Annotated[
        bool, typer.Option("--positions", help="Зберігати позиції (для фраз).")
    ] = False,
    limit: Annotated[int | None, typer.Option(help="Взяти лише перші N документів.")] = None,
    workers: Annotated[
        int | None,
        typer.Option(min=1, help="Скільки воркерів (за замовчуванням 4 для потоків/процесів)."),
    ] = None,
    executor: Annotated[
        Executors | None,
        typer.Option(help="Чим виконувати: serial, threads або processes (див. типовий вибір)."),
    ] = None,
) -> None:
    """Будує індекс і зберігає його у файл.

    Типовий вибір за моїми вимірами (лаба 5): з --storage array процеси швидші за
    один потік, тому беремо processes з 4 воркерами. Для slots і для позицій
    (--positions) процеси програють через pickle, тому беремо serial."""
    if not corpus.exists():
        raise _fail(f"корпус не знайдено: {corpus}")
    # типовий виконавець залежить від способу зберігання (див. документацію команди)
    if executor is None:
        executor = "processes" if storage == "array" and corpus.is_dir() else "serial"
    if workers is None:
        workers = min(os.cpu_count() or 1, 4) if executor != "serial" else 1
    # бінарний формат не зберігає позиції, тож для фраз беремо pickle
    if fmt is None:
        fmt = "pickle" if positions else "binary"
    if fmt == "binary" and positions:
        raise _fail("формат binary не зберігає позиції: оберіть --format pickle")
    if storage == "array" and positions:
        raise _fail("--positions не працює з --storage array")

    # скільки документів чекати — для смужки прогресу (для .jsonl невідомо)
    total: int | None = None
    if corpus.is_dir():
        total = len(list_corpus_files(corpus))
        if limit is not None:
            total = min(total, limit)

    try:
        with Progress(
            TextColumn("[bold]індексація"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=err,
        ) as progress:
            task = progress.add_task("index", total=total)
            idx = build_index(
                corpus,
                storage=storage,
                positions=positions,
                limit=limit,
                on_doc=lambda: progress.advance(task),
                workers=workers,
                executor=executor,
            )
        save(idx, out_path, fmt)
    except CorpusError as exc:
        raise _fail(str(exc)) from exc
    except OSError as exc:
        raise _fail(f"не вдалося записати {out_path}: {exc}") from exc

    err.print(
        f"[green]готово:[/green] {len(idx.doc_meta)} документів, {len(idx)} слів → {out_path}"
    )


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help='Запит, напр. "(кіт OR пес) AND NOT миша".')],
    index_path: Annotated[Path, typer.Option("--index", "-i", help="Файл індексу.")] = Path(
        "index.bin"
    ),
    k: Annotated[int, typer.Option("-k", help="Скільки результатів показати.")] = 10,
    scorer: Annotated[ScorerName, typer.Option(help="Формула ранжування.")] = "bm25",
    json_out: Annotated[bool, typer.Option("--json", help="Вивести JSON замість таблиці.")] = False,
    width: Annotated[int, typer.Option(help="Ширина уривка в символах.")] = 80,
    source: Annotated[
        Path | None,
        typer.Option(help="JSONL від findex crawl: звідти беру текст для уривків."),
    ] = None,
) -> None:
    """Шукає в збереженому індексі й показує топ-k результатів."""
    try:
        with open_index(index_path) as idx:
            searcher = Searcher(idx, make_scorer(scorer))
            results = searcher.search(query, k)
            terms = searcher.terms(query)
            rows = [(r, snippet_for(idx, r.doc_id, terms, width, source)) for r in results]
    except IndexLoadError as exc:
        raise _fail(str(exc)) from exc
    except QueryError as exc:
        raise _fail(f"поганий запит: {exc}") from exc

    if json_out:
        payload = [
            {"doc_id": r.doc_id, "score": round(r.score, 4), "title": r.title, "snippet": s}
            for r, s in rows
        ]
        # ensure_ascii=False, щоб українські літери лишались літерами
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return

    if not rows:
        out.print("[yellow]нічого не знайдено[/yellow]")
        return

    table = Table(title=f"«{query}» — {len(rows)} результатів")
    table.add_column("#", justify="right")
    table.add_column("бал", justify="right")
    table.add_column("документ")
    table.add_column("уривок", overflow="fold")
    for i, (r, snip) in enumerate(rows, 1):
        table.add_row(str(i), f"{r.score:.3f}", r.title, snip)
    out.print(table)


@app.command("stats")
def stats_cmd(
    corpus: Annotated[Path, typer.Argument(help="Папка з .txt або файл .jsonl.")],
    limit: Annotated[int | None, typer.Option(help="Взяти лише перші N документів.")] = None,
    eager: Annotated[bool, typer.Option("--eager", help="Жадібний режим (списки).")] = False,
    top: Annotated[int, typer.Option(help="Скільки найчастіших слів показати.")] = 10,
    json_out: Annotated[bool, typer.Option("--json", help="Вивести JSON.")] = False,
) -> None:
    """Статистика корпусу: документи, слова, словник, найчастіші слова."""
    if not corpus.exists():
        raise _fail(f"корпус не знайдено: {corpus}")
    try:
        st = collect_stats(corpus, limit, eager, top)
    except CorpusError as exc:
        raise _fail(str(exc)) from exc

    if json_out:
        payload = {
            "mode": st.mode,
            "documents": st.documents,
            "tokens": st.tokens,
            "vocabulary": st.vocabulary,
            "elapsed_s": round(st.elapsed_s, 4),
            "peak_bytes": st.peak_bytes,
            "top": [{"word": w, "count": c} for w, c in st.top],
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return

    out.print(
        f"режим: {st.mode}\nдокументів: {st.documents}\nслів: {st.tokens}\n"
        f"у словнику: {st.vocabulary}\nчас: {st.elapsed_s:.2f} с, "
        f"пік пам'яті: {st.peak_bytes / 1024:.0f} КіБ"
    )
    table = Table(title="найчастіші слова")
    table.add_column("слово")
    table.add_column("разів", justify="right")
    for word, count in st.top:
        table.add_row(word, str(count))
    out.print(table)


# ---------------------------------------------------------------- crawl (лаба 6)
class _SlowCallbackCounter(logging.Handler):
    """Рахує попередження asyncio у debug-режимі: "Executing ... took 0.3 seconds"."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "took" in record.getMessage():
            self.count += 1


def _render_stats(stats: CrawlStats) -> Table:
    """Таблиця з живими лічильниками, яку перемальовує rich."""
    t = Table.grid(padding=(0, 2))
    t.add_row("сторінок зібрано", f"[bold green]{stats.pages}[/]")
    t.add_row("сторінок за секунду", f"{stats.pages_per_second:.1f}")
    t.add_row("запитів чекає відповіді", f"{stats.in_flight}")
    t.add_row("помилок", f"[red]{stats.errors}[/]" if stats.errors else "0")
    t.add_row("повторів", f"{stats.retries}")
    t.add_row("глибина черги", f"{stats.queue_depth}")
    t.add_row("бачили адрес", f"{stats.discovered}")
    t.add_row("минуло, с", f"{stats.elapsed:.1f}")
    return t


def _append_line(f: IO[str], line: str) -> None:
    """Дописує один рядок у файл і одразу скидає на диск (викликається з потоку)."""
    f.write(line)
    f.flush()


async def _crawl_to_file(
    urls: list[str], out_path: Path, stats: CrawlStats, **options: object
) -> None:
    """Пише сторінки в JSONL по одній, у міру надходження (всі в пам'яті не тримаю)."""
    # створюю папку для файлу, якщо її ще немає (наприклад data/)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        # aclosing гарантує, що генератор закриється навіть при помилці
        async with contextlib.aclosing(crawl(urls, stats=stats, **options)) as pages:  # type: ignore[arg-type]  # options зібрані як словник object, pyright не знає типів окремих ключів
            async for page in pages:
                record = {
                    "id": page.url,
                    "url": page.url,
                    "status": page.status,
                    "depth": page.depth,
                    "fetched_at": page.fetched_at,
                    "title": page.title,
                    # перший рядок тексту — заголовок: його індекс бере як title
                    "text": f"{page.title}\n{page.text}",
                }
                # запис у файл блокувальний, тому роблю його в потоці, щоб не гальмувати цикл
                line = json.dumps(record, ensure_ascii=False) + "\n"
                await asyncio.to_thread(_append_line, f, line)


@app.command("crawl")
def crawl_cmd(
    urls: Annotated[list[str], typer.Argument(help="Початкові адреси (http або https).")],
    contact: Annotated[
        str, typer.Option(help="Ваша адреса для зв'язку (e-mail або URL), піде в User-Agent.")
    ],
    max_pages: Annotated[int, typer.Option(min=1, help="Стеля на число сторінок.")] = 500,
    concurrency: Annotated[int, typer.Option(min=1, help="Скільки запитів одночасно.")] = 10,
    per_host: Annotated[
        int, typer.Option(min=1, help="Скільки запитів одночасно до одного хоста.")
    ] = 2,
    delay: Annotated[
        float, typer.Option(min=0.0, help="Мінімальна пауза між запитами до хоста, с.")
    ] = 0.2,
    max_depth: Annotated[
        int, typer.Option(min=0, help="Наскільки глибоко йти за посиланнями.")
    ] = 3,
    timeout: Annotated[float, typer.Option(min=0.1, help="Таймаут одного запиту, с.")] = 10.0,
    out_path: Annotated[Path, typer.Option("--out", "-o", help="JSONL зі сторінками.")] = Path(
        "data/crawl.jsonl"
    ),
    log_file: Annotated[Path, typer.Option("--log", help="Файл зі статусом кожного URL.")] = Path(
        "crawl.log"
    ),
    summary: Annotated[
        Path | None, typer.Option(help="Записати підсумок у JSON (для вимірів).")
    ] = None,
    allow_host: Annotated[
        list[str] | None, typer.Option(help="Додатковий хост (з портом), куди можна ходити.")
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Режим налагодження asyncio: шукає блокування циклу.")
    ] = False,
) -> None:
    """Обходить сайт асинхронно і пише сторінки в JSONL (потім: findex index crawl.jsonl)."""
    starts = [c for u in urls if (c := canonicalize(u)) is not None]
    if not starts:
        raise _fail("жодної правильної http(s)-адреси")
    try:
        version = metadata.version("findex")
    except metadata.PackageNotFoundError:
        version = "dev"
    # чесний User-Agent з адресою для зв'язку
    user_agent = f"findex-crawler/{version} (student course project; contact: {contact})"
    hosts = {host_of(u) for u in starts} | set(allow_host or [])

    # лог краулера йде у файл, а не на екран (там живі лічильники)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    slow = _SlowCallbackCounter()
    watched = [logging.getLogger("findex.crawl"), logging.getLogger("asyncio")]
    saved = [(lg, lg.level, lg.propagate) for lg in watched]
    for lg in watched:
        lg.addHandler(file_handler)
        lg.propagate = False
        lg.setLevel(logging.INFO if lg.name == "findex.crawl" else logging.WARNING)
    logging.getLogger("asyncio").addHandler(slow)

    stats = CrawlStats()
    t0 = time.perf_counter()
    try:
        with Live(get_renderable=lambda: _render_stats(stats), console=err, refresh_per_second=4):
            asyncio.run(
                _crawl_to_file(
                    starts,
                    out_path,
                    stats,
                    user_agent=user_agent,
                    max_pages=max_pages,
                    concurrency=concurrency,
                    per_host=per_host,
                    delay=delay,
                    max_depth=max_depth,
                    timeout=timeout,
                    allowed_hosts=hosts,
                ),
                debug=debug,
            )
    except OSError as exc:
        raise _fail(f"не вдалося записати файл: {exc}") from exc
    finally:
        for lg, level, propagate in saved:
            lg.removeHandler(file_handler)
            lg.setLevel(level)
            lg.propagate = propagate
        logging.getLogger("asyncio").removeHandler(slow)
        file_handler.close()

    wall = time.perf_counter() - t0
    err.print(
        f"[green]готово:[/green] {stats.pages} сторінок за {wall:.1f} с "
        f"({stats.pages_per_second:.1f}/с), помилок {stats.errors}, "
        f"повторів {stats.retries} → {out_path}"
    )
    if debug:
        err.print(f"debug: повільних колбеків (блокування циклу): {slow.count}")
    if summary is not None:
        payload = {
            "pages": stats.pages,
            "errors": stats.errors,
            "retries": stats.retries,
            "skipped": stats.skipped,
            "discovered": stats.discovered,
            "bytes": stats.nbytes,
            "wall": round(wall, 3),
            "pages_per_second": round(stats.pages_per_second, 3),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "slow_callbacks": slow.count,
            "concurrency": concurrency,
        }
        summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
