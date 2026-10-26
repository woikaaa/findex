"""
командний рядок findex: три команди — index, search, stats.

Усі команди друкують результат у stdout, а повідомлення про хід роботи (логи,
прогрес) — у stderr. Так вивід можна передати в інший інструмент через канал.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from findex.corpus import CorpusError, list_corpus_files
from findex.index import build_index
from findex.model import Storage
from findex.query import QueryError
from findex.ranking import Searcher
from findex.scoring import ScorerName, make_scorer
from findex.snippets import snippet_for
from findex.stats import collect_stats
from findex.store import Fmt, IndexLoadError, open_index, save

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
) -> None:
    """Будує індекс і зберігає його у файл."""
    if not corpus.exists():
        raise _fail(f"корпус не знайдено: {corpus}")
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
) -> None:
    """Шукає в збереженому індексі й показує топ-k результатів."""
    try:
        with open_index(index_path) as idx:
            searcher = Searcher(idx, make_scorer(scorer))
            results = searcher.search(query, k)
            terms = searcher.terms(query)
            rows = [(r, snippet_for(idx, r.doc_id, terms, width)) for r in results]
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


if __name__ == "__main__":
    app()
