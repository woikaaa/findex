"""
паралельна збірка індексу (лаба 5).

Ідея проста: корпус ділю на шматки, кожен шматок будує свою частину індексу
(build_partial), а потім усі частини склеюються (merge). Виконавець можна
міняти, а код лишається той самий:

  serial     — звичайний цикл в одному потоці (так було в лабі 4);
  threads    — ThreadPoolExecutor (потоки; під GIL CPU-робота не прискорюється);
  processes  — ProcessPoolExecutor (окремі процеси, кожен зі своїм GIL).
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from findex.index import Partial, build_partial, merge
from findex.model import Index, Storage

# способи виконання. Literal, щоб typer і pyright знали всі можливі значення
Executors = Literal["serial", "threads", "processes"]
EXECUTORS: tuple[Executors, ...] = get_args(Executors)


@dataclass(frozen=True)
class Chunk:
    """Шматок корпусу: які файли і з якого doc_id починається діапазон."""

    first_doc_id: int
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class ParallelResult:
    """Індекс і те, що я виміряла під час збірки."""

    index: Index
    executor: Executors
    workers: int
    chunk_sizes: list[int]  # скільки файлів у кожному шматку
    worker_cpu_s: float  # сума процесорного часу всіх воркерів
    build_s: float  # скільки секунд (за годинником) йшла збірка частин
    merge_s: float  # скільки секунд (за годинником) йшло склеювання
    merge_cpu_s: float  # скільки процесорного часу пішло на склеювання (у батьківському процесі)


def split_into_chunks(paths: Sequence[Path], n_chunks: int) -> list[Chunk]:
    """Ділить файли на n_chunks шматків ПІДРЯД, приблизно однакових за розміром у байтах.

    Шматки йдуть підряд, тому doc_id кожного шматка - це простий діапазон
    [first_doc_id, first_doc_id + len(paths)). Різні шматки не перетинаються."""
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    n_chunks = min(n_chunks, max(1, len(paths)))
    sizes = [_size(p) for p in paths]
    total = sum(sizes)
    chunks: list[Chunk] = []
    start = 0
    done = 0  # скільки байтів уже роздано
    for k in range(1, n_chunks + 1):
        if k == n_chunks:
            end = len(paths)  # останній шматок бере все, що лишилось
        else:
            # іду вперед, поки не набралась k-а частка всіх байтів
            target = total * k / n_chunks
            end = start
            # кожному наступному шматку лишаю хоч по одному файлу
            while end < len(paths) - (n_chunks - k) and done < target:
                done += sizes[end]
                end += 1
            if end == start:
                end = start + 1
                done += sizes[start]
        chunks.append(Chunk(start, tuple(paths[start:end])))
        start = end
    return chunks


def _size(path: Path) -> int:
    """Розмір файла; якщо файл зник, рахуємо як 0."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _build_chunk(chunk: Chunk, storage: Storage, positions: bool) -> Partial:
    """Функція рівня модуля: її можна відправити в інший процес (спосіб запуску spawn).

    У воркер їдуть лише шляхи, а не прочитані тексти: так через pickle
    пересилається мало даних."""
    return build_partial(chunk.paths, chunk.first_doc_id, storage, positions)


def _make_executor(kind: Executors, workers: int) -> Executor | None:
    if kind == "serial":
        return None
    if kind == "threads":
        return ThreadPoolExecutor(max_workers=workers)
    # spawn однаковий на Windows, macOS і Linux; fork на Linux дав би інші результати
    ctx = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(max_workers=workers, mp_context=ctx)


def build_parallel(
    paths: Sequence[Path],
    workers: int = 1,
    executor: Executors = "serial",
    storage: Storage = "slots",
    positions: bool = False,
    on_doc: Callable[[], None] | None = None,
) -> ParallelResult:
    """Будує індекс з файлів `paths` вибраним способом.

    Якщо воркер впав, помилка піднімається тут із його стеком, а інші завдання
    скасовуються: збірка не зависає."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    chunks = split_into_chunks(paths, workers)
    t0 = time.perf_counter()
    parts: list[Partial | None] = [None] * len(chunks)

    pool = _make_executor(executor, workers)
    if pool is None:
        # звичайний цикл: прогрес оновлюється після кожного файла
        for i, ch in enumerate(chunks):
            parts[i] = build_partial(ch.paths, ch.first_doc_id, storage, positions, on_doc)
    else:
        futures: dict[Future[Partial], int] = {}
        try:
            for i, ch in enumerate(chunks):
                futures[pool.submit(_build_chunk, ch, storage, positions)] = i
            for fut in as_completed(futures):
                i = futures[fut]
                parts[i] = fut.result()  # якщо воркер впав, виняток летить сюди
                if on_doc is not None:
                    for _ in chunks[i].paths:  # шматок готовий: рахую всі його файли
                        on_doc()
        except BaseException:
            # скасовую те, що ще не почалось, і не чекаю решту
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown()
    build_s = time.perf_counter() - t0

    done = [p for p in parts if p is not None]
    t1 = time.perf_counter()
    c1 = time.process_time()
    index = merge(done)  # склеюємо в порядку зростання doc_id
    merge_cpu = time.process_time() - c1
    merge_s = time.perf_counter() - t1

    return ParallelResult(
        index=index,
        executor=executor,
        workers=workers,
        chunk_sizes=[len(c.paths) for c in chunks],
        worker_cpu_s=sum(p.cpu_seconds for p in done),
        build_s=build_s,
        merge_s=merge_s,
        merge_cpu_s=merge_cpu,
    )
