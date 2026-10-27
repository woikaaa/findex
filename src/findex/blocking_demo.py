"""
дослід: що буде, якщо важку роботу зробити всередині корутини (лаба 6).

Поки один «годинник» (ticker) раз на 10 мс записує час, я запускаю індексацію
двома способами:
  1) прямо в корутині — цикл подій стоїть, годинник мовчить;
  2) через asyncio.to_thread — цикл вільний, годинник тикає.
Показую найбільшу паузу між тиками (так виглядає «замерзлий» краулер).

    python -m findex.blocking_demo data/
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from findex.index import build_index


async def _ticker(gaps: list[float], stop: asyncio.Event) -> None:
    last = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(0.01)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now


async def measure(label: str, corpus: Path, threaded: bool) -> None:
    gaps: list[float] = []
    stop = asyncio.Event()
    tick = asyncio.create_task(_ticker(gaps, stop))
    await asyncio.sleep(0.05)  # дати годиннику розігнатись
    if threaded:
        await asyncio.to_thread(build_index, corpus)
    else:
        build_index(corpus)  # БЛОКУЄ цикл: жодна інша корутина не рухається
    await asyncio.sleep(0.05)
    stop.set()
    await tick
    print(f"{label:34} найбільша пауза між тиками: {max(gaps) * 1000:7.0f} мс")


async def main_async(corpus: Path) -> None:
    await measure("індексація ВСЕРЕДИНІ корутини", corpus, threaded=False)
    await measure("індексація в потоці (to_thread)", corpus, threaded=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("corpus", type=Path)
    args = ap.parse_args()
    asyncio.run(main_async(args.corpus), debug=True)  # debug покаже і повідомлення asyncio


if __name__ == "__main__":
    main()
