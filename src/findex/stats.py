from __future__ import annotations

import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def run_lazy(root: Path, limit: int | None) -> Counter[str]:
    # беру документи по одному (ледачо)
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)  # усе ще ліниво: бере лише `limit` документів

    # генераторний вираз: слова течуть по одному з документа в Counter
    token_stream = (tok for doc in docs for tok in tokenize(doc.text))
    # Counter сам рахує, скільки разів зустрівся кожен токен
    return Counter(token_stream)


def run_eager(root: Path, limit: int | None) -> Counter[str]:
    docs = list(iter_documents(root))  # (1) увесь корпус у пам'яті
    if limit is not None:
        docs = docs[:limit]

    all_tokens: list[list[str]] = []
    for doc in docs:
        all_tokens.append(list(tokenize(doc.text)))  # (2) слова кожного документа списком

    flat: list[str] = []
    for doc_tokens in all_tokens:
        flat.extend(doc_tokens)  # (3) один гігантський список усіх слів

    # рахую тільки в кінці, коли все вже в пам'яті
    return Counter(flat)  # (4) рахуємо лише в самому кінці


def _doc_count(root: Path, limit: int | None) -> int:
    """Рахує документи (проходить по них ліниво, нічого не зберігаючи)."""
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)
    # просто додаю одиницю за кожен документ
    return sum(1 for _ in docs)


@dataclass(frozen=True)
class CorpusStats:
    """Підсумок по корпусу: те, що друкує `findex stats`."""

    mode: str
    documents: int
    tokens: int
    vocabulary: int
    elapsed_s: float
    peak_bytes: int
    top: list[tuple[str, int]]


def collect_stats(
    root: Path, limit: int | None = None, eager: bool = False, top: int = 50
) -> CorpusStats:
    """Рахує статистику корпусу ледачим або жадібним способом і міряє час та пам'ять."""
    # вмикаємо вимірювання пам'яті й засікаємо час
    tracemalloc.start()
    start = time.perf_counter()

    counts = run_eager(root, limit) if eager else run_lazy(root, limit)

    # рахую, скільки секунд пройшло
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()  # нас цікавить лише пік
    # зупиняю вимірювання пам'яті
    tracemalloc.stop()

    return CorpusStats(
        mode="eager (lists)" if eager else "lazy (generators)",
        documents=_doc_count(root, limit),
        tokens=sum(counts.values()),  # усього слів
        vocabulary=len(counts),  # різних слів
        elapsed_s=elapsed,
        peak_bytes=peak,
        top=counts.most_common(top),  # найчастіші слова
    )
