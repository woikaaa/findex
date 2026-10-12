from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def run_lazy(root: Path, limit: int | None) -> Counter:
    # беру документи по одному (ледачо)
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)  # усе ще ліниво: бере лише `limit` документів

    # генераторний вираз: слова течуть по одному з документа в Counter
    token_stream = (tok for doc in docs for tok in tokenize(doc.text))
    # Counter сам рахує, скільки разів зустрівся кожен токен
    return Counter(token_stream)


def run_eager(root: Path, limit: int | None) -> Counter:
    docs = list(iter_documents(root))          # (1) увесь корпус у пам'яті
    if limit is not None:
        docs = docs[:limit]

    all_tokens: list[list[str]] = []
    for doc in docs:
        all_tokens.append(list(tokenize(doc.text)))  # (2) слова кожного документа списком

    flat: list[str] = []
    for doc_tokens in all_tokens:
        flat.extend(doc_tokens)                 # (3) один гігантський список усіх слів

    # рахую тільки в кінці, коли все вже в пам'яті
    return Counter(flat)                        # (4) рахуємо лише в самому кінці


def _doc_count(root: Path, limit: int | None) -> int:
    """Рахує документи (проходить по них ліниво, нічого не зберігаючи)."""
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)
    # просто додаю одиницю за кожен документ
    return sum(1 for _ in docs)


def main() -> None:
    # аргументи командного рядка
    parser = argparse.ArgumentParser(description="findex corpus statistics")
    parser.add_argument("root", type=Path, help="corpus directory or .jsonl file")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only process the first N documents (uses itertools.islice)",
    )
    parser.add_argument(
        "--eager", action="store_true",
        help="use the deliberately greedy, list-based version (for M4 comparison)",
    )
    parser.add_argument(
        "--top", type=int, default=50, help="how many top terms to print",
    )
    # зчитую те, що я написала в терміналі
    args = parser.parse_args()

    # вмикаємо вимірювання пам'яті й засікаємо час
    tracemalloc.start()
    start = time.perf_counter()

    counts = run_eager(args.root, args.limit) if args.eager else run_lazy(args.root, args.limit)

    # рахую, скільки секунд пройшло
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()  # нас цікавить лише пік
    # зупиняю вимірювання пам'яті
    tracemalloc.stop()

    n_docs = _doc_count(args.root, args.limit)
    n_tokens = sum(counts.values())   # усього слів
    vocab_size = len(counts)          # різних слів

    # підпис режиму для виводу
    mode = "eager (lists)" if args.eager else "lazy (generators)"
    print(f"mode:              {mode}")
    print(f"documents:         {n_docs}")
    print(f"total tokens:      {n_tokens}")
    print(f"vocabulary size:   {vocab_size}")
    print(f"elapsed:           {elapsed:.4f} s")
    print(f"peak memory:       {peak / 1024:.1f} KiB  ({peak} bytes)")
    print()
    print(f"top {args.top} terms:")
    # друкую найчастіші слова
    for term, count in counts.most_common(args.top):
        print(f"  {term:<20} {count}")


if __name__ == "__main__":
    main()
