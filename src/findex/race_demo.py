"""
демонстрація гонитви (race condition) між потоками (лаба 5).

Кілька потоків додають по 1 до спільного лічильника. Операція `counter += 1` — це
три кроки: прочитати, додати, записати. Якщо інтерпретатор перемкне потік між
кроками, чужий приріст перезапишеться і втратиться.

Запуск:
    python -m findex.race_demo
    python -m findex.race_demo --threads 8 --n 500000

Порівняйте звичайний Python (GIL) і збірку 3.13t без GIL:
    uv run --python 3.13t python -m findex.race_demo
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections import Counter
from collections.abc import Callable


def run_threads(work: Callable[[], None], n_threads: int) -> None:
    """Запускає work в n_threads потоках і чекає на всіх."""
    threads = [threading.Thread(target=work) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def race_int(n_threads: int, n: int, use_lock: bool) -> int:
    """Спільне число, яке всі потоки збільшують. Повертає, скільки оновлень ВТРАЧЕНО."""
    box = [0]  # число в списку, щоб потоки бачили одне й те саме
    lock = threading.Lock()

    def work() -> None:
        for _ in range(n):
            if use_lock:
                with lock:  # лише один потік за раз виконує цей блок
                    box[0] += 1
            else:
                box[0] += 1  # прочитати, додати, записати: тут і буває гонитва

    run_threads(work, n_threads)
    return n_threads * n - box[0]


def race_counter(n_threads: int, n: int, use_lock: bool) -> int:
    """Те саме зі спільним Counter слів (як у лабі 1). Повертає втрачені оновлення."""
    counts: Counter[str] = Counter()
    lock = threading.Lock()

    def work() -> None:
        for _ in range(n):
            if use_lock:
                with lock:
                    counts["слово"] += 1
            else:
                counts["слово"] += 1

    run_threads(work, n_threads)
    return n_threads * n - counts["слово"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=200_000, help="скільки разів кожен потік додає 1")
    ap.add_argument("--rounds", type=int, default=3, help="скільки разів повторити дослід")
    ap.add_argument("--switch", type=float, default=None, help="sys.setswitchinterval, секунди")
    args = ap.parse_args()

    if args.switch is not None:
        sys.setswitchinterval(args.switch)  # частіше перемикати потоки: гонитву видно легше

    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"Python {sys.version.split()[0]}, sys._is_gil_enabled() = {gil}")
    print(f"потоків: {args.threads}, оновлень на потік: {args.n}, очікуємо {args.threads * args.n}")
    for name, fn in (("int", race_int), ("Counter", race_counter)):
        for use_lock in (False, True):
            lost = [fn(args.threads, args.n, use_lock) for _ in range(args.rounds)]
            tag = "з Lock   " if use_lock else "без Lock "
            print(f"{name:8} {tag} втрачено оновлень за {args.rounds} запусків: {lost}")


if __name__ == "__main__":
    main()
