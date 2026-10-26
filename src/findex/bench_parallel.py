from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from findex.corpus import list_corpus_files
from findex.parallel import Executors, build_parallel


# ----------------------------------------------------------------- одна клітинка
def peak_rss_mb() -> float:
    """Пік пам'яті цього процесу (RSS) у мегабайтах."""
    if sys.platform == "win32":
        try:
            import psutil  # на Windows немає модуля resource, тому беру psutil
        except ImportError:
            return float("nan")  # psutil немає (наприклад у 3.13t без dev-залежностей)
        return psutil.Process().memory_info().peak_wset / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # на Linux ru_maxrss у кілобайтах, на macOS у байтах
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


def run_cell(
    corpus: Path, executor: Executors, workers: int, storage: str
) -> dict[str, float]:
    """Одна збірка з вимірами. Викликається в окремому процесі."""
    paths = list_corpus_files(corpus)
    res = build_parallel(paths, workers, executor, storage, False)  # type: ignore[arg-type]  # executor і storage приходять із командного рядка як str
    return {
        "wall": res.build_s + res.merge_s,
        "cpu": res.worker_cpu_s + res.merge_cpu_s,
        "merge": res.merge_s,
        "rss_mb": peak_rss_mb(),
    }


def _run_cell_subprocess(
    corpus: Path, executor: str, workers: int, storage: str
) -> dict[str, float]:
    cmd = [
        sys.executable, "-m", "findex.bench_parallel", "cell",
        str(corpus), executor, str(workers), storage,
    ]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------- опис машини
def cpu_model() -> str:
    """Назва процесора (щоб записати її в README)."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gil_status() -> str:
    """Чи ввімкнений GIL у цьому інтерпретаторі (3.13t без GIL відповідає False)."""
    check = getattr(sys, "_is_gil_enabled", None)
    return "GIL увімкнено" if check is None or check() else "GIL ВИМКНЕНО"


# ----------------------------------------------------------------------- таблиця
def default_workers() -> list[int]:
    cores = os.cpu_count() or 1
    return sorted({1, 2, 4, 8, cores})


def collect(
    corpus: Path, executors: list[str], workers: list[int], repeats: int, storage: str
) -> list[dict[str, Any]]:
    """Міряє всі клітинки й повертає рядки з медіанами."""
    rows: list[dict[str, Any]] = []
    plan: list[tuple[str, int]] = [("serial", 1)]
    plan += [(e, w) for e in executors for w in workers]
    print("  прогрів (цей запуск не рахується)...", file=sys.stderr)
    _run_cell_subprocess(corpus, "serial", 1, storage)
    for executor, w in plan:
        runs: list[dict[str, float]] = []
        for r in range(repeats):
            runs.append(_run_cell_subprocess(corpus, executor, w, storage))
            print(f"  {executor:9} workers={w:<2} повтор {r + 1}/{repeats}: "
                  f"{runs[-1]['wall']:.2f} с", file=sys.stderr)  # fmt: skip
        rows.append(
            {
                "executor": executor,
                "workers": w,
                **{k: statistics.median(x[k] for x in runs) for k in runs[0]},
            }
        )
    base = rows[0]["wall"]
    for row in rows:
        row["speedup"] = base / row["wall"]  # прискорення відносно serial
    return rows


def to_markdown(rows: list[dict[str, Any]], corpus: Path, repeats: int) -> str:
    n_files = len(list_corpus_files(corpus))
    lines = [
        f"Машина: {cpu_model()}, ядер (os.cpu_count): {os.cpu_count()}, "
        f"Python {platform.python_version()} ({gil_status()}).",
        f"Корпус: {n_files} файлів. Кожна клітинка — медіана з {repeats} запусків.",
        "",
        "| виконавець | воркерів | wall, с | CPU, с | merge, с | RSS батька, МБ | прискорення |",
        "| ---------- | -------- | ------- | ------ | -------- | -------------- | ----------- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['executor']} | {r['workers']} | {r['wall']:.2f} | {r['cpu']:.2f} | "
            f"{r['merge']:.2f} | {r['rss_mb']:.0f} | {r['speedup']:.2f}× |"
        )
    lines += _amdahl_lines(rows)
    return "\n".join(lines) + "\n"


def _amdahl_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Закон Амдала: merge йде в одному процесі, тому він обмежує прискорення зверху."""
    procs = [r for r in rows if r["executor"] == "processes"]
    if not procs:
        return []
    last = max(procs, key=lambda r: r["workers"])
    serial_wall = rows[0]["wall"]
    s = min(
        1.0, last["merge"] / serial_wall
    )  # частка роботи, яку не можна розпаралелити
    n = last["workers"]
    limit = 1 / (s + (1 - s) / n)
    return [
        "",
        f"Закон Амдала: послідовна частина (merge при {n} воркерах) = {s:.1%} від wall serial, "
        f"тому з {n} воркерами прискорення не більше ніж {limit:.2f}× "
        f"(формула 1 / (s + (1 − s) / N)); реально вийшло {last['speedup']:.2f}×.",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser(
        "run", help="виміряти всі клітинки і записати таблицю та графік"
    )
    run.add_argument("corpus", type=Path)
    run.add_argument("--workers", type=int, nargs="+", default=default_workers())
    run.add_argument("--executors", nargs="+", default=["threads", "processes"])
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--storage", default="slots")
    run.add_argument("--out", type=Path, default=Path("docs"))

    cell = sub.add_parser("cell", help="(службова) одна клітинка, JSON у stdout")
    cell.add_argument("corpus", type=Path)
    cell.add_argument("executor")
    cell.add_argument("workers", type=int)
    cell.add_argument("storage")

    args = ap.parse_args()
    if args.cmd == "cell":
        print(
            json.dumps(run_cell(args.corpus, args.executor, args.workers, args.storage))
        )
        return

    rows = collect(
        args.corpus, args.executors, args.workers, args.repeats, args.storage
    )
    md = to_markdown(rows, args.corpus, args.repeats)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bench.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.out / "bench.md").write_text(md, encoding="utf-8")
    from findex.plotting import draw_graph

    draw_graph(rows, args.out / "speedup.png")
    print(md)
    print(f"записано: {args.out}/bench.md, bench.json, speedup.png")


if __name__ == "__main__":
    main()
