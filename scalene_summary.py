"""
scalene_summary.py: короткий підсумок профілю Scalene (лаба 8).

    scalene run --cpu-only -o docs/scalene_search.json profile_run.py search data/wiki.bin --plain
    python scalene_summary.py docs/scalene_search.json

Показує функції проєкту (файли з findex у шляху) за часом процесора і ділить цей час
на Python (повільний інтерпретатор), native (код на C: re, math, вбудовані функції) і system.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("використання: python scalene_summary.py файл.json [скільки_рядків]")
    data = cast("dict[str, Any]", json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rows: list[tuple[float, float, float, str]] = []
    for filename, info in cast("dict[str, Any]", data["files"]).items():
        # беру лише код проєкту (і сам скрипт запуску), бібліотеки Python пропускаю
        if "findex" not in filename and "profile_run" not in filename:
            continue
        short = filename.replace("\\", "/").split("/")[-1]
        for fn in cast("list[dict[str, Any]]", info.get("functions", [])):
            py = float(fn.get("n_cpu_percent_python", 0.0))
            nat = float(fn.get("n_cpu_percent_c", 0.0))
            sysp = float(fn.get("n_sys_percent", 0.0))
            rows.append((py + nat + sysp, py, nat, f"{short}:{fn['line']}"))
    rows.sort(reverse=True)
    print(f"{'функція':45} {'всього %':>9} {'Python %':>9} {'native %':>9}")
    for total, py, nat, name in rows[:top]:
        print(f"{name:45} {total:9.1f} {py:9.1f} {nat:9.1f}")


if __name__ == "__main__":
    main()
