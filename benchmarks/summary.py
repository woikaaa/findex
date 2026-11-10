"""Таблиця з результатів pytest-benchmark: мілісекунди лаби 3, мілісекунди NumPy, прискорення.

python benchmarks/summary.py bench.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("використання: python benchmarks/summary.py bench.json")
    data = cast("dict[str, Any]", json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    # група -> {"lab3" або "numpy": медіана в мс, ...}
    groups: dict[str, dict[str, float]] = {}
    queries: dict[str, str] = {}
    other: list[tuple[str, float]] = []
    for b in cast("list[dict[str, Any]]", data["benchmarks"]):
        name = str(b["name"])
        ms = float(b["stats"]["median"]) * 1000
        group = str(b.get("group") or "")
        if group.startswith("search-"):
            engine = "lab3" if "lab3" in name else "numpy"
            groups.setdefault(group, {})[engine] = ms
            queries[group] = str(b.get("extra_info", {}).get("query", ""))
        else:
            other.append((name, ms))
    print("| випадок | запит | лаба 3, мс | NumPy, мс | прискорення |")
    print("| ------- | ----- | ---------- | --------- | ----------- |")
    for group in ("search-frequent", "search-rare", "search-three"):
        if group not in groups:
            continue
        a, c = groups[group]["lab3"], groups[group]["numpy"]
        print(f"| {group[7:]} | `{queries[group]}` | {a:.4f} | {c:.4f} | {a / c:.1f}x |")
    print("\n| інше | медіана, мс |")
    print("| ---- | ----------- |")
    for name, ms in other:
        print(f"| {name} | {ms:.3f} |")


if __name__ == "__main__":
    main()
