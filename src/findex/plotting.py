# pyright: basic
"""малювання графіка прискорення (matplotlib не має повних типів, тому тут не strict)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def draw_graph(rows: list[dict[str, Any]], path: Path) -> None:
    """Графік: прискорення від числа воркерів + пряма ідеального зростання."""
    import matplotlib

    matplotlib.use("Agg")  # без вікна, лише файл
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    top = max(r["workers"] for r in rows)
    ax.plot([1, top], [1, top], linestyle="--", color="#888888", label="ідеально (лінійно)")
    colors = {"threads": "#c0392b", "processes": "#1f77b4"}
    for kind in ("threads", "processes"):
        pts = sorted((r["workers"], r["speedup"]) for r in rows if r["executor"] == kind)
        if pts:
            ax.plot(*zip(*pts, strict=True), marker="o", color=colors[kind], label=kind)
    ax.axhline(1.0, color="#cccccc", linewidth=0.8)
    ax.set_xlabel("число воркерів")
    ax.set_ylabel("прискорення відносно serial")
    ax.set_title("Паралельна індексація: прискорення")
    ax.set_xticks(sorted({r["workers"] for r in rows}))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
