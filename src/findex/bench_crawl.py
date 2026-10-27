"""
вимірювання краулера для concurrency 1, 5 і 20 (лаба 6).

Той самий початковий URL і той самий max_pages для кожної клітинки. Кожен запуск —
окремий процес `findex crawl`, щоб пік пам'яті (RSS) не змішувався.

Запуск на моєму демо-сайті (він стартує сам у цьому ж скрипті):
    python -m findex.bench_crawl --demo --contact me@example.com

Запуск на іншому сайті (лише якщо його robots.txt або власник це дозволяє):
    python -m findex.bench_crawl --url https://мій-сайт/ --contact me@example.com
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from findex.demosite import start_server
from findex.sysinfo import cpu_model


def run_once(url: str, contact: str, concurrency: int, a: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        summary = Path(tmp) / "summary.json"
        cmd = [
            sys.executable, "-m", "findex", "crawl", url,
            "--contact", contact,
            "--max-pages", str(a.max_pages),
            "--concurrency", str(concurrency),
            "--per-host", str(a.per_host if a.per_host else concurrency),
            "--delay", str(a.delay),
            "--max-depth", str(a.max_depth),
            "--out", str(Path(tmp) / "c.jsonl"),
            "--log", str(Path(tmp) / "c.log"),
            "--summary", str(summary),
        ]  # fmt: skip
        # timeout: якщо запуск завис, не чекаю вічно, а показую, що він встиг написати
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(
                f"concurrency={concurrency}: запуск завис довше {a.timeout} с\n{exc.stderr}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"concurrency={concurrency}: краулер упав\n{exc.stderr}") from exc
        return json.loads(summary.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", help="початковий URL (якщо без --demo)")
    ap.add_argument("--demo", action="store_true", help="запустити власний демо-сайт")
    ap.add_argument("--contact", required=True)
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--per-host", type=int, default=0, help="0 = дорівнює concurrency")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--latency", type=float, default=0.05, help="затримка демо-сайту, с")
    ap.add_argument("--timeout", type=float, default=180.0, help="ліміт на один запуск, с")
    ap.add_argument("--out", type=Path, default=Path("docs"))
    a = ap.parse_args()

    if a.demo:
        server, _ = start_server(port=0, pages=max(500, a.max_pages * 2), latency=a.latency)
        url = f"http://127.0.0.1:{server.server_port}/"
        site = f"власний демо-сайт (localhost, затримка {a.latency * 1000:.0f} мс на відповідь)"
    elif a.url:
        url, site = a.url, a.url
    else:
        ap.error("потрібен --demo або --url")

    rows: list[dict[str, Any]] = []
    for c in a.concurrency:
        runs: list[dict[str, Any]] = []
        for r in range(a.repeats):
            runs.append(run_once(url, a.contact, c, a))
            print(f"  concurrency={c:<3} повтор {r + 1}/{a.repeats}: {runs[-1]['wall']:.2f} с",
                  file=sys.stderr)  # fmt: skip
        rows.append(
            {
                "concurrency": c,
                **{
                    k: statistics.median(x[k] for x in runs)
                    for k in ("wall", "pages_per_second", "errors", "peak_rss_mb", "pages")
                },
            }
        )

    lines = [
        f"Машина: {cpu_model()}, ядер: {os.cpu_count()}, Python {platform.python_version()}.",
        f"Сайт: {site}. max_pages = {a.max_pages}, пауза на хост = {a.delay} с, "
        f"ліміт на хост = {a.per_host or 'дорівнює concurrency'}. Медіана з {a.repeats} запусків.",
        "",
        "| concurrency | сторінок | wall, с | сторінок/с | помилок | пік RSS, МБ |",
        "| ----------- | -------- | ------- | ---------- | ------- | ----------- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['concurrency']} | {r['pages']:.0f} | {r['wall']:.2f} | "
            f"{r['pages_per_second']:.1f} | {r['errors']:.0f} | {r['peak_rss_mb']:.0f} |"
        )
    md = "\n".join(lines) + "\n"
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "crawl_bench.md").write_text(md, encoding="utf-8")
    (a.out / "crawl_bench.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
