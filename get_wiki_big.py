"""
get_wiki_big.py: докачує ВЕЛИКИЙ корпус статей української Вікіпедії (лаба 8, ~10 разів більше).

    python get_wiki_big.py                 # до 35 МБ у папку data/wiki10x
    python get_wiki_big.py --mb 20         # менше

Бере випадкові статті (не менше 5000 символів), поки не набереться потрібний обсяг.
Файли, що вже є, не качає вдруге, тож скрипт можна зупинити (Ctrl+C) і запустити знову.
Між запитами пауза, щоб Вікіпедія не блокувала.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PAUSE = 1.5  # секунд між запитами
MIN_CHARS = 5000  # дрібні статті пропускаю
API = "https://uk.wikipedia.org/w/api.php"


def fetch_random(limit: int = 20) -> dict[str, dict[str, object]]:
    """Один запит: `limit` випадкових статей разом із повним текстом."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": limit,
            "prop": "extracts",
            "explaintext": 1,
            "exlimit": limit,
            "format": "json",
        }
    )
    req = urllib.request.Request(
        f"{API}?{params}", headers={"User-Agent": "findex-lab/1.0 (student project)"}
    )
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = json.load(resp)
            return data.get("query", {}).get("pages", {})
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 5:
                wait = 10 * attempt
                print(f"   429, чекаю {wait} с (спроба {attempt})...")
                time.sleep(wait)
                continue
            raise
    return {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mb", type=float, default=35.0, help="скільки МБ набрати")
    p.add_argument("--out", type=Path, default=Path("data/wiki10x"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    total = sum(f.stat().st_size for f in a.out.glob("*.txt"))
    count = len(list(a.out.glob("*.txt")))
    print(f"Уже є: {count} файлів, {total / 1e6:.1f} МБ. Ціль: {a.mb} МБ.")
    while total < a.mb * 1e6:
        try:
            pages = fetch_random()
        except (OSError, ValueError) as exc:
            print("помилка запиту:", exc)
            time.sleep(5)
            continue
        for page in pages.values():
            text = page.get("extract")
            pid = page.get("pageid")
            if not isinstance(text, str) or len(text) < MIN_CHARS or pid is None:
                continue
            path = a.out / f"wiki_{pid}.txt"
            if path.exists():
                continue
            path.write_text(text, encoding="utf-8")
            total += path.stat().st_size
            count += 1
        print(f"{count} файлів, {total / 1e6:.1f} МБ")
        time.sleep(PAUSE)
    print(f"Готово: {count} файлів, {total / 1e6:.1f} МБ у {a.out}")


if __name__ == "__main__":
    main()
