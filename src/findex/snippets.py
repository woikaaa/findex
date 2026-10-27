from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from findex.model import Index
from findex.tokenize import TOKEN_RE


def make_snippet(
    text: str,
    terms: Iterable[str],
    width: int = 80,
    mark: tuple[str, str] = ("**", "**"),
) -> str:
    """Робить уривок тексту з виділеними словами запиту."""
    # нормалізую, як при створенні індексу
    text = unicodedata.normalize("NFC", text)
    # множина швидша для перевірки
    wanted = set(terms)
    # усі місця тексту (початок, кінець), де стоїть слово запиту
    hits = [
        (m.start(), m.end()) for m in TOKEN_RE.finditer(text) if m.group(0).casefold() in wanted
    ]
    # якщо слів у тексті немає, уривка теж немає
    if not hits:
        return ""

    def score(start: int) -> int:  # скільки різних слів запиту поруч із `start`
        return len({text[s:e].casefold() for s, e in hits if start - width <= s <= start + width})

    # обираємо місце, де поруч найбільше різних слів запиту
    best = max(hits[:300], key=lambda h: (score(h[0]), -h[0]))[0]
    # межі уривка, щоб не вийти за текст
    lo, hi = max(0, best - width), min(len(text), best + width)  # межі вікна
    # збираю уривок по шматках
    out: list[str] = []
    pos = lo
    for s, e in hits:
        if s < lo or e > hi:
            continue  # це слово поза вікном
        out.append(text[pos:s])  # текст до слова
        out.append(mark[0] + text[s:e] + mark[1])  # саме слово у зірочках
        pos = e
    out.append(text[pos:hi])
    body = " ".join("".join(out).split())  # прибираємо зайві пробіли й переноси
    # "…" показує, що уривок обрізаний з початку або з кінця
    return ("… " if lo > 0 else "") + body + (" …" if hi < len(text) else "")


def _text_from_jsonl(source: Path, doc_key: str) -> str:
    """Шукає в .jsonl (його пише findex crawl) запис із таким id (адресою) і дає його текст."""
    # читаю файл по рядку, у пам'яті все одразу не тримаю
    with source.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                loaded: object = json.loads(line)
            except json.JSONDecodeError:
                continue  # зіпсований рядок пропускаю
            if not isinstance(loaded, dict):
                continue
            record = cast("dict[str, object]", loaded)
            # для сторінок із краулера id — це їхня адреса, вона ж лежить у індексі як path
            if record.get("id") == doc_key:
                text = record.get("text", "")
                return text if isinstance(text, str) else ""
    return ""


def snippet_for(
    index: Index,
    doc_id: int,
    terms: Iterable[str],
    width: int = 80,
    source: Path | None = None,
) -> str:
    """Читає текст документа й робить з нього уривок.

    Для .txt беру сам файл. Для сторінок з краулера (у індексі там адреса, а не файл)
    текст шукаю в JSONL, який передали як source.
    """
    # де лежить файл цього документа
    meta = index.doc_meta[doc_id]
    path = Path(meta.path)
    if path.suffix == ".txt" and path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
    elif source is not None and source.suffix == ".jsonl" and source.exists():
        text = _text_from_jsonl(source, meta.path)
    else:
        return ""
    return make_snippet(text, terms, width) if text else ""
