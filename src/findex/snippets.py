from __future__ import annotations

import unicodedata
from pathlib import Path

from findex.tokenize import TOKEN_RE


def make_snippet(text: str, terms, width: int = 80, mark: tuple[str, str] = ("**", "**")) -> str:
    """Робить уривок тексту з виділеними словами запиту."""
    # нормалізую, як при створенні індексу
    text = unicodedata.normalize("NFC", text)
    # множина швидша для перевірки
    terms = set(terms)
    # усі місця тексту (початок, кінець), де стоїть слово запиту
    hits = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)
            if m.group(0).casefold() in terms]
    # якщо слів у тексті немає, уривка теж немає
    if not hits:
        return ""

    def score(start: int) -> int:     # скільки різних слів запиту поруч із `start`
        return len({text[s:e].casefold() for s, e in hits
                    if start - width <= s <= start + width})

    # обираємо місце, де поруч найбільше різних слів запиту
    best = max(hits[:300], key=lambda h: (score(h[0]), -h[0]))[0]
    # межі уривка, щоб не вийти за текст
    lo, hi = max(0, best - width), min(len(text), best + width)  # межі вікна
    # збираю уривок по шматках
    out, pos = [], lo
    for s, e in hits:
        if s < lo or e > hi:
            continue  # це слово поза вікном
        out.append(text[pos:s])                      # текст до слова
        out.append(mark[0] + text[s:e] + mark[1])    # саме слово у зірочках
        pos = e
    out.append(text[pos:hi])
    body = " ".join("".join(out).split())  # прибираємо зайві пробіли й переноси
    # "…" показує, що уривок обрізаний з початку або з кінця
    return ("… " if lo > 0 else "") + body + (" …" if hi < len(text) else "")


def snippet_for(index, doc_id: int, terms, width: int = 80) -> str:
    """Читає файл документа й робить з нього уривок."""
    # де лежить файл цього документа
    path = Path(index.doc_meta[doc_id].path)
    # уривок роблю лише з .txt файлів
    if path.suffix != ".txt" or not path.exists():
        return ""
    return make_snippet(path.read_text(encoding="utf-8", errors="replace"), terms, width)
