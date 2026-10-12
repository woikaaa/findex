"""
snippets.py — a ±width character window around the best match, with the
matched terms highlighted.

We re-tokenize the NFC-normalised text with the *same* regex as the index but
keep character offsets, so the snippet shows the ORIGINAL casing.
Only works for the directory-of-.txt corpus (for .jsonl the doc text is not
re-readable by path) — then the snippet is empty.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from findex.tokenize import TOKEN_RE


def make_snippet(text: str, terms, width: int = 80, mark: tuple[str, str] = ("**", "**")) -> str:
    text = unicodedata.normalize("NFC", text)
    terms = set(terms)
    hits = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)
            if m.group(0).casefold() in terms]
    if not hits:
        return ""

    def score(start: int) -> int:     # how many distinct query terms near `start`
        return len({text[s:e].casefold() for s, e in hits
                    if start - width <= s <= start + width})

    best = max(hits[:300], key=lambda h: (score(h[0]), -h[0]))[0]
    lo, hi = max(0, best - width), min(len(text), best + width)
    out, pos = [], lo
    for s, e in hits:
        if s < lo or e > hi:
            continue
        out.append(text[pos:s])
        out.append(mark[0] + text[s:e] + mark[1])
        pos = e
    out.append(text[pos:hi])
    body = " ".join("".join(out).split())
    return ("… " if lo > 0 else "") + body + (" …" if hi < len(text) else "")


def snippet_for(index, doc_id: int, terms, width: int = 80) -> str:
    path = Path(index.doc_meta[doc_id].path)
    if path.suffix != ".txt" or not path.exists():
        return ""
    return make_snippet(path.read_text(encoding="utf-8", errors="replace"), terms, width)
