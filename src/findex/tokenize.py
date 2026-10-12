"""
tokenize.py — turns raw text into a lazy stream of normalized tokens.

Policy (documented, per the lab's requirement):

  1. Unicode-normalize with NFC first, so that visually identical
     characters encoded differently (e.g. "é" as one code point vs.
     "e" + a combining accent) become the *same* token. Without this,
     "café" and "café" would be counted as two different words.

  2. Casefold, not lower(). casefold() is the aggressive, locale
     independent normalization meant for caseless comparison
     ("Straße".casefold() == "strasse"); plain lower() misses this.

  3. Tokens are matched with re.finditer over the pattern below, not
     re.findall — finditer yields match objects one at a time instead
     of building a list of every match up front, which is the whole
     point of a "streaming" tokenizer.

  4. Apostrophes: kept *inside* a word (don't, п'ять) but not at the
     edges. We treat ' and the Unicode right single quote U+2019 as
     an apostrophe, because real text (especially Ukrainian: п'ять,
     м'ята) uses both interchangeably.

  5. Hyphens: kept inside a word (well-known) but not as leading/
     trailing punctuation, same reasoning as apostrophes.

  6. Digits: \\w already matches digits, so "covid19" or "2024" are
     tokens on their own; we do not special-case numbers further in
     this lab.

  7. \\w in Python 3 is Unicode-aware by default, so Cyrillic,
     accented Latin, and other scripts are matched correctly with no
     extra flags.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

# A token is a run of "word" characters, optionally containing a single
# internal apostrophe or hyphen (but never starting/ending with one).
# \w is Unicode-aware, so this also matches Cyrillic, accented Latin, etc.
_TOKEN_RE = re.compile(
    r"\w+(?:['\u2019-]\w+)*",
    re.UNICODE,
)


TOKEN_RE = _TOKEN_RE  # public alias, reused by snippets.py


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens from ``text``, one at a time.

    Steps applied, in order: NFC normalization -> casefold -> regex
    match via re.finditer (lazy, unlike re.findall).
    """
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\u0301", "")  # прибираємо знак наголосу (Архітекту́ра)
    folded = normalized.casefold()
    for match in _TOKEN_RE.finditer(folded):
        yield match.group(0)
