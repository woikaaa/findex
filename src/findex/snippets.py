from __future__ import annotations

import json
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from findex.model import Index
from findex.tokenize import TOKEN_RE

# наступний пробільний символ (по ньому безпечно різати текст на шматки для нормалізації)
_SPACE_RE = re.compile(r"\s")
# приблизний розмір шматка для нормалізації, символів
_CHUNK = 4096


def _to_nfc(text: str) -> str:
    """Те саме, що unicodedata.normalize("NFC", text), але швидше для довгих текстів.

    Проблема, яку показав бенчмарк: коли в тексті є хоч один символ, якому потрібна
    нормалізація (наприклад, знак наголосу U+0301), normalize обробляє ВЕСЬ текст повільним
    шляхом (20 мс на 600 КБ). Тому різаю текст на шматки по ~4000 символів у місцях ПЕРЕД
    пробілом і нормалізую кожен окремо: швидкий шлях спрацьовує для всіх шматків, де
    нічого міняти не треба. Різати перед пробілом безпечно: пробіл не склеюється з
    попереднім символом і не переставляється з ним, тож результат такий самий, як для
    всього тексту (це перевіряє тест на випадкових рядках зі знаками наголосу)."""
    if text.isascii():
        return text  # у чистому ASCII нормалізувати нічого
    parts: list[str] = []
    pos, n = 0, len(text)
    while pos < n:
        end = pos + _CHUNK
        if end >= n:
            end = n
        else:
            m = _SPACE_RE.search(text, end)
            end = m.start() if m else n
        parts.append(unicodedata.normalize("NFC", text[pos:end]))
        pos = end
    return "".join(parts)


# скільки перших влучань розглядаю, коли шукаю найкраще місце для уривка
MAX_HITS = 300


def make_snippet(
    text: str,
    terms: Iterable[str],
    width: int = 80,
    mark: tuple[str, str] = ("**", "**"),
) -> str:
    """Робить уривок тексту з виділеними словами запиту.

    Профіль (лаба 8) показав, що це 95% часу сторінки результатів. Що я змінила, а результат
    лишився той самий (це перевіряє тест порівнянням зі старою версією):
      1. Нормалізація NFC шматками (_to_nfc): normalize на цілому тексті було 24% часу.
      2. Текст читаю лише до того місця, де вже знайдено MAX_HITS влучань і минуло ще
         `width` символів: далі ні вибір місця, ні вікно вже не можуть змінитися.
      3. Слово приводжу до casefold один раз (раніше двічі), а скільки різних слів поруч
         рахую через бінарний пошук (bisect) по відсортованих початках, а не проходом по всіх
         влучаннях (раніше це був вкладений цикл).
    """
    # нормалізую, як при створенні індексу (шматками, див. _to_nfc)
    text = _to_nfc(text)
    # множина швидша для перевірки
    wanted = set(terms)
    # початки, кінці і слова влучань (три списки, бо початки треба шукати через bisect)
    starts: list[int] = []
    ends: list[int] = []
    words: list[str] = []
    stop_after: int | None = None  # після цієї позиції нові влучання вже ні на що не впливають
    for m in TOKEN_RE.finditer(text):
        start = m.start()
        if stop_after is not None and start > stop_after:
            break
        word = m.group(0).casefold()
        if word in wanted:
            starts.append(start)
            ends.append(m.end())
            words.append(word)
            if len(starts) == MAX_HITS:
                stop_after = start + width
    # якщо слів у тексті немає, уривка теж немає
    if not starts:
        return ""

    def score(i: int) -> int:  # скільки різних слів запиту поруч із влучанням номер i
        lo_i = bisect_left(starts, starts[i] - width)
        hi_i = bisect_right(starts, starts[i] + width)
        return len(set(words[lo_i:hi_i]))

    # обираю місце, де поруч найбільше різних слів запиту (при рівності лівіше)
    best = starts[max(range(min(MAX_HITS, len(starts))), key=lambda i: (score(i), -starts[i]))]
    # межі уривка, щоб не вийти за текст
    lo, hi = max(0, best - width), min(len(text), best + width)  # межі вікна
    # збираю уривок по шматках
    out: list[str] = []
    pos = lo
    for i in range(bisect_left(starts, lo), len(starts)):
        s, e = starts[i], ends[i]
        if s >= hi:
            break  # далі всі слова поза вікном
        if e > hi:
            continue  # слово вилазить за вікно
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
