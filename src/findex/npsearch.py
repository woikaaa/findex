# Заглиблені типи NumPy для pyright у режимі strict "частково невідомі" (це особливість
# заглушок типів NumPy, а не помилка в коді). Вимикаю лише ці три перевірки в цьому файлі.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""
пошук на масивах NumPy (лаба 8).

У лабі 3 пошук ходив по кожному запису слова в циклі Python і на кожен запис викликав
`scorer.score`. Профіль показав, що це 59% часу пошуку. Тут записи слова лежать в двох
масивах int32 (номери документів і частоти), довжини документів в одному масиві int32,
а бали всіх записів слова рахуються ОДНИМ виразом NumPy.

Результат такий самий, як у `findex.ranking.Searcher`: ті самі документи, ті самі бали
(до останнього біта) і той самий порядок, зокрема при рівних балах (меншим doc_id раніше).
Це перевіряють тести `tests/test_npsearch.py`.
"""

from __future__ import annotations

import functools
import json
import math
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from findex.model import DocMeta, Index
from findex.query import And, Node, Not, Or, Phrase, QueryError, Term, parse_cached
from findex.ranking import SearchResult
from findex.scoring import BM25, Scorer, TfIdf
from findex.store import MAGIC
from findex.utils import timed

IntArray = NDArray[np.int32]
FloatArray = NDArray[np.float64]
PosArray = NDArray[np.intp]  # позиції в масиві (результат argpartition, lexsort)


class NpIndex:
    """Індекс у вигляді масивів NumPy.

    ids  - номери документів усіх слів підряд (int32);
    tfs  - частоти слів у документах, в тому ж порядку (int32);
    spans - слово -> (звідки починаються його записи, скільки їх);
    doc_lengths - довжина кожного документа, індекс масиву це doc_id (int32), у пропущених
                  номерів нуль.
    Записи одного слова йдуть за зростанням doc_id, як і в звичайному Index."""

    def __init__(
        self,
        ids: IntArray,
        tfs: IntArray,
        spans: dict[str, tuple[int, int]],
        doc_lengths: IntArray,
        doc_meta: dict[int, DocMeta],
        avg_doc_length: float,
    ) -> None:
        self.ids = ids
        self.tfs = tfs
        self.spans = spans
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta
        self.avg_doc_length = avg_doc_length
        self.num_docs = len(doc_meta)  # скільки документів насправді (а не найбільший номер)
        # номери всіх документів по порядку: потрібні для NOT
        self.all_ids: IntArray = np.array(sorted(doc_meta), dtype=np.int32)
        self.has_positions = False
        self._log_tf: FloatArray | None = None

    # ---- побудова
    @classmethod
    def from_index(cls, index: Index) -> NpIndex:
        """Перекладає звичайний Index у масиви NumPy (позиції слів не переносяться)."""
        total = sum(index.df(t) for t in index.postings)
        ids = np.empty(total, dtype=np.int32)
        tfs = np.empty(total, dtype=np.int32)
        spans: dict[str, tuple[int, int]] = {}
        pos = 0
        for term, entry in index.postings.items():
            n = len(entry[0]) if isinstance(entry, tuple) else len(entry)
            if isinstance(entry, tuple):
                # array('I') -> NumPy без повільного циклу
                ids[pos : pos + n] = np.frombuffer(entry[0], dtype=np.uint32)
                tfs[pos : pos + n] = np.frombuffer(entry[1], dtype=np.uint32)
            else:
                ids[pos : pos + n] = np.fromiter((p.doc_id for p in entry), np.int32, n)
                tfs[pos : pos + n] = np.fromiter((p.tf for p in entry), np.int32, n)
            spans[term] = (pos, n)
            pos += n
        lengths = _lengths_array(index.doc_lengths)
        return cls(ids, tfs, spans, lengths, dict(index.doc_meta), index.avg_doc_length)

    @classmethod
    def from_file(cls, path: Path, mmap: bool = True) -> NpIndex:
        """Читає бінарний файл findex (формат FIDX1) прямо в масиви NumPy.

        mmap=True: великі масиви НЕ читаються в пам'ять, а відображаються з файлу
        (np.memmap): ОС підвантажує з диска лише ті сторінки, які справді потрібні."""
        path = Path(path)
        with open(path, "rb") as f:
            if f.read(len(MAGIC)) != MAGIC:
                raise ValueError("not a findex binary file")
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = cast("dict[str, Any]", json.loads(f.read(hlen).decode("utf-8")))
            offset = f.tell()
        n = int(header["n_postings"])
        # порядок байтів як у файлі (на звичайних комп'ютерах це little)
        dt = np.dtype("<i4" if header["byteorder"] == "little" else ">i4")
        ids: IntArray
        tfs: IntArray
        if n == 0:
            ids = np.empty(0, dtype=np.int32)
            tfs = np.empty(0, dtype=np.int32)
        elif mmap:
            # значення менші за 2^31, тому uint32 з файлу можна читати як int32
            ids = cast("IntArray", np.memmap(path, dtype=dt, mode="r", offset=offset, shape=(n,)))
            tfs = cast(
                "IntArray", np.memmap(path, dtype=dt, mode="r", offset=offset + 4 * n, shape=(n,))
            )
        else:
            with open(path, "rb") as f:
                f.seek(offset)
                ids = np.fromfile(f, dtype=dt, count=n).astype(np.int32)
                tfs = np.fromfile(f, dtype=dt, count=n).astype(np.int32)
        spans = {t: (int(o), int(c)) for t, (o, c) in header["terms"].items()}
        doc_lengths = {int(k): int(v) for k, v in header["doc_lengths"].items()}
        doc_meta = {int(k): DocMeta(*v) for k, v in header["doc_meta"].items()}
        avg = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0.0
        return cls(ids, tfs, spans, _lengths_array(doc_lengths), doc_meta, avg)

    # ---- читання
    def __contains__(self, term: object) -> bool:
        return term in self.spans

    def df(self, term: str) -> int:
        """У скількох документах є слово."""
        span = self.spans.get(term)
        return 0 if span is None else span[1]

    def doc_ids(self, term: str) -> IntArray:
        """Номери документів слова (вигляд на великий масив, без копіювання)."""
        span = self.spans.get(term)
        if span is None:
            return np.empty(0, dtype=np.int32)
        start, n = span
        return self.ids[start : start + n]

    def term_tfs(self, term: str) -> IntArray:
        """Частоти слова, в тому ж порядку, що й doc_ids."""
        span = self.spans.get(term)
        if span is None:
            return np.empty(0, dtype=np.int32)
        start, n = span
        return self.tfs[start : start + n]

    @property
    def log_tf(self) -> FloatArray:
        """Таблиця math.log(tf) для всіх tf (потрібна TF-IDF; math.log, а не np.log, щоб
        бали збігалися з лабою 3 до останнього біта)."""
        if self._log_tf is None:
            top = int(self.tfs.max()) if len(self.tfs) else 0
            table = np.zeros(top + 1, dtype=np.float64)
            for i in range(1, top + 1):
                table[i] = math.log(i)
            self._log_tf = table
        return self._log_tf

    @property
    def nbytes(self) -> int:
        """Скільки байтів займають великі масиви (для таблиці пам'яті)."""
        return int(self.ids.nbytes + self.tfs.nbytes + self.doc_lengths.nbytes)


def _lengths_array(doc_lengths: dict[int, int]) -> IntArray:
    """Довжини документів одним масивом: індекс це doc_id."""
    size = max(doc_lengths) + 1 if doc_lengths else 0
    arr = np.zeros(size, dtype=np.int32)
    for doc_id, n in doc_lengths.items():
        arr[doc_id] = n
    return arr


# ------------------------------------------------------------------ оцінка балів
def bm25_scores(ix: NpIndex, term: str, scorer: BM25) -> FloatArray:
    """Бали BM25 усіх записів слова одним виразом. Формула та сама, що в scoring.BM25."""
    ids = ix.doc_ids(term)
    df = ix.df(term)
    idf = math.log(1 + (ix.num_docs - df + 0.5) / (df + 0.5))  # одне число на слово
    tf = ix.term_tfs(term).astype(np.float64)
    dl = ix.doc_lengths[ids]  # довжини документів цих записів
    norm = 1 - scorer.b + scorer.b * dl / ix.avg_doc_length  # поправка на довжину
    return idf * tf * (scorer.k1 + 1) / (tf + scorer.k1 * norm)


def tfidf_scores(ix: NpIndex, term: str) -> FloatArray:
    """Бали TF-IDF усіх записів слова. Формула та сама, що в scoring.TfIdf."""
    df = ix.df(term)
    idf = math.log(ix.num_docs / df)
    return (1 + ix.log_tf[ix.term_tfs(term)]) * idf


# ------------------------------------------------------------------ відбір документів
def _evaluate(node: Node, ix: NpIndex) -> IntArray:
    """Документи, що підходять під запит: відсортований масив без повторів."""
    if isinstance(node, Term):
        return ix.doc_ids(node.text)
    if isinstance(node, Phrase):
        # позицій у масивах немає, тому фраз тут немає (як і в бінарному форматі)
        raise QueryError("phrase queries need an index built with --positions")
    if isinstance(node, And):
        return np.intersect1d(_evaluate(node.left, ix), _evaluate(node.right, ix), True)
    if isinstance(node, Or):
        return np.union1d(_evaluate(node.left, ix), _evaluate(node.right, ix)).astype(np.int32)
    if isinstance(node, Not):
        return np.setdiff1d(ix.all_ids, _evaluate(node.operand, ix), True)
    raise TypeError(f"unknown query node: {node!r}")


def top_k(matched: IntArray, scores: FloatArray, k: int) -> PosArray:
    """Позиції k найкращих. Порядок: бал за спаданням, при рівних балах менший doc_id першим.

    np.argpartition знаходить k найбільших за O(n) без повного сортування. Але на межі
    можуть бути рівні бали, і argpartition взяв би з них будь-який. Тому з нього беру лише
    ПОРІГ (найменший бал серед k найбільших), збираю всіх, хто не нижче за поріг, і
    сортую тільки їх."""
    n = len(scores)
    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.intp)
    if k < n:
        best = np.argpartition(scores, n - k)[n - k :]
        threshold = scores[best].min()
        cand = np.flatnonzero(scores >= threshold)
    else:
        cand = np.arange(n)
    # lexsort сортує за останнім ключем першим: спочатку -бал, потім doc_id
    order = np.lexsort((matched[cand], -scores[cand]))
    return cand[order][:k]


class NpSearcher:
    """Те саме, що `Searcher` з лаби 3, але бали й топ-k рахує NumPy.

    Має ті самі методи search, count, terms, тому підставляється замість Searcher."""

    def __init__(self, index: NpIndex, scorer: Scorer | None = None) -> None:
        if scorer is None:
            scorer = BM25()
        if not isinstance(scorer, BM25 | TfIdf):
            raise TypeError("NpSearcher підтримує лише BM25 і TfIdf")
        self.index = index
        self.scorer = scorer
        # як і в Searcher: свій кеш на кожен об'єкт
        self._match = functools.lru_cache(maxsize=256)(self._match_uncached)

    def _match_uncached(self, query: str) -> IntArray:
        matched = _evaluate(parse_cached(query), self.index)
        out = np.asarray(matched, dtype=np.int32)
        out.flags.writeable = False  # значення лежить у кеші, його не можна псувати
        return out

    def _term_scores(self, term: str) -> FloatArray:
        if isinstance(self.scorer, BM25):
            return bm25_scores(self.index, term, self.scorer)
        return tfidf_scores(self.index, term)

    @timed
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        matched = self._match(query)
        node = parse_cached(query)
        # бали по всіх номерах документів; нулі там, де слова немає
        total = np.zeros(len(self.index.doc_lengths), dtype=np.float64)
        for term in node.positive_terms():
            if term not in self.index:
                continue
            # у слова doc_id не повторюються, тому += по списку номерів безпечне
            total[self.index.doc_ids(term)] += self._term_scores(term)
        scores = total[matched]
        pick = top_k(matched, scores, k)
        meta = self.index.doc_meta
        return [
            SearchResult(int(matched[i]), float(scores[i]), meta[int(matched[i])].title)
            for i in pick
        ]

    def count(self, query: str) -> int:
        """Скільки документів підходить під запит усього."""
        return len(self._match(query))

    def terms(self, query: str) -> frozenset[str]:
        """Слова запиту, які мають бути підсвічені в уривках."""
        return parse_cached(query).positive_terms()


__all__ = ["NpIndex", "NpSearcher", "bm25_scores", "tfidf_scores", "top_k"]
