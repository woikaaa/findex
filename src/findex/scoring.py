from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, get_args

from findex.model import Index, PostingLike

# назви формул; Literal не дає помилитися
ScorerName = Literal["bm25", "tfidf"]
SCORER_NAMES: tuple[ScorerName, ...] = get_args(ScorerName)


class Scorer(Protocol):
    """Опис того, що має вміти будь-який "оцінювач": рахувати бал слова в документі."""

    def score(self, term: str, posting: PostingLike, index: Index) -> float: ...


class _Callable:
    """Дозволяє писати scorer(...) замість scorer.score(...)."""

    def score(self, term: str, posting: PostingLike, index: Index) -> float:
        raise NotImplementedError

    def __call__(self, term: str, posting: PostingLike, index: Index) -> float:
        # просто викликаю score
        return self.score(term, posting, index)


@dataclass(frozen=True)
class TfIdf(_Callable):
    """tf-idf = (1 + ln tf) * ln(N / df).  Слово, що є в кожному документі, отримує 0."""

    def score(self, term: str, posting: PostingLike, index: Index) -> float:
        # у скількох документах є слово
        df = index.df(term)
        if df == 0:
            # якщо слова немає, балів теж немає
            return 0.0  # слова немає в корпусі
        # сама формула TF-IDF
        return (1 + math.log(posting.tf)) * math.log(index.num_docs / df)


@dataclass(frozen=True)
class BM25(_Callable):
    """Okapi BM25.
        idf   = ln(1 + (N - df + 0.5) / (df + 0.5))           (ніколи не від'ємний)
        tfpart = tf (k1 + 1) / (tf + k1 (1 - b + b * dl / avgdl))
    k1: як швидко повторення слова перестають додавати бали.
    b: наскільки сильно враховується довжина документа (b = 0 -> довжина ігнорується)."""

    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: PostingLike, index: Index) -> float:
        df = index.df(term)
        if df == 0:
            return 0.0
        # скільки всього документів
        n = index.num_docs
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))  # рідкісне слово важить більше
        dl = index.doc_length(posting.doc_id)  # довжина цього документа
        norm = 1 - self.b + self.b * dl / index.avg_doc_length  # поправка на довжину
        # збираю формулу BM25 докупи
        return idf * posting.tf * (self.k1 + 1) / (posting.tf + self.k1 * norm)


def make_scorer(name: ScorerName) -> Scorer:
    """Повертає формулу за назвою. Типи перевіряє pyright: обидві підходять під Scorer."""
    scorers: dict[ScorerName, Scorer] = {"bm25": BM25(), "tfidf": TfIdf()}
    return scorers[name]
