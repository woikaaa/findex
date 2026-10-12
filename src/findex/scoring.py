"""
scoring.py — a Scorer *protocol* and two interchangeable implementations.

A Scorer is anything with  score(term, posting, index) -> float.
TfIdf and BM25 do not inherit from Scorer: they just have the right method
(structural typing / duck typing). They are also callable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from findex.model import Index, Posting


class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: Index) -> float: ...


class _Callable:
    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


@dataclass(frozen=True)
class TfIdf(_Callable):
    """tf-idf = (1 + ln tf) * ln(N / df).  A term found in every document scores 0."""

    def score(self, term, posting, index):
        df = index.df(term)
        if df == 0:
            return 0.0
        return (1 + math.log(posting.tf)) * math.log(index.num_docs / df)


@dataclass(frozen=True)
class BM25(_Callable):
    """Okapi BM25.
        idf   = ln(1 + (N - df + 0.5) / (df + 0.5))           (never negative)
        tfpart = tf (k1 + 1) / (tf + k1 (1 - b + b * dl / avgdl))
    k1: how fast repeated occurrences saturate.  b: strength of length normalisation
    (b = 0 -> document length is ignored)."""
    k1: float = 1.5
    b: float = 0.75

    def score(self, term, posting, index):
        df = index.df(term)
        if df == 0:
            return 0.0
        n = index.num_docs
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        dl = index.doc_length(posting.doc_id)
        norm = 1 - self.b + self.b * dl / index.avg_doc_length
        return idf * posting.tf * (self.k1 + 1) / (posting.tf + self.k1 * norm)
