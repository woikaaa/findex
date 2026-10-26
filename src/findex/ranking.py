"""
пошук з ранжуванням: розбір - відбір - оцінка -топ-k - уривки.

    python -m findex.ranking index.bin "queen OR duchess" --scorer bm25 -k 5
    python -m findex.ranking index.bin "cheshire cat" -v --repeat 2     # побачити cache hit
"""

from __future__ import annotations

import functools
import heapq
import logging
from dataclasses import dataclass, field

from findex.model import Index
from findex.query import parse_cached
from findex.scoring import BM25, Scorer
from findex.utils import timed

log = logging.getLogger("findex.search")


@dataclass(order=True)
class SearchResult:
    doc_id: int = field(compare=False)
    score: float
    title: str = field(compare=False)

    def __str__(self) -> str:
        return f"{self.score:7.3f}  #{self.doc_id:<4} {self.title}"


class Searcher:
    """Об'єднує індекс і оцінювач, вміє шукати й повертати топ результатів."""

    def __init__(self, index: Index, scorer: Scorer | None = None) -> None:
        # запам'ятовую індекс
        self.index = index
        self.scorer = scorer or BM25()  # за замовчуванням BM25
        # lru_cache потребує аргументів, які можна хешувати. Index хешувати
        # не можна (він Mapping), тому робимо окремий кеш НА КОЖЕН Searcher,
        # обгорнувши метод конкретного об'єкта.
        self._match = functools.lru_cache(maxsize=256)(self._match_uncached)

    def _match_uncached(self, query: str) -> tuple[int, ...]:
        """Знаходить документи, що підходять під запит (без кешу)."""
        # розбираю запит і беру номери документів
        return tuple(sorted(parse_cached(query).evaluate(self.index)))

    @timed
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        # перевіряємо, чи кеш спрацював: порівнюємо лічильник влучань до і після
        hits_before = self._match.cache_info().hits
        matched = self._match(query)
        if self._match.cache_info().hits > hits_before:
            log.info("cache hit for %r", query)

        # беру дерево запиту (воно теж з кешу)
        node = parse_cached(query)
        # множина, щоб швидко перевіряти, чи документ підходить
        wanted = set(matched)
        scores = dict.fromkeys(matched, 0.0)  # у кожного знайденого документа бал 0
        # додаємо бали за кожне слово запиту (крім тих, що під NOT)
        for term in node.positive_terms():
            if term not in self.index:
                continue
            # іду по документах, де є це слово
            for posting in self.index[term]:
                if posting.doc_id in wanted:
                    scores[posting.doc_id] += self.scorer.score(term, posting, self.index)

        # топ-k за O(n log k); якщо бали рівні, першим іде менший doc_id
        top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        # складаю список результатів
        return [SearchResult(d, s, self.index.doc_meta[d].title) for d, s in top]

    def terms(self, query: str) -> frozenset[str]:
        """Слова запиту, які мають бути підсвічені в уривках."""
        # слова запиту без тих, що під NOT
        return parse_cached(query).positive_terms()
