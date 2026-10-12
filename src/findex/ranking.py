"""
ranking.py — ranked search: parse -> match -> score -> top-k -> snippets.

    python -m findex.ranking index.bin "queen OR duchess" --scorer bm25 -k 5
    python -m findex.ranking index.bin "cheshire cat" -v --repeat 2     # see cache hit
"""

from __future__ import annotations

import argparse
import functools
import heapq
import logging
from dataclasses import dataclass, field
from pathlib import Path

from findex import store
from findex.model import Index
from findex.query import parse_cached
from findex.scoring import BM25, Scorer, TfIdf
from findex.snippets import snippet_for
from findex.utils import timed

log = logging.getLogger("findex.search")


@dataclass(order=True)
class SearchResult:
    """order=True generates <, <=, >, >= from the fields with compare=True.
    Only `score` compares, so sorted(results) orders by score."""
    doc_id: int = field(compare=False)
    score: float
    title: str = field(compare=False)

    def __str__(self) -> str:
        return f"{self.score:7.3f}  #{self.doc_id:<4} {self.title}"


class Searcher:
    def __init__(self, index: Index, scorer: Scorer | None = None) -> None:
        self.index = index
        self.scorer = scorer or BM25()
        # lru_cache needs hashable arguments; the Index is not hashable (it is a
        # Mapping), so we build one cache PER Searcher, wrapping a bound method.
        self._match = functools.lru_cache(maxsize=256)(self._match_uncached)

    def _match_uncached(self, query: str) -> tuple[int, ...]:
        return tuple(sorted(parse_cached(query).evaluate(self.index)))

    @timed
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        hits_before = self._match.cache_info().hits
        matched = self._match(query)
        if self._match.cache_info().hits > hits_before:
            log.info("cache hit for %r", query)

        node = parse_cached(query)
        wanted = set(matched)
        scores = dict.fromkeys(matched, 0.0)
        for term in node.positive_terms():
            if term not in self.index:
                continue
            for posting in self.index[term]:
                if posting.doc_id in wanted:
                    scores[posting.doc_id] += self.scorer.score(term, posting, self.index)

        # top-k in O(n log k); tie-break: smaller doc_id first
        top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        return [SearchResult(d, s, self.index.doc_meta[d].title) for d, s in top]

    def terms(self, query: str) -> frozenset[str]:
        return parse_cached(query).positive_terms()


def main() -> None:
    p = argparse.ArgumentParser(description="ranked search")
    p.add_argument("index", type=Path)
    p.add_argument("query")
    p.add_argument("--scorer", choices=("bm25", "tfidf"), default="bm25")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--repeat", type=int, default=1, help="run the query N times (cache demo)")
    p.add_argument("--no-snippets", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="show @timed / cache log")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="[%(name)s] %(message)s")

    with store.open_index(a.index) as index:
        searcher = Searcher(index, BM25() if a.scorer == "bm25" else TfIdf())
        for _ in range(a.repeat):
            results = searcher.search(a.query, a.k)
        print(f"{len(results)} result(s)  [{a.scorer}]  {index!r}")
        terms = searcher.terms(a.query)
        for r in sorted(results, reverse=True):
            print(" ", r)
            if not a.no_snippets:
                snip = snippet_for(index, r.doc_id, terms)
                if snip:
                    print("      ", snip)


if __name__ == "__main__":
    main()
