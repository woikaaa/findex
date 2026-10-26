"""Перевірки ранжування, кешу, декоратора і уривків."""

from __future__ import annotations

import logging

import pytest

from findex.index import build_index
from findex.model import Index, Posting
from findex.ranking import Searcher, SearchResult
from findex.scoring import BM25, SCORER_NAMES, TfIdf, make_scorer
from findex.snippets import make_snippet


@pytest.mark.parametrize("name", SCORER_NAMES)
def test_scorers_are_interchangeable(index: Index, name: str) -> None:
    scorer = make_scorer(name)  # type: ignore[arg-type]
    res = Searcher(index, scorer).search("queen OR duchess", 3)
    assert res == sorted(res, reverse=True) and len(res) == 3
    assert scorer("alice", Posting(0, 1), index) == scorer.score("alice", Posting(0, 1), index)


def test_make_scorer_unknown_name() -> None:
    with pytest.raises(KeyError):
        make_scorer("nope")  # type: ignore[arg-type]


def test_tfidf_and_bm25_are_different_classes() -> None:
    assert isinstance(make_scorer("bm25"), BM25)
    assert isinstance(make_scorer("tfidf"), TfIdf)


def test_bm25_rare_word_beats_common_word(index: Index) -> None:
    mid = sorted(index.doc_lengths, key=lambda d: index.doc_lengths[d])[6]
    terms = sorted(index, key=index.df)
    rare, common = terms[0], terms[-1]
    b = BM25()
    assert b.score(rare, Posting(mid, 1), index) > b.score(common, Posting(mid, 1), index)


def test_bm25_term_frequency_saturates(index: Index) -> None:
    mid = sorted(index.doc_lengths, key=lambda d: index.doc_lengths[d])[6]
    rare = sorted(index, key=index.df)[0]
    b = BM25()
    s = {tf: b.score(rare, Posting(mid, tf), index) for tf in (1, 2, 20, 21)}
    assert (s[21] - s[20]) < 0.05 * (s[2] - s[1])  # приріст між 20 і 21 майже нульовий


def test_bm25_prefers_shorter_document(index: Index) -> None:
    ids = sorted(index.doc_lengths, key=lambda d: index.doc_lengths[d])
    rare = sorted(index, key=index.df)[0]
    b = BM25()
    assert b.score(rare, Posting(ids[0], 1), index) > b.score(rare, Posting(ids[-1], 1), index)


def test_search_result_orders_by_score_only() -> None:
    a, b = SearchResult(1, 0.5, "x"), SearchResult(2, 0.9, "y")
    assert a < b and sorted([b, a]) == [a, b]


def test_search_returns_at_most_k(index: Index) -> None:
    assert len(Searcher(index).search("alice", 2)) == 2


def test_empty_result_for_unknown_word(index: Index) -> None:
    assert Searcher(index).search("zzzzqqq") == []


def test_lru_cache_hit_on_repeated_query(index: Index) -> None:
    s = Searcher(index)
    s.search("alice rabbit")
    s.search("alice rabbit")
    info = s._match.cache_info()
    assert info.hits == 1 and info.misses == 1


def test_timed_keeps_metadata_and_logs(index: Index, caplog: pytest.LogCaptureFixture) -> None:
    # @timed не губить ім'я функції (functools.wraps) і пише в лог
    assert build_index.__name__ == "build_index"
    with caplog.at_level(logging.INFO, logger="findex.timing"):
        Searcher(index).search("alice", 1)
    assert any("Searcher.search took" in r.getMessage() for r in caplog.records)


def test_snippet_highlights_and_keeps_case() -> None:
    text = "x " * 100 + "The Queen arrived. " + "y " * 100
    snip = make_snippet(text, {"queen"}, width=20)
    assert "**Queen**" in snip and snip.startswith("… ") and snip.endswith(" …")


def test_snippet_empty_when_no_match() -> None:
    assert make_snippet("nothing here", {"queen"}) == ""
