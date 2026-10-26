"""Перевірки пошуку: merge проти set, парсер запитів, фрази."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from findex.model import Index
from findex.query import And, Not, Or, ParseError, Phrase, QueryError, Term, parse
from findex.search import ENGINES, evaluate


@pytest.mark.parametrize("op", [0, 1, 2], ids=["and", "or", "andnot"])
def test_merge_equals_set_on_random_lists(op: int) -> None:
    rng = random.Random(1)  # фіксую випадковість, щоб тест повторювався
    for _ in range(200):
        a = sorted(rng.sample(range(50), rng.randint(0, 20)))
        b = sorted(rng.sample(range(50), rng.randint(0, 20)))
        assert ENGINES["merge"][op](a, b) == ENGINES["set"][op](a, b)


@pytest.mark.parametrize(
    "query", ["alice rabbit", "queen OR duchess", "alice NOT rabbit", "NOT alice"]
)
def test_engines_agree_on_queries(index: Index, query: str) -> None:
    assert evaluate(index, query, "merge") == evaluate(index, query, "set")


def test_parser_precedence_and_before_or() -> None:
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))


def test_parser_parens_not_and_phrase() -> None:
    tree = parse("python AND (async OR await) NOT java")
    assert tree == And(And(Term("python"), Or(Term("async"), Term("await"))), Not(Term("java")))
    assert parse('"event loop"') == Phrase(("event", "loop"))


def test_operator_overloading_builds_same_tree() -> None:
    built = Term("a") & (Term("b") | Term("c")) & ~Term("d")
    assert built == parse("a AND (b OR c) NOT d")


@pytest.mark.parametrize("bad", ["", "(a", "a OR", "NOT", "a )"])
def test_parse_errors(bad: str) -> None:
    with pytest.raises(ParseError):
        parse(bad)


def test_phrase_respects_word_order(index: Index) -> None:
    assert parse('"cheshire cat"').evaluate(index) == {5, 7}
    assert parse('"cat cheshire"').evaluate(index) == set()


def test_not_removes_documents(index: Index) -> None:
    assert parse("queen NOT duchess").evaluate(index) == {7}


def test_phrase_needs_positions(corpus_dir: Path) -> None:
    from findex.index import build_index

    plain = build_index(corpus_dir, "slots", positions=False)
    with pytest.raises(QueryError, match="positions"):
        parse('"cheshire cat"').evaluate(plain)
