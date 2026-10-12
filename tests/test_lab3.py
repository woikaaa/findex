"""Перевірки для лаби 3. python tests/test_lab3.py"""

import logging
import sys
import tempfile
from pathlib import Path

# додаємо папку src, щоб Python знайшов пакет findex
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from findex import store
from findex.index import build_index
from findex.model import Posting
from findex.query import And, Not, Or, ParseError, Phrase, Term, parse
from findex.ranking import Searcher, SearchResult
from findex.scoring import BM25, TfIdf
from findex.snippets import make_snippet

# маленький набір текстів для тестів; індекс будуємо один раз для всіх тестів
DATA = Path(__file__).resolve().parent / "fixtures"
IDX = build_index(DATA, "slots", positions=True)


#  об'єкт Index
def test_index_dunders():
    # перевіряємо спеціальні методи: len, in, [], for, repr
    assert len(IDX) == len(IDX.postings) > 0
    assert "alice" in IDX and "zzzz" not in IDX
    assert all(isinstance(p, Posting) for p in IDX["alice"])
    assert sorted(IDX) == sorted(IDX.postings)
    assert "Index(terms=" in repr(IDX)
    try:
        IDX["zzzz"]
        raise AssertionError("expected KeyError")  # слова немає, має бути KeyError
    except KeyError:
        pass


def test_index_properties_and_cached_property():
    assert IDX.num_docs == 13
    first = IDX.avg_doc_length
    assert first == sum(IDX.doc_lengths.values()) / 13
    assert "avg_doc_length" in IDX.__dict__      # значення закешоване в об'єкті
    assert IDX.df("alice") == len(IDX["alice"])


def test_array_storage_behaves_the_same():
    # array-сховище поводиться так само, як звичайне
    arr = build_index(DATA, "array")
    assert len(arr) == len(IDX) and [p.doc_id for p in arr["alice"]] == list(IDX.doc_ids("alice"))


def test_open_index_closes_even_on_exception():
    # навіть якщо в with сталась помилка, індекс закривається
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "i.pkl"
        store.save(IDX, path)
        holder = []
        try:
            with store.open_index(path) as ix:
                holder.append(ix)
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert holder[0].closed and len(holder[0]) == 0


# парсер (дерева!)
def test_parser_precedence_tree():
    # AND сильніший за OR
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))


def test_parser_parens_not_and_phrase():
    # дужки, NOT і фраза в лапках
    assert parse("python AND (async OR await) NOT java") == And(
        And(Term("python"), Or(Term("async"), Term("await"))), Not(Term("java")))
    assert parse('"event loop"') == Phrase(("event", "loop"))


def test_operator_overloading_builds_same_tree():
    # дерево зі значків & | ~ таке саме, як з рядка
    built = Term("a") & (Term("b") | Term("c")) & ~Term("d")
    assert built == parse("a AND (b OR c) NOT d")


def test_parse_errors():
    # погані запити мають давати ParseError
    for bad in ("", "(a", "a OR", "NOT", "a )"):
        try:
            parse(bad)
            raise AssertionError(f"no error for {bad!r}")
        except ParseError:
            pass


def test_evaluate_and_phrase():
    # фраза має шукатись саме в такому порядку слів
    assert parse("cheshire cat").evaluate(IDX) == {5, 7}
    assert parse('"cheshire cat"').evaluate(IDX) == {5, 7}
    assert parse('"cat cheshire"').evaluate(IDX) == set()
    assert parse("queen NOT duchess").evaluate(IDX) == {7}


# ранжування
def test_scorers_are_interchangeable():
    # TfIdf і BM25 можна підставляти замість одне одного
    for scorer in (TfIdf(), BM25()):
        # пошуковик з цим оцінювачем
        s = Searcher(IDX, scorer)
        # шукаю три найкращі
        res = s.search("queen OR duchess", 3)
        assert res == sorted(res, reverse=True) and len(res) == 3
        assert scorer("alice", Posting(0, 1), IDX) == scorer.score("alice", Posting(0, 1), IDX)


def test_bm25_sanity_checks():
    # три перевірки: рідкісне важить більше, насичення, короткий кращий за довгий
    b = BM25()
    mid = sorted(IDX.doc_lengths, key=IDX.doc_lengths.get)[6]
    terms = sorted(IDX, key=IDX.df)
    rare, common = terms[0], terms[-1]
    assert b.score(rare, Posting(mid, 1), IDX) > b.score(common, Posting(mid, 1), IDX)
    s = {tf: b.score(rare, Posting(mid, tf), IDX) for tf in (1, 2, 20, 21)}
    assert (s[21] - s[20]) < 0.05 * (s[2] - s[1])
    ids = sorted(IDX.doc_lengths, key=IDX.doc_lengths.get)
    assert b.score(rare, Posting(ids[0], 1), IDX) > b.score(rare, Posting(ids[-1], 1), IDX)


def test_search_result_ordering():
    # результати порівнюються лише за балом
    a, b = SearchResult(1, 0.5, "x"), SearchResult(2, 0.9, "y")
    assert a < b and sorted([b, a]) == [a, b]


# ---------------------------------------------------------------- декоратори і кеш
def test_timed_keeps_metadata_and_logs():
    # @timed не губить ім'я функції і пише в лог
    assert build_index.__name__ == "build_index" and build_index.__doc__ is not None
    records = []
    handler = logging.Handler()
    handler.emit = records.append  # збираємо повідомлення логера в список
    log = logging.getLogger("findex.timing")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        Searcher(IDX).search("alice", 1)
    finally:
        log.removeHandler(handler)
    assert any("Searcher.search took" in r.getMessage() for r in records)


def test_lru_cache_hit_on_repeated_query():
    # другий однаковий запит бере результат із кешу
    s = Searcher(IDX)
    s.search("alice rabbit")
    s.search("alice rabbit")
    assert s._match.cache_info().hits == 1 and s._match.cache_info().misses == 1


#  уривки
def test_snippet_highlights_and_keeps_case():
    # слово виділяється зірочками, а регістр літер лишається оригінальним
    text = "x " * 100 + "The Queen arrived. " + "y " * 100
    snip = make_snippet(text, {"queen"}, width=20)
    assert "**Queen**" in snip and snip.startswith("… ") and snip.endswith(" …")
    assert make_snippet("nothing here", {"queen"}) == ""


if __name__ == "__main__":
    # запускаємо кожну функцію test_* і друкуємо результат
    tests = [o for n, o in list(globals().items()) if n.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print("PASS", t.__name__)
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, e)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
