"""Перевірки лаби 8: пошук на NumPy дає те саме, що пошук з лаби 3, а новий make_snippet
дає той самий текст, що й старий."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from snippet_reference import reference_make_snippet

from findex.corpus import Document
from findex.index import build_index_from_docs
from findex.model import Index
from findex.npsearch import NpIndex, NpSearcher, top_k
from findex.query import QueryError
from findex.ranking import Searcher, SearchResult
from findex.scoring import BM25, SCORER_NAMES, ScorerName, TfIdf, make_scorer
from findex.snippets import _to_nfc, make_snippet
from findex.store import save


def as_tuples(results: list[SearchResult]) -> list[tuple[int, float, str]]:
    """Все, що є в результаті; бал порівнюю ТОЧНО (==), без допуску."""
    return [(r.doc_id, r.score, r.title) for r in results]


def queries_for(index: Index) -> list[str]:
    """Набір різних запитів з реальних слів індексу."""
    t = sorted(index.postings, key=lambda w: (-index.df(w), w))
    return [
        t[0],  # найчастіше слово
        t[-1],  # найрідше слово
        f"{t[3]} OR {t[10]} OR {t[20]}",
        f"{t[2]} AND {t[5]}",
        f"{t[1]} AND NOT {t[7]}",
        f"NOT {t[0]}",
        f"({t[4]} OR {t[6]}) AND {t[1]}",
        f"{t[3]} {t[9]}",
        "такогослованемає",
    ]


# ---------------------------------------------------------------- те саме, що лаба 3
@pytest.mark.parametrize("name", SCORER_NAMES)
@pytest.mark.parametrize("k", [1, 3, 10, 1000])
def test_same_results_as_lab3(index: Index, name: ScorerName, k: int) -> None:
    old = Searcher(index, make_scorer(name))
    new = NpSearcher(NpIndex.from_index(index), make_scorer(name))
    for q in queries_for(index):
        assert as_tuples(new.search(q, k)) == as_tuples(old.search(q, k)), q
        assert new.count(q) == old.count(q)
        assert new.terms(q) == old.terms(q)


def test_equal_scores_smaller_doc_id_first() -> None:
    """Однакові документи дають однакові бали: порядок має визначатися лише doc_id."""
    docs = [Document(f"d{i}", Path(f"d{i}.txt"), "кіт пес кіт") for i in range(8)]
    index = build_index_from_docs(docs)
    new = NpSearcher(NpIndex.from_index(index))
    res = new.search("кіт", 3)
    assert [r.doc_id for r in res] == [0, 1, 2]
    assert as_tuples(res) == as_tuples(Searcher(index).search("кіт", 3))


words = st.sampled_from(["кіт", "пес", "миша", "сир", "дім", "ліс", "ріка", "гора"])
texts = st.lists(words, min_size=1, max_size=12).map(" ".join)


@settings(max_examples=150, deadline=None)
@given(
    corpus=st.lists(texts, min_size=1, max_size=15),
    query=st.lists(words, min_size=1, max_size=3),
    op=st.sampled_from([" OR ", " AND ", " AND NOT "]),
    k=st.integers(min_value=0, max_value=20),
    name=st.sampled_from(SCORER_NAMES),
)
def test_property_same_as_lab3(
    corpus: list[str], query: list[str], op: str, k: int, name: ScorerName
) -> None:
    """На випадкових маленьких корпусах (там багато рівних балів) результат той самий."""
    docs = [Document(f"d{i}", Path(f"d{i}.txt"), t) for i, t in enumerate(corpus)]
    index = build_index_from_docs(docs)
    q = op.join(query)
    old = Searcher(index, make_scorer(name))
    new = NpSearcher(NpIndex.from_index(index), make_scorer(name))
    assert as_tuples(new.search(q, k)) == as_tuples(old.search(q, k))


# ---------------------------------------------------------------- масиви int32 і файл
def test_postings_are_int32(index: Index) -> None:
    ix = NpIndex.from_index(index)
    assert ix.ids.dtype == np.int32
    assert ix.tfs.dtype == np.int32
    assert ix.doc_lengths.dtype == np.int32
    assert len(ix.ids) == sum(index.df(t) for t in index.postings)
    # довжини документів лежать в одному масиві, індекс це doc_id
    for doc_id, n in index.doc_lengths.items():
        assert ix.doc_lengths[doc_id] == n


@pytest.mark.parametrize("mmap", [True, False])
def test_from_file_matches(index: Index, tmp_path: Path, mmap: bool) -> None:
    path = tmp_path / "i.bin"
    no_positions = Index(
        "slots", False, index.postings, index.doc_lengths, index.doc_meta
    )  # у бінарному форматі позицій немає
    save(no_positions, path, "binary")
    ix = NpIndex.from_file(path, mmap=mmap)
    assert ix.ids.dtype == np.int32
    assert isinstance(ix.ids, np.memmap) == mmap
    old = Searcher(no_positions)
    new = NpSearcher(ix)
    for q in queries_for(index):
        assert as_tuples(new.search(q, 10)) == as_tuples(old.search(q, 10)), q


def test_from_file_rejects_foreign_file(tmp_path: Path) -> None:
    bad = tmp_path / "x.bin"
    bad.write_bytes(b"not an index")
    with pytest.raises(ValueError, match="not a findex binary file"):
        NpIndex.from_file(bad)


# ---------------------------------------------------------------- крайні випадки
def test_phrase_query_is_not_supported(index: Index) -> None:
    new = NpSearcher(NpIndex.from_index(index))
    with pytest.raises(QueryError):
        new.search('"cheshire cat"')


def test_only_bm25_and_tfidf(index: Index) -> None:
    class Odd:
        def score(self, term: str, posting: object, index: Index) -> float:
            return 1.0

    with pytest.raises(TypeError):
        NpSearcher(NpIndex.from_index(index), Odd())  # type: ignore[arg-type]  # навмисно


def test_k_zero_and_empty(index: Index) -> None:
    new = NpSearcher(NpIndex.from_index(index))
    assert new.search("alice", 0) == []
    assert new.search("такогослованемає", 5) == []


def test_default_scorer_is_bm25(index: Index) -> None:
    assert isinstance(NpSearcher(NpIndex.from_index(index)).scorer, BM25)
    assert isinstance(NpSearcher(NpIndex.from_index(index), TfIdf()).scorer, TfIdf)


@given(
    scores=st.lists(st.sampled_from([0.0, 0.5, 1.0, 2.0]), min_size=0, max_size=30),
    k=st.integers(min_value=0, max_value=35),
)
def test_top_k_matches_full_sort(scores: list[float], k: int) -> None:
    """top_k з argpartition дає те саме, що повне сортування (багато рівних балів)."""
    arr = np.array(scores, dtype=np.float64)
    ids = np.arange(len(scores), dtype=np.int32) * 3  # doc_id зростають, але не підряд
    expected = sorted(range(len(scores)), key=lambda i: (-scores[i], ids[i]))[:k]
    assert top_k(ids, arr, k).tolist() == expected


# ---------------------------------------------------------------- новий make_snippet
SNIP_WORDS = ["кіт", "Кіт", "ПЕС", "миша", "сир", "й", "ß", "straße", "a", "b"]


@settings(max_examples=250, deadline=None)
@given(
    words_=st.lists(st.sampled_from(SNIP_WORDS), min_size=0, max_size=700),
    terms=st.sets(st.sampled_from(["кіт", "пес", "миша", "a", "b", "strasse"]), max_size=4),
    width=st.sampled_from([0, 5, 40, 80, 200]),
    decompose=st.booleans(),
)
def test_snippet_same_as_reference(
    words_: list[str], terms: set[str], width: int, decompose: bool
) -> None:
    """Швидкий make_snippet дає точно той самий текст, що й старий (еталон у тестах)."""
    text = " ".join(words_)
    if decompose:
        text = text.replace("й", "й")  # розкладена форма (потрібна нормалізація)
    assert make_snippet(text, terms, width) == reference_make_snippet(text, terms, width)


def test_snippet_same_on_fixtures(corpus_dir: Path) -> None:
    for path in sorted(corpus_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for terms in ({"alice"}, {"the", "queen"}, {"rabbit", "hole", "down"}, {"zzzz"}):
            assert make_snippet(text, terms) == reference_make_snippet(text, terms), path


@settings(max_examples=300, deadline=None)
@given(
    parts=st.lists(
        st.sampled_from(
            [
                "а",
                "и",
                "й",
                "и\u0306",
                "e\u0301",
                "\u0301",
                "\u0308",
                "i",
                " ",
                "\n",
                "\u00a0",
                "ab",
            ]
        ),
        max_size=3000,
    )
)
def test_chunked_nfc_equals_normalize(parts: list[str]) -> None:
    """Нормалізація шматками дає точно те саме, що normalize на всьому тексті."""
    text = "".join(parts)
    assert _to_nfc(text) == unicodedata.normalize("NFC", text)
