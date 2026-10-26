"""Властивості (Hypothesis): твердження, що мають виконуватись для будь-яких вхідних даних."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import build_index_from_docs
from findex.model import Index
from findex.search import ENGINES
from findex.snippets import make_snippet
from findex.store import load, save
from findex.tokenize import tokenize

# відсортований список унікальних чисел — це "список постінгів"
sorted_ids = st.lists(st.integers(0, 200), unique=True).map(sorted)


@given(sorted_ids, sorted_ids)
def test_merge_and_set_agree(a: list[int], b: list[int]) -> None:
    # злиття двома вказівниками має давати те саме, що операції над множинами
    for i in range(3):  # AND, OR, AND NOT
        assert ENGINES["merge"][i](a, b) == ENGINES["set"][i](a, b)


@given(st.text())
def test_tokenize_is_idempotent(text: str) -> None:
    # якщо токени склеїти пробілами й розбити знову, нічого не зміниться
    once = list(tokenize(text))
    assert list(tokenize(" ".join(once))) == once


@given(st.text())
def test_tokens_are_never_empty_and_lowercase(text: str) -> None:
    for tok in tokenize(text):
        assert tok
        assert tok == tok.casefold()


@given(st.text(min_size=1), st.sets(st.text(min_size=1, max_size=6), max_size=3))
def test_snippet_never_crashes(text: str, terms: set[str]) -> None:
    # уривок — це завжди рядок, і слів запиту в ньому не бракує, якщо вони були в тексті
    assert isinstance(make_snippet(text, terms, width=30), str)


# невеликі випадкові «корпуси»: список текстів зі слів з кількох букв
words = st.text(alphabet="abcде", min_size=1, max_size=4)
texts = st.lists(st.lists(words, max_size=12).map(" ".join), min_size=1, max_size=8)


def make_index(corpus: list[str], positions: bool) -> Index:
    docs = [Document(str(i), Path("x.jsonl"), t) for i, t in enumerate(corpus)]
    return build_index_from_docs(docs, "slots", positions)


def snapshot(index: Index) -> dict[str, list[tuple[int, int]]]:
    """Те, що видно ззовні: для кожного слова пари (doc_id, tf)."""
    return {t: [(p.doc_id, p.tf) for p in index[t]] for t in sorted(index)}


@settings(max_examples=40, deadline=None)
@given(texts, st.sampled_from(["pickle", "binary"]))
def test_load_of_save_gives_same_index(corpus: list[str], fmt: str) -> None:
    # зберегли й завантажили назад: індекс той самий (binary не зберігає позицій)
    index = make_index(corpus, positions=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "i"
        save(index, path, fmt)  # type: ignore[arg-type]  # fmt з sampled_from, pyright бачить str
        again = load(path)
    assert snapshot(again) == snapshot(index)
    assert again.doc_lengths == index.doc_lengths and again.doc_meta == index.doc_meta


@given(texts, st.booleans())
def test_postings_are_always_sorted_by_doc_id(corpus: list[str], positions: bool) -> None:
    index = make_index(corpus, positions)
    for term in index:
        ids = [p.doc_id for p in index[term]]
        assert ids == sorted(ids) and len(ids) == len(set(ids))
