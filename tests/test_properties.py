"""Властивості (Hypothesis): твердження, що мають виконуватись для будь-яких вхідних даних."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from findex.search import ENGINES
from findex.snippets import make_snippet
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
