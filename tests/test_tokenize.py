"""Перевірки токенізатора."""

from __future__ import annotations

import pytest

from findex.tokenize import tokenize


def toks(text: str) -> list[str]:
    """Допоміжна функція: слова з тексту одним списком."""
    return list(tokenize(text))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello WORLD Hello", ["hello", "world", "hello"]),  # регістр згладжується
        ("Straße", ["strasse"]),  # casefold, а не lower
        ("Привіт, світ!", ["привіт", "світ"]),  # кирилиця
        ("don't", ["don't"]),  # апостроф всередині слова
        ("п'ять", ["п'ять"]),
        ("'hello'", ["hello"]),  # лапки по краях — це розділові знаки
        ("well-known", ["well-known"]),  # дефіс всередині слова
        ("Python 3.12 released", ["python", "3", "12", "released"]),  # цифри
        ("Wait... what?! Really?", ["wait", "what", "really"]),  # пунктуація
        ("", []),  # порожній текст
        ("   \n\t ", []),  # самі пробіли
        ("Архітекту́ра", ["архітектура"]),  # наголос U+0301 зникає
    ],
)
def test_tokenize_cases(text: str, expected: list[str]) -> None:
    assert toks(text) == expected


def test_combining_accent_equals_precomposed() -> None:
    # "é" одним символом і "e" + окремий значок мають дати те саме слово
    assert toks("café") == toks("café") == ["café"]


def test_tokenize_is_lazy() -> None:
    # це генератор: слова віддаються по одному, а не списком
    gen = tokenize("a b c")
    assert next(gen) == "a"
