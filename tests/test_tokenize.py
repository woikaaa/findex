"""
Plain assert-style checks for tokenize().
Run with:  python tests/test_tokenize.py
(A real pytest suite arrives in Lab 4 — for now these are just asserts.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from findex.tokenize import tokenize


def toks(text: str) -> list[str]:
    return list(tokenize(text))


def test_mixed_case_is_folded():
    assert toks("Hello WORLD Hello") == ["hello", "world", "hello"]


def test_german_sharp_s_casefolds_like_ss():
    # casefold(), not lower(): "straße" and "strasse" become the same token
    assert toks("Straße") == ["strasse"]


def test_cyrillic_text_is_tokenized():
    assert toks("Привіт, світ!") == ["привіт", "світ"]


def test_combining_mark_accent_normalizes_to_nfc():
    # "café" written as e + combining acute (U+0301) vs. as é (U+00E9)
    composed = "café"          # é is one code point (U+00E9)
    decomposed = "cafe\u0301"  # e + combining acute accent
    assert toks(composed) == toks(decomposed) == ["café"]


def test_apostrophe_kept_inside_word():
    assert toks("don't") == ["don't"]
    assert toks("п'ять") == ["п'ять"]


def test_apostrophe_not_kept_at_edges():
    # a leading/trailing quote is punctuation, not part of the word
    assert toks("'hello'") == ["hello"]


def test_hyphen_kept_inside_word():
    assert toks("well-known") == ["well-known"]


def test_digits_form_their_own_tokens():
    assert toks("Python 3.12 released in 2023") == [
        "python", "3", "12", "released", "in", "2023",
    ]


def test_punctuation_is_dropped():
    assert toks("Wait... what?! Really?") == ["wait", "what", "really"]


def test_empty_string_yields_nothing():
    assert toks("") == []

def test_stress_mark_is_removed():
    # у Вікіпедії наголос позначений U+0301: Архітекту́ра має бути одним словом
    assert toks("Архітекту\u0301ра") == ["архітектура"]

    
if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError:
            failed += 1
            print(f"FAIL {t.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
