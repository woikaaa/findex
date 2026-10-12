"""
Прості перевірки через assert для tokenize(). python tests/test_tokenize.py
"""

import sys
from pathlib import Path

# додаємо папку src, щоб Python знайшов пакет findex
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from findex.tokenize import tokenize


def toks(text: str) -> list[str]:
    """Допоміжна функція: слова з тексту одним списком."""
    return list(tokenize(text))


def test_mixed_case_is_folded():
    # великі й малі літери стають однаковими
    assert toks("Hello WORLD Hello") == ["hello", "world", "hello"]


def test_german_sharp_s_casefolds_like_ss():
    # casefold(), а не lower(): "straße" і "strasse" стають одним словом
    assert toks("Straße") == ["strasse"]


def test_cyrillic_text_is_tokenized():
    # кирилиця працює, розділові знаки відкидаються
    assert toks("Привіт, світ!") == ["привіт", "світ"]


def test_combining_mark_accent_normalizes_to_nfc():
    # "café" як e + окремий значок наголосу (U+0301) і як один символ é (U+00E9)
    composed = "café"          # é це один символ (U+00E9)
    decomposed = "café"  # e + окремий значок наголосу
    assert toks(composed) == toks(decomposed) == ["café"]


def test_apostrophe_kept_inside_word():
    # апостроф всередині слова лишається
    assert toks("don't") == ["don't"]
    assert toks("п'ять") == ["п'ять"]


def test_apostrophe_not_kept_at_edges():
    # лапка на початку чи в кінці це розділовий знак, а не частина слова
    assert toks("'hello'") == ["hello"]


def test_hyphen_kept_inside_word():
    # дефіс всередині слова лишається
    assert toks("well-known") == ["well-known"]


def test_digits_form_their_own_tokens():
    # цифри теж стають окремими словами
    assert toks("Python 3.12 released in 2023") == [
        "python", "3", "12", "released", "in", "2023",
    ]


def test_punctuation_is_dropped():
    # розділові знаки зникають
    assert toks("Wait... what?! Really?") == ["wait", "what", "really"]


def test_empty_string_yields_nothing():
    # порожній текст не дає жодного слова
    assert toks("") == []


def test_stress_mark_is_removed():
    # у Вікіпедії наголос позначений U+0301: Архітекту́ра має бути одним словом
    assert toks("Архітекту́ра") == ["архітектура"]


if __name__ == "__main__":
    # проста перевірка без pytest: запускаємо кожну функцію test_*
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
