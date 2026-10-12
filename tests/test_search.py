"""Перевірки через assert: merge проти set, хешування, збереження і завантаження. python tests/test_search.py"""

import random
import sys
import tempfile
from pathlib import Path

# додаємо папку src, щоб Python знайшов пакет findex
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from findex import store
from findex.index import build_index
from findex.model import DocMeta, Posting
from findex.search import ENGINES, evaluate

# маленький набір текстів для тестів (лежить у репозиторії)
DATA = Path(__file__).resolve().parent / "fixtures"


def test_merge_equals_set_on_random_lists():
    # на випадкових списках обидва способи мають давати однаковий результат
    rng = random.Random(1)  # фіксуємо випадковість, щоб тест був повторюваним
    for _ in range(200):
        a = sorted(rng.sample(range(50), rng.randint(0, 20)))
        b = sorted(rng.sample(range(50), rng.randint(0, 20)))
        for i in range(3):  # 0 = AND, 1 = OR, 2 = AND NOT
            assert ENGINES["merge"][i](a, b) == ENGINES["set"][i](a, b)


def test_posting_and_docmeta_are_hashable():
    # frozen-класи можна класти у множину, однакові записи зливаються в один
    assert len({Posting(1, 2), Posting(1, 2), Posting(2, 1)}) == 2
    assert len({DocMeta("a", "t"), DocMeta("a", "t")}) == 1


def test_all_storages_give_same_answers():
    # три способи зберігання мають давати однакову відповідь на запит
    q = "alice NOT rabbit"
    answers = [evaluate(build_index(DATA, s), q) for s in ("dataclass", "slots", "array")]
    assert answers[0] == answers[1] == answers[2] and answers[0]


def test_engines_agree_on_queries():
    # merge і set дають однакові результати на справжніх запитах
    idx = build_index(DATA, "slots")
    for q in ("alice rabbit", "queen OR duchess", "alice NOT rabbit", "NOT alice"):
        assert evaluate(idx, q, "merge") == evaluate(idx, q, "set")


def test_roundtrip_both_formats():
    # зберегли й завантажили назад: відповідь не змінилась (pickle і binary)
    idx = build_index(DATA, "slots")
    want = evaluate(idx, "alice queen")
    with tempfile.TemporaryDirectory() as d:
        for fmt in ("pickle", "binary"):
            path = Path(d) / f"i.{fmt}"
            # зберігаю у файл цим форматом
            store.save(idx, path, fmt)
            assert evaluate(store.load(path), "alice queen") == want


def test_postings_sorted_and_positions_flag():
    # списки документів відсортовані, а позицій стільки, скільки разів слово є
    idx = build_index(DATA, "slots", positions=True)
    for plist in idx.postings.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)
        assert all(len(p.positions) == p.tf for p in plist)


if __name__ == "__main__":
    # запускаємо кожну функцію test_* і друкуємо результат
    tests = [o for n, o in list(globals().items()) if n.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print("PASS", t.__name__)
        except AssertionError:
            failed += 1; print("FAIL", t.__name__)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
