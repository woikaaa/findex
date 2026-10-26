"""Перевірки побудови індексу, частин (Partial) і злиття (merge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from findex.corpus import CorpusError, list_corpus_files
from findex.index import build_index, build_partial, merge
from findex.model import Index, Posting, Storage

STORAGES: list[Storage] = ["dataclass", "slots", "array"]


def test_index_basic_shape(index: Index) -> None:
    assert index.num_docs == 13
    assert len(index) == len(index.postings) > 0
    assert "alice" in index
    assert "zzzz-not-a-word" not in index


def test_getitem_missing_term_raises_keyerror(index: Index) -> None:
    with pytest.raises(KeyError):
        index["zzzz-not-a-word"]


def test_postings_are_sorted_and_positions_match_tf(index: Index) -> None:
    for plist in index.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)
        assert all(len(p.positions) == p.tf for p in plist)


def test_avg_doc_length_is_cached(index: Index) -> None:
    first = index.avg_doc_length
    assert first == sum(index.doc_lengths.values()) / index.num_docs
    assert "avg_doc_length" in index.__dict__  # cached_property кладе значення в об'єкт


def test_df_equals_posting_count(index: Index) -> None:
    assert index.df("alice") == len(index["alice"])


def test_posting_and_docmeta_are_hashable() -> None:
    assert len({Posting(1, 2), Posting(1, 2), Posting(2, 1)}) == 2


@pytest.mark.parametrize("storage", STORAGES)
def test_all_storages_agree(corpus_dir: Path, index: Index, storage: Storage) -> None:
    other = build_index(corpus_dir, storage)
    assert sorted(other) == sorted(index)
    assert list(other.doc_ids("alice")) == list(index.doc_ids("alice"))
    assert other.doc_lengths == index.doc_lengths


def test_positions_with_array_storage_is_rejected(corpus_dir: Path) -> None:
    with pytest.raises(ValueError, match="positions"):
        build_index(corpus_dir, "array", positions=True)


def test_limit_takes_first_documents(corpus_dir: Path) -> None:
    assert build_index(corpus_dir, limit=4).num_docs == 4


def test_bad_corpus_path_raises(tmp_path: Path) -> None:
    bad = tmp_path / "x.csv"
    bad.write_text("a,b", encoding="utf-8")
    with pytest.raises(CorpusError):
        build_index(bad)


# --- Partial і merge: основа для паралельної збірки ---
def test_sequential_partial_merge_equals_build_index(corpus_dir: Path, index: Index) -> None:
    paths = list_corpus_files(corpus_dir)
    merged = merge([build_partial(paths, 0, "slots", True)])
    assert merged.postings == index.postings
    assert merged.doc_lengths == index.doc_lengths


def test_split_partials_merge_to_same_index(corpus_dir: Path, index: Index) -> None:
    paths = list_corpus_files(corpus_dir)
    mid = len(paths) // 2
    # другому шматку одразу даю його діапазон doc_id
    parts = [
        build_partial(paths[:mid], 0, "slots", True),
        build_partial(paths[mid:], mid, "slots", True),
    ]
    assert merge(parts).postings == index.postings


def test_merge_rejects_overlapping_ranges(corpus_dir: Path) -> None:
    paths = list_corpus_files(corpus_dir)
    a = build_partial(paths[:3], 0)
    b = build_partial(paths[3:6], 1)  # діапазон перетинається з a
    with pytest.raises(ValueError, match="overlap"):
        merge([a, b])


def test_merge_rejects_mixed_settings(corpus_dir: Path) -> None:
    paths = list_corpus_files(corpus_dir)
    a = build_partial(paths[:3], 0, "slots")
    b = build_partial(paths[3:6], 3, "dataclass")
    with pytest.raises(ValueError, match="different"):
        merge([a, b])


def test_merge_empty_list_fails() -> None:
    with pytest.raises(ValueError):
        merge([])
