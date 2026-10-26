"""Перевірки читання корпусу: папка з .txt і файл .jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from findex.corpus import CorpusError, iter_documents, list_corpus_files, read_document
from findex.index import build_index
from findex.stats import collect_stats


def test_list_corpus_files_is_sorted(corpus_dir: Path) -> None:
    files = list_corpus_files(corpus_dir)
    assert files == sorted(files) and len(files) == 13


def test_read_document_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_document(tmp_path / "nope.txt") is None


def test_iter_documents_rejects_wrong_path(tmp_path: Path) -> None:
    bad = tmp_path / "a.csv"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(CorpusError, match="neither"):
        list(iter_documents(bad))


def test_jsonl_corpus_is_indexed(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    rows = [{"id": "a", "text": "кіт сидить"}, {"id": "b", "text": "пес бігає"}]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    idx = build_index(path)
    assert idx.num_docs == 2 and "кіт" in idx and "пес" in idx


@pytest.mark.parametrize("eager", [False, True])
def test_stats_lazy_and_eager_agree(corpus_dir: Path, eager: bool) -> None:
    st = collect_stats(corpus_dir, eager=eager, top=5)
    assert st.documents == 13 and len(st.top) == 5
