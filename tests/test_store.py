"""Перевірки збереження і завантаження індексу."""

from __future__ import annotations

from pathlib import Path

import pytest

from findex.index import build_index
from findex.model import Index
from findex.search import evaluate
from findex.store import IndexLoadError, load, open_index, save


@pytest.mark.parametrize("fmt", ["pickle", "binary"])
def test_roundtrip_keeps_answers(corpus_dir: Path, tmp_path: Path, fmt: str) -> None:
    idx = build_index(corpus_dir, "slots")  # binary не зберігає позицій
    path = tmp_path / f"i.{fmt}"
    save(idx, path, fmt)  # type: ignore[arg-type]
    assert evaluate(load(path), "alice queen") == evaluate(idx, "alice queen")


def test_pickle_keeps_positions(index: Index, saved_index: Path) -> None:
    again = load(saved_index)
    assert again.has_positions
    assert evaluate(again, '"cheshire cat"') == [5, 7]


def test_binary_refuses_positions(index: Index, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positions"):
        save(index, tmp_path / "i.bin", "binary")


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IndexLoadError, match="not found"):
        load(tmp_path / "nope.bin")


def test_load_garbage_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"this is not an index")
    with pytest.raises(IndexLoadError, match="cannot read"):
        load(bad)


def test_open_index_closes_even_on_exception(saved_index: Path) -> None:
    holder: list[Index] = []
    with pytest.raises(RuntimeError), open_index(saved_index) as ix:
        holder.append(ix)
        raise RuntimeError("boom")
    assert holder[0].closed and len(holder[0]) == 0
