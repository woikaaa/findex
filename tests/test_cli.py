"""Перевірки командного рядка через typer.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("index", "search", "stats"):
        assert cmd in res.output


def test_index_then_search_json(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "i.bin"
    res = runner.invoke(app, ["index", str(corpus_dir), "-o", str(out)])
    assert res.exit_code == 0 and out.exists()
    res = runner.invoke(app, ["search", "rabbit", "-i", str(out), "--json", "-k", "2"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert len(data) == 2 and {"doc_id", "score", "title", "snippet"} <= set(data[0])


def test_search_table_has_results(saved_index: Path) -> None:
    res = runner.invoke(app, ["search", '"cheshire cat"', "-i", str(saved_index)])
    assert res.exit_code == 0 and "результатів" in res.output


def test_search_no_results(saved_index: Path) -> None:
    res = runner.invoke(app, ["search", "zzzzqqq", "-i", str(saved_index)])
    assert res.exit_code == 0 and "нічого не знайдено" in res.output


def test_search_bad_query_exit_2(saved_index: Path) -> None:
    res = runner.invoke(app, ["search", "a AND", "-i", str(saved_index)])
    assert res.exit_code == 2 and "Traceback" not in res.output


def test_search_missing_index_exit_2(tmp_path: Path) -> None:
    res = runner.invoke(app, ["search", "a", "-i", str(tmp_path / "no.bin")])
    assert res.exit_code == 2 and "not found" in res.output


def test_index_missing_corpus_exit_2(tmp_path: Path) -> None:
    res = runner.invoke(app, ["index", str(tmp_path / "nope")])
    assert res.exit_code == 2


def test_index_positions_with_binary_is_rejected(corpus_dir: Path, tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        ["index", str(corpus_dir), "-o", str(tmp_path / "i"), "--positions", "--format", "binary"],
    )
    assert res.exit_code == 2


def test_stats_json(corpus_dir: Path) -> None:
    res = runner.invoke(app, ["stats", str(corpus_dir), "--json", "--top", "3"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["documents"] == 13 and len(data["top"]) == 3


@pytest.mark.parametrize("bad", ["--scorer", "--storage"])
def test_unknown_literal_value_is_rejected(bad: str, saved_index: Path, corpus_dir: Path) -> None:
    # typer перевіряє значення Literal сам: зайве значення дає код 2
    args = ["search", "a", "-i", str(saved_index), bad, "nope"]
    if bad == "--storage":
        args = ["index", str(corpus_dir), bad, "nope"]
    assert runner.invoke(app, args).exit_code == 2
