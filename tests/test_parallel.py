"""Перевірки паралельної збірки індексу."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import list_corpus_files
from findex.index import build_index
from findex.model import Index
from findex.parallel import EXECUTORS, Executors, build_parallel, split_into_chunks


@pytest.mark.parametrize("n", [1, 2, 3, 5, 13, 50])
def test_chunks_are_contiguous_and_cover_everything(corpus_dir: Path, n: int) -> None:
    paths = list_corpus_files(corpus_dir)
    chunks = split_into_chunks(paths, n)
    assert len(chunks) == min(n, len(paths))
    flat = [p for c in chunks for p in c.paths]
    assert flat == paths  # нічого не загубили і порядок той самий
    # діапазон doc_id кожного шматка починається там, де скінчився попередній
    expected = 0
    for c in chunks:
        assert c.first_doc_id == expected and len(c.paths) >= 1
        expected += len(c.paths)


def test_chunks_are_balanced_by_size(tmp_path: Path) -> None:
    # один великий файл і багато малих: великий має сидіти у своєму шматку
    (tmp_path / "a_big.txt").write_text("x " * 5000, encoding="utf-8")
    for i in range(10):
        (tmp_path / f"b{i}.txt").write_text("y", encoding="utf-8")
    chunks = split_into_chunks(list_corpus_files(tmp_path), 2)
    assert len(chunks[0].paths) == 1 and len(chunks[1].paths) == 10


def test_chunks_reject_zero() -> None:
    with pytest.raises(ValueError):
        split_into_chunks([], 0)


def _same(a: Index, b: Index) -> None:
    assert a.postings == b.postings
    assert a.doc_lengths == b.doc_lengths and a.doc_meta == b.doc_meta


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("kind", ["serial", "threads"])
def test_parallel_equals_sequential(
    corpus_dir: Path, index: Index, kind: Executors, workers: int
) -> None:
    paths = list_corpus_files(corpus_dir)
    res = build_parallel(paths, workers, kind, "slots", True)
    _same(res.index, index)
    assert sum(res.chunk_sizes) == len(paths)


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 3])
def test_processes_equal_sequential(corpus_dir: Path, index: Index, workers: int) -> None:
    # справжні процеси (spawn): повільніше, тому позначено slow
    res = build_parallel(list_corpus_files(corpus_dir), workers, "processes", "slots", True)
    _same(res.index, index)
    assert res.worker_cpu_s >= 0  # на Windows годинник грубий (~16 мс), тож 0 теж можливий


@pytest.mark.parametrize("kind", ["serial", "threads"])
def test_worker_failure_propagates(corpus_dir: Path, kind: Executors) -> None:
    # неправильне сховище ламає воркер: збірка має впасти, а не зависнути
    with pytest.raises(ValueError):
        build_parallel(list_corpus_files(corpus_dir), 2, kind, "nope")  # type: ignore[arg-type]  # навмисно неправильне сховище, щоб воркер упав


@pytest.mark.slow
def test_process_worker_failure_propagates(corpus_dir: Path) -> None:
    with pytest.raises(ValueError):
        build_parallel(list_corpus_files(corpus_dir), 2, "processes", "nope")  # type: ignore[arg-type]  # навмисно неправильне сховище, щоб воркер упав


def test_progress_callback_counts_every_file(corpus_dir: Path) -> None:
    paths = list_corpus_files(corpus_dir)
    seen: list[int] = []
    build_parallel(paths, 3, "threads", on_doc=lambda: seen.append(1))
    assert len(seen) == len(paths)


def test_build_index_accepts_workers(corpus_dir: Path, index: Index) -> None:
    _same(build_index(corpus_dir, "slots", True, workers=2, executor="threads"), index)


def test_unknown_executor_rejected(corpus_dir: Path) -> None:
    with pytest.raises(ValueError, match="executor"):
        build_index(corpus_dir, executor="gpu", workers=2)  # type: ignore[arg-type]  # навмисно неправильний виконавець


def test_executors_constant() -> None:
    assert set(EXECUTORS) == {"serial", "threads", "processes"}


def test_cli_workers_flag(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "i.bin"
    res = CliRunner().invoke(
        app, ["index", str(corpus_dir), "-o", str(out), "--workers", "2", "--executor", "threads"]
    )
    assert res.exit_code == 0 and out.exists()
    bad = CliRunner().invoke(app, ["index", str(corpus_dir), "--executor", "gpu"])
    assert bad.exit_code == 2


def test_race_demo_with_lock_never_loses_updates() -> None:
    from findex.race_demo import race_counter, race_int

    assert race_int(4, 5000, use_lock=True) == 0
    assert race_counter(4, 5000, use_lock=True) == 0


def test_bench_cell_and_markdown(corpus_dir: Path) -> None:
    from findex.bench_parallel import run_cell, to_markdown

    cell = run_cell(corpus_dir, "threads", 2, "slots")
    assert cell["wall"] > 0 and cell["rss_mb"] > 0
    rows = [
        {"executor": "serial", "workers": 1, "speedup": 1.0, **cell},
        {"executor": "processes", "workers": 2, "speedup": 1.5, **cell},
    ]
    md = to_markdown(rows, corpus_dir, repeats=1)
    assert "| serial | 1 |" in md and "Закон Амдала" in md


@pytest.mark.slow
def test_cli_default_executor_for_array_storage(corpus_dir: Path, tmp_path: Path) -> None:
    # з --storage array типово беремо процеси (вони там швидші); результат той самий
    out = tmp_path / "i.bin"
    res = CliRunner().invoke(
        app, ["index", str(corpus_dir), "-o", str(out), "--storage", "array", "--format", "binary"]
    )
    assert res.exit_code == 0 and out.exists()


def test_cli_default_executor_for_slots_is_serial(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "i.bin"
    res = CliRunner().invoke(app, ["index", str(corpus_dir), "-o", str(out)])
    assert res.exit_code == 0 and out.exists()
