"""Крізь-тест: справжній локальний демо-сайт -> findex crawl -> JSONL -> індекс -> пошук."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.demosite import start_server
from findex.index import build_index
from findex.ranking import Searcher

runner = CliRunner()


@pytest.fixture
def demo_url() -> str:
    server, _state = start_server(port=0, pages=40, latency=0.0, chaos=True)
    yield f"http://127.0.0.1:{server.server_port}/"  # type: ignore[misc]  # yield у фікстурі повертає str, а не Generator, pyright скаржиться на тип
    server.shutdown()


def test_crawl_cli_then_index_then_search(demo_url: str, tmp_path: Path) -> None:
    out, log, summary = tmp_path / "c.jsonl", tmp_path / "c.log", tmp_path / "s.json"
    res = runner.invoke(
        app,
        [
            "crawl", demo_url, "--contact", "me@example.org", "--max-pages", "30",
            "--concurrency", "5", "--per-host", "5", "--delay", "0", "--max-depth", "6",
            "--timeout", "1", "--out", str(out), "--log", str(log), "--summary", str(summary),
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert 20 <= len(records) <= 30 and all(r["status"] == 200 for r in records)
    assert not any("/private/" in r["url"] for r in records)  # robots.txt послухано

    # лог: статус кожного URL, повтори і причина збою
    text = log.read_text(encoding="utf-8")
    assert "RETRY" in text and "/flaky" in text  # 503 двічі, потім 200
    assert "redirect loop" in text  # /loop-a <-> /loop-b
    secret = [ln for ln in text.splitlines() if "/private/secret" in ln]
    assert secret and all("SKIP robots" in ln for ln in secret)  # пропущено, а не завантажено
    stats = json.loads(summary.read_text(encoding="utf-8"))
    assert stats["pages"] == len(records) and stats["errors"] >= 1

    # JSONL читається тим самим iter_documents з лаби 1, і за ним можна шукати
    docs = list(iter_documents(out))
    assert len(docs) == len(records)
    idx = build_index(out)
    assert idx.num_docs == len(records)
    found = Searcher(idx).search("атом OR планета", 3)
    assert found and found[0].title.startswith("Сторінка")


def test_crawl_cli_requires_contact(demo_url: str) -> None:
    assert runner.invoke(app, ["crawl", demo_url]).exit_code == 2


def test_crawl_cli_rejects_bad_url() -> None:
    res = runner.invoke(app, ["crawl", "mailto:a@b.c", "--contact", "x@y.z"])
    assert res.exit_code == 2


def test_blocking_demo_and_bench_helpers_import() -> None:
    # самі скрипти запускають вручну; тут лише перевіряю, що модулі імпортуються
    import findex.bench_crawl
    import findex.blocking_demo

    assert callable(findex.bench_crawl.main) and callable(findex.blocking_demo.main)


def test_snippet_from_crawled_jsonl(tmp_path: Path) -> None:
    """Для документів із краулера (.jsonl) уривок теж має знаходитись."""
    from findex.index import build_index
    from findex.snippets import snippet_for

    f = tmp_path / "c.jsonl"
    records = [
        {"id": "http://x/1", "title": "Один", "text": "Один\nкіт і пес гуляють"},
        {"id": "http://x/2", "title": "Два", "text": "Два\nатом і зірка"},
    ]
    f.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    idx = build_index(f)
    doc = next(i for i, m in idx.doc_meta.items() if m.title == "Два")
    # без source уривка немає (в індексі лише адреса), із source він є
    assert snippet_for(idx, doc, {"атом"}) == ""
    assert "**атом**" in snippet_for(idx, doc, {"атом"}, source=f)
