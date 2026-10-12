"""
corpus.py — reads a corpus one document at a time, without ever
holding the whole thing in memory.

Supports two layouts:
  1. A directory tree of ``*.txt`` files   (Path("data").rglob("*.txt"))
  2. A single JSON-Lines file (``*.jsonl``), one document per line,
     each line like: {"id": "...", "text": "..."}

Both readers are generators: calling iter_documents(root) does no
work at all. Work only happens while something pulls values from it
with next() / a for-loop.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """One document in the corpus."""
    doc_id: str
    path: Path
    text: str


def _iter_txt_dir(root: Path) -> Iterator[Document]:
    """Yield one Document per *.txt file under root, lazily.

    rglob() itself is a generator — it does not build a list of every
    path before we start; it discovers files as we ask for the next one.
    """
    for path in sorted(root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (UnicodeDecodeError, OSError) as exc:
            # A single bad file must never kill a multi-hour index build.
            logger.warning("skipping unreadable file %s: %s", path, exc)
            continue
        yield Document(doc_id=path.stem, path=path, text=text)


def _iter_jsonl(path: Path) -> Iterator[Document]:
    """Yield one Document per line of a JSON-Lines file, lazily.

    We open the file and iterate it line by line — this is exactly the
    "files are iterators" idiom from the lab theory: one line is read
    into memory at a time, never the whole file.
    """
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping bad JSON on line %d: %s", line_no, exc)
                continue
            doc_id = str(record.get("id", line_no))
            text = record.get("text", "")
            yield Document(doc_id=doc_id, path=path, text=text)


def iter_documents(root: Path) -> Iterator[Document]:
    """Stream every document under ``root``.

    ``root`` can be:
      - a directory  -> every *.txt file under it (recursively), OR
      - a *.jsonl file -> one document per line.

    This function is itself a generator: `next(iter_documents(root))`
    returns almost instantly, even for a corpus that would not fit in
    memory as a whole, because nothing beyond the first file/line has
    been touched yet.
    """
    root = Path(root)
    if root.is_dir():
        yield from _iter_txt_dir(root)
    elif root.is_file() and root.suffix == ".jsonl":
        yield from _iter_jsonl(root)
    else:
        raise ValueError(
            f"{root} is neither a directory of .txt files nor a .jsonl file"
        )
