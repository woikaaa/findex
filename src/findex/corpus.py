"""
читає корпус текстів по одному документу, не тримаючи весь корпус у пам'яті.

"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# логер, щоб писати попередження про погані файли
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """Один документ корпусу."""
    doc_id: str   # ім'я документа (для файлу це ім'я без розширення)
    path: Path    # звідки він прочитаний
    text: str     # весь текст документа


def _iter_txt_dir(root: Path) -> Iterator[Document]:
    # sorted() потрібен, щоб порядок файлів завжди був однаковим
    for path in sorted(root.rglob("*.txt")):
        # пробую прочитати файл, а раптом він зіпсований
        try:
            # errors="replace" замінює погані символи, а не падає з помилкою
            text = path.read_text(encoding="utf-8", errors="replace")
        except (UnicodeDecodeError, OSError) as exc:
            # один зіпсований файл не повинен зупиняти всю програму
            logger.warning("skipping unreadable file %s: %s", path, exc)
            # переходжу до наступного файлу
            continue
        # yield віддає документ назовні і ставить функцію "на паузу"
        yield Document(doc_id=path.stem, path=path, text=text)


def _iter_jsonl(path: Path) -> Iterator[Document]:
    # відкриваю файл, він сам закриється після with
    with path.open(encoding="utf-8", errors="replace") as f:
        # читаю файл по рядку, номер рядка мені потрібен для повідомлень
        for line_no, line in enumerate(f, start=1):
            # прибираю зайві пробіли з країв
            line = line.strip()
            if not line:
                continue  # порожні рядки пропускаємо
            try:
                # перетворюю рядок JSON на словник
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping bad JSON on line %d: %s", line_no, exc)
                continue
            # якщо в записі немає id, беремо номер рядка
            doc_id = str(record.get("id", line_no))
            # якщо тексту немає, беру порожній рядок
            text = record.get("text", "")
            yield Document(doc_id=doc_id, path=path, text=text)


def iter_documents(root: Path) -> Iterator[Document]:
    # роблю з рядка справжній шлях, щоб працювали методи Path
    root = Path(root)
    # якщо це папка, читаю з неї .txt файли
    if root.is_dir():
        # yield from передає все, що віддає та функція
        yield from _iter_txt_dir(root)
    # якщо це файл .jsonl, читаю його по рядках
    elif root.is_file() and root.suffix == ".jsonl":
        yield from _iter_jsonl(root)
    else:
        # нічого не підійшло, тому кажу про помилку
        raise ValueError(
            f"{root} is neither a directory of .txt files nor a .jsonl file"
        )
