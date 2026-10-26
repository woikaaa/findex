"""
збереження і завантаження Index. Два формати:

  1. pickle  — один рядок щоб зберегти, один щоб завантажити, працює з будь-яким
     об'єктом Python.
  2. binary  — власний формат: магічний підпис + заголовок JSON + сирі масиви array('I').
     Завантажувати безпечно (ми лише читаємо JSON і копіюємо байти).
"""

from __future__ import annotations

import json
import pickle
import struct
import sys
from array import array
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from findex.model import DocMeta, Entry, Index
from findex.utils import timed

# підпис на початку нашого файлу, щоб відрізнити його від чужих
MAGIC = b"FIDX1"

# у якому форматі зберігаємо індекс
Fmt = Literal["pickle", "binary"]


class IndexLoadError(Exception):
    """Файл індексу не існує або зіпсований (очікувана помилка, а не баг)."""


#  pickle
def save_pickle(index: Index, path: Path) -> None:
    """Зберігає індекс через pickle."""
    with open(path, "wb") as f:
        # pickle сам перетворює об'єкт на байти
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Index:
    # УВАГА, БЕЗПЕКА: pickle.load може виконати БУДЬ-ЯКИЙ код. Файл pickle
    # це не просто дані, а маленька програма для pickle: вона може наказати
    # викликати будь-яку функцію (os.system, ...) під час завантаження.
    # Завантажуй лише файли, які створила САМА. Ніколи не відкривай pickle,
    # отриманий з інтернету чи поштою.
    with open(path, "rb") as f:
        # тут і є небезпека: чужий файл міг би виконати код
        return cast("Index", pickle.load(f))


#  binary
def _to_arrays(index: Index) -> dict[str, tuple[array[int], array[int]]]:
    """Перетворює будь-яке сховище індексу на пари масивів (id, tf)."""
    # позиції в бінарному форматі я не зберігаю
    if index.has_positions:
        raise ValueError("binary format does not store positions")
    out: dict[str, tuple[array[int], array[int]]] = {}
    # перекладаю кожне слово в два масиви
    for term, entry in index.postings.items():
        if isinstance(entry, tuple):
            # тут уже все у потрібному вигляді
            out[term] = entry
        else:
            out[term] = (
                array("I", (p.doc_id for p in entry)),
                array("I", (p.tf for p in entry)),
            )
    return out


def save_binary(index: Index, path: Path) -> None:
    """Зберігає індекс у власному бінарному форматі."""
    # готую дані у вигляді масивів
    arrays = _to_arrays(index)
    # складаємо всі числа слів в одні два великі масиви
    ids_all, tfs_all = array("I"), array("I")
    terms: dict[str, list[int]] = {}
    # пишу всі слова в два спільні масиви
    for term, (ids, tfs) in arrays.items():
        terms[term] = [len(ids_all), len(ids)]  # [звідки починається, скільки] елементів
        ids_all.extend(ids)
        tfs_all.extend(tfs)

    # заголовок: усе, що не є великими масивами чисел, у вигляді JSON
    header = json.dumps(
        {
            "byteorder": sys.byteorder,
            "n_postings": len(ids_all),
            "doc_lengths": index.doc_lengths,
            "doc_meta": {i: [m.path, m.title] for i, m in index.doc_meta.items()},
            "terms": terms,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    # порядок у файлі: підпис, довжина заголовка, заголовок, два масиви
    with open(path, "wb") as f:
        # спочатку пишу підпис файлу
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(ids_all.tobytes())
        f.write(tfs_all.tobytes())


def load_binary(path: Path) -> Index:
    """Читає індекс із власного бінарного формату."""
    with open(path, "rb") as f:
        # перевіряю, що це справді мій файл
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError("not a findex binary file")
        # читаю, яка довжина заголовка
        (hlen,) = struct.unpack("<Q", f.read(8))
        # читаю заголовок
        header = cast("dict[str, Any]", json.loads(f.read(hlen).decode("utf-8")))
        n = header["n_postings"]
        ids_all, tfs_all = array("I"), array("I")
        # читаю числа прямо з файлу в масив
        ids_all.fromfile(f, n)
        tfs_all.fromfile(f, n)
    # якщо файл записали на комп'ютері з іншим порядком байтів, розвертаємо
    if header["byteorder"] != sys.byteorder:
        ids_all.byteswap()
        tfs_all.byteswap()

    # розрізаємо великі масиви назад на списки окремих слів
    postings: dict[str, Entry] = {
        t: (ids_all[o : o + c], tfs_all[o : o + c]) for t, (o, c) in header["terms"].items()
    }
    # складаю індекс назад з прочитаних частин
    return Index(
        storage="array",
        has_positions=False,
        postings=postings,
        doc_lengths={int(k): v for k, v in header["doc_lengths"].items()},
        doc_meta={int(k): DocMeta(*v) for k, v in header["doc_meta"].items()},
    )


#  головні функції
def save(index: Index, path: Path, fmt: Fmt = "pickle") -> None:
    """Зберігає індекс у вибраному форматі."""
    if fmt == "binary":
        save_binary(index, Path(path))
    else:
        save_pickle(index, Path(path))


@timed
def load(path: Path) -> Index:
    """Сама визначає формат за першими байтами файлу."""
    # на випадок, якщо передали рядок
    path = Path(path)
    try:
        with open(path, "rb") as f:
            # читаю лише перші байти, щоб зрозуміти формат
            magic = f.read(len(MAGIC))
        return load_binary(path) if magic == MAGIC else load_pickle(path)
    except FileNotFoundError as exc:
        raise IndexLoadError(f"index file not found: {path}") from exc
    except Exception as exc:  # зіпсований файл може дати будь-яку помилку pickle чи json
        raise IndexLoadError(f"cannot read index {path}: {exc}") from exc


@contextmanager
def open_index(path: Path) -> Generator[Index]:
    """with open_index("index.bin") as ix: ...

    Усе, що до `yield`, це вхід у with (__enter__), усе після нього це
    вихід (__exit__). `finally` гарантує, що close() виконається, навіть
    якщо в тілі with була помилка."""
    # вхід у with: завантажую індекс
    index = load(path)
    try:
        yield index
    finally:
        # вихід з with: звільняю пам'ять
        index.close()
