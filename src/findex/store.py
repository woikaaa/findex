"""
store.py — save / load an Index. Two formats:

  1. pickle  — one line to save, one to load, handles any Python object.
  2. binary  — our own format: magic + JSON header + raw array('I') buffers.
               Safe to load (we only parse JSON and copy bytes).
"""

from __future__ import annotations

import json
import pickle
import struct
import sys
from array import array
from contextlib import contextmanager
from pathlib import Path

from findex.model import DocMeta, Index
from findex.utils import timed

MAGIC = b"FIDX1"


# ---------------------------------------------------------------- pickle
def save_pickle(index: Index, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Index:
    # SECURITY: pickle.load can execute ARBITRARY CODE. A pickle file is not
    # data, it is a small program for the unpickler: it may tell it to call
    # any function (os.system, ...) while loading. Only load files YOU
    # created. Never load a pickle received over a network / by e-mail.
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- binary
def _to_arrays(index: Index) -> dict[str, tuple[array, array]]:
    if index.has_positions:
        raise ValueError("binary format does not store positions")
    if index.storage == "array":
        return index.postings
    out = {}
    for term, plist in index.postings.items():
        out[term] = (array("I", (p.doc_id for p in plist)),
                     array("I", (p.tf for p in plist)))
    return out


def save_binary(index: Index, path: Path) -> None:
    arrays = _to_arrays(index)
    ids_all, tfs_all = array("I"), array("I")
    terms: dict[str, list[int]] = {}
    for term, (ids, tfs) in arrays.items():
        terms[term] = [len(ids_all), len(ids)]      # [offset, count] in elements
        ids_all.extend(ids)
        tfs_all.extend(tfs)

    header = json.dumps({
        "byteorder": sys.byteorder,
        "n_postings": len(ids_all),
        "doc_lengths": index.doc_lengths,
        "doc_meta": {i: [m.path, m.title] for i, m in index.doc_meta.items()},
        "terms": terms,
    }, ensure_ascii=False).encode("utf-8")

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(ids_all.tobytes())
        f.write(tfs_all.tobytes())


def load_binary(path: Path) -> Index:
    with open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError("not a findex binary file")
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen).decode("utf-8"))
        n = header["n_postings"]
        ids_all, tfs_all = array("I"), array("I")
        ids_all.fromfile(f, n)
        tfs_all.fromfile(f, n)
    if header["byteorder"] != sys.byteorder:
        ids_all.byteswap()
        tfs_all.byteswap()

    postings = {t: (ids_all[o:o + c], tfs_all[o:o + c])
                for t, (o, c) in header["terms"].items()}
    return Index(
        storage="array",
        has_positions=False,
        postings=postings,
        doc_lengths={int(k): v for k, v in header["doc_lengths"].items()},
        doc_meta={int(k): DocMeta(*v) for k, v in header["doc_meta"].items()},
    )


# ---------------------------------------------------------------- front door
def save(index: Index, path: Path, fmt: str = "pickle") -> None:
    {"pickle": save_pickle, "binary": save_binary}[fmt](index, Path(path))


@timed
def load(path: Path) -> Index:
    """Auto-detect the format by the first bytes of the file."""
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
    return load_binary(path) if magic == MAGIC else load_pickle(path)


@contextmanager
def open_index(path: Path):
    """with open_index("index.bin") as ix: ...

    Everything before `yield` is __enter__, everything after is __exit__.
    `finally` guarantees close() runs even if the body raises."""
    index = load(path)
    try:
        yield index
    finally:
        index.close()
