"""
model.py — the data types of the index.

They live in their OWN module on purpose: pickle stores the *module path*
of every class it saves. If Posting were defined in index.py and we ran
`python -m findex.index`, that module would be called `__main__`, and
`python -m findex.search` could not unpickle the file later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True, slots=True)
class Posting:
    """One (document, term-frequency) pair. Frozen => hashable, slots => small."""
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    """What we need to show a result to the user."""
    path: str
    title: str


@dataclass
class PlainPosting:
    """Same as Posting but a plain dataclass (has __dict__).
    Exists ONLY for the M4 memory comparison."""
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


class Index(Mapping):
    """The inverted index, as an object that behaves like a read-only dict.

    Inheriting collections.abc.Mapping: we implement only __getitem__,
    __iter__ and __len__; Mapping gives keys(), items(), values(), get() ...

    storage:
      "dataclass" -> postings[term] = list[PlainPosting]
      "slots"     -> postings[term] = list[Posting]
      "array"     -> postings[term] = (array('I') of doc_ids, array('I') of tfs)
    index[term] always returns a list of posting objects, whatever the storage.
    """

    def __init__(self, storage: str, has_positions: bool, postings: dict,
                 doc_lengths: dict[int, int], doc_meta: dict[int, DocMeta]) -> None:
        self.storage = storage
        self.has_positions = has_positions
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta
        self.closed = False

    # ---- Mapping protocol (what len(), `in`, [], for ... call behind the scenes)
    def __getitem__(self, term: str):
        entry = self.postings[term]            # KeyError if the term is absent
        if self.storage == "array":
            ids, tfs = entry
            return [Posting(i, t) for i, t in zip(ids, tfs)]
        return entry

    def __iter__(self):
        return iter(self.postings)

    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, term: object) -> bool:   # faster than Mapping's default
        return term in self.postings

    def __repr__(self) -> str:
        return (f"Index(terms={len(self):_}, docs={self.num_docs:_}, "
                f"storage={self.storage!r}, positions={self.has_positions})")

    # ---- computed attributes
    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @cached_property
    def avg_doc_length(self) -> float:
        """Computed once, then cached on the instance (BM25 needs it per document)."""
        return sum(self.doc_lengths.values()) / self.num_docs if self.doc_lengths else 0.0

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        entry = self.postings.get(term)
        if entry is None:
            return 0
        return len(entry[0]) if self.storage == "array" else len(entry)

    # ---- helpers from Lab 2
    def doc_ids(self, term: str):
        """Sorted doc_ids containing `term` (empty if unknown)."""
        entry = self.postings.get(term)
        if entry is None:
            return []
        if self.storage == "array":
            return entry[0]
        return [p.doc_id for p in entry]

    def all_doc_ids(self) -> list[int]:
        return sorted(self.doc_meta)

    def close(self) -> None:
        """Release the big structures (called by open_index, even on exceptions)."""
        self.postings = {}
        self.doc_lengths = {}
        self.doc_meta = {}
        self.__dict__.pop("avg_doc_length", None)
        self.closed = True
