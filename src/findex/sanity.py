"""
sanity.py — the three ranking sanity checks, on YOUR corpus.

    python -m findex.sanity index.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

from findex import store
from findex.model import Posting
from findex.scoring import BM25


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("index", type=Path)
    a = p.parse_args()
    bm25 = BM25()

    with store.open_index(a.index) as ix:
        n = ix.num_docs
        lengths = sorted(ix.doc_lengths, key=ix.doc_lengths.get)
        short, mid, long_ = lengths[0], lengths[len(lengths) // 2], lengths[-1]
        terms = sorted(ix, key=lambda t: (ix.df(t), t))
        rare = next(t for t in terms if ix.df(t) >= 1)
        common = terms[-1]

        print(f"corpus: {ix!r}, avg doc length {ix.avg_doc_length:.1f}\n")

        s_rare = bm25.score(rare, Posting(mid, 1), ix)
        s_common = bm25.score(common, Posting(mid, 1), ix)
        print("1) rare term beats common term (same doc, tf=1)")
        print(f"   {rare!r} (df={ix.df(rare)}): {s_rare:.3f}   "
              f"{common!r} (df={ix.df(common)}): {s_common:.3f}   ->  {s_rare > s_common}\n")

        print(f"2) saturation for {rare!r} in a median-length doc (k1={bm25.k1})")
        sc = {tf: bm25.score(rare, Posting(mid, tf), ix) for tf in (1, 2, 10, 20, 21)}
        for tf, v in sc.items():
            print(f"   tf={tf:<2} score={v:.3f}")
        print(f"   gain 1st->2nd occurrence: {sc[2] - sc[1]:.3f}   "
              f"gain 20th->21st: {sc[21] - sc[20]:.4f}\n")

        s_short = bm25.score(rare, Posting(short, 1), ix)
        s_long = bm25.score(rare, Posting(long_, 1), ix)
        print("3) short doc with one match beats a very long doc with one match")
        print(f"   short (len {ix.doc_length(short)}): {s_short:.3f}   "
              f"long (len {ix.doc_length(long_)}): {s_long:.3f}   ->  {s_short > s_long}")
        assert n > 0


if __name__ == "__main__":
    main()
