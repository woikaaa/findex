"""
evaluate.py — precision@5 of TF-IDF vs BM25 on hand-labelled queries.

    python -m findex.evaluate index.bin queries.json

queries.json: [{"query": "...", "relevant": ["doc_stem", ...]}, ...]
(doc_stem = file name without .txt).  Multi-word queries are run as OR
(a OR b OR c) so that ranking has something to rank; with implicit AND most
queries would match only one or two documents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findex import store
from findex.ranking import Searcher
from findex.scoring import BM25, TfIdf


def p_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    return sum(1 for d in retrieved[:k] if d in relevant) / k


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("index", type=Path)
    p.add_argument("queries", type=Path)
    p.add_argument("-k", type=int, default=5)
    a = p.parse_args()
    labelled = json.loads(a.queries.read_text(encoding="utf-8"))

    with store.open_index(a.index) as ix:
        scorers = {"TF-IDF": Searcher(ix, TfIdf()), "BM25": Searcher(ix, BM25())}
        totals = dict.fromkeys(scorers, 0.0)
        print(f"| Query | relevant | P@{a.k} TF-IDF | P@{a.k} BM25 |")
        print("| ----- | -------- | ---------- | -------- |")
        for item in labelled:
            q = " OR ".join(item["query"].split())
            rel = set(item["relevant"])
            row = []
            for name, s in scorers.items():
                stems = [Path(ix.doc_meta[r.doc_id].path).stem for r in s.search(q, a.k)]
                v = p_at_k(stems, rel, a.k)
                totals[name] += v
                row.append(f"{v:.2f}")
            print(f"| {item['query']} | {len(rel)} | {row[0]} | {row[1]} |")
        n = len(labelled)
        print(f"| **mean** | | **{totals['TF-IDF'] / n:.2f}** | **{totals['BM25'] / n:.2f}** |")


if __name__ == "__main__":
    main()
