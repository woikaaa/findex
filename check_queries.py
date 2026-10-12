import json
from pathlib import Path

from findex import store
from findex.ranking import Searcher

with store.open_index("index.bin") as ix:
    s = Searcher(ix)
    for item in json.loads(Path("queries.json").read_text(encoding="utf-8")):
        q = " OR ".join(item["query"].split())
        stems = [Path(ix.doc_meta[r.doc_id].path).stem for r in s.search(q, 5)]
        print(item["query"])
        print("   топ-5:        ", stems)
        print("   мої relevant: ", item["relevant"])