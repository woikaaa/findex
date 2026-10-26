"""
швидка перевірка: що пошук знаходить у топ-5 для кожного мого запиту з queries.json
і які документи я позначила як релевантні (щоб побачити різницю очима).
"""

import json
from pathlib import Path

from findex import store
from findex.ranking import Searcher

# відкриваємо індекс; with сам закриє його в кінці
with store.open_index("index.bin") as ix:
    # пошуковик зі стандартним BM25
    s = Searcher(ix)
    # queries.json: список запитів і документів, які я вважаю правильними
    for item in json.loads(Path("queries.json").read_text(encoding="utf-8")):
        # слова запиту з'єднуємо через OR, щоб було що ранжувати
        q = " OR ".join(item["query"].split())
        # беремо імена файлів п'яти найкращих результатів
        stems = [Path(ix.doc_meta[r.doc_id].path).stem for r in s.search(q, 5)]
        # показую запит, а потім мої очікування
        print(item["query"])
        print("   топ-5:        ", stems)
        print("   мої relevant: ", item["relevant"])
