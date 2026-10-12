import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TITLES = [
    "Київ", "Львів", "Одеса", "Харків", "Дніпро", "Карпати",
    "Тарас Шевченко", "Леся Українка", "Іван Франко", "Григорій Сковорода",
    "Python", "Алгоритм", "Структура даних", "Пошукова система", "Інтернет",
    "Штучний інтелект", "Машинне навчання", "Програмування", "Операційна система",
    "Фізика", "Хімія", "Біологія", "Математика", "Астрономія", "Сонячна система",
    "Історія України", "Київська Русь", "Козацтво", "Українська мова",
    "Музика", "Кіно", "Література", "Живопис", "Архітектура", "Театр",
    "Футбол", "Олімпійські ігри", "Шахи", "Кава", "Чай", "Борщ",
]

PAUSE = 2.0      # секунд між запитами, щоб Вікіпедія не блокувала
RETRIES = 5

out = Path("data/wiki")
out.mkdir(parents=True, exist_ok=True)


def fetch(title):
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "format": "json", "redirects": 1,
    })
    url = "https://uk.wikipedia.org/w/api.php?" + params
    req = urllib.request.Request(url, headers={"User-Agent": "findex-lab/1.0 (student project)"})
    for attempt in range(1, RETRIES + 1):
        try:
            data = json.load(urllib.request.urlopen(req, timeout=30))
            page = next(iter(data["query"]["pages"].values()))
            return page.get("extract", "")
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < RETRIES:
                wait = 10 * attempt
                print(f"   429, чекаю {wait} с (спроба {attempt})...")
                time.sleep(wait)
                continue
            raise
    return ""


for title in TITLES:
    name = title.replace(" ", "_").replace("(", "").replace(")", "")
    path = out / (name + ".txt")
    if path.exists() and path.stat().st_size > 0:
        print("вже є", title)
        continue
    try:
        text = fetch(title)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        print("помилка", title, exc)
        continue
    if text:
        path.write_text(text, encoding="utf-8")
        print("ok", title, len(text))
    time.sleep(PAUSE)