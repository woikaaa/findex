"""
завантажує статті української Вікіпедії у папку data/wiki, кожну в окремий .txt файл.

Файли, які вже завантажені, пропускаються, тож скрипт можна запускати
повторно, якщо щось не вдалося з першого разу.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# назви статей, які ми качаємо
TITLES = [
    "Київ",
    "Львів",
    "Одеса",
    "Харків",
    "Дніпро",
    "Карпати",
    "Тарас Шевченко",
    "Леся Українка",
    "Іван Франко",
    "Григорій Сковорода",
    "Python",
    "Алгоритм",
    "Структура даних",
    "Пошукова система",
    "Інтернет",
    "Штучний інтелект",
    "Машинне навчання",
    "Програмування",
    "Операційна система",
    "Фізика",
    "Хімія",
    "Біологія",
    "Математика",
    "Астрономія",
    "Сонячна система",
    "Історія України",
    "Київська Русь",
    "Козацтво",
    "Українська мова",
    "Музика",
    "Кіно",
    "Література",
    "Живопис",
    "Архітектура",
    "Театр",
    "Футбол",
    "Олімпійські ігри",
    "Шахи",
    "Кава",
    "Чай",
    "Борщ",
]

PAUSE = 2.0  # секунд між запитами, щоб Вікіпедія не блокувала
RETRIES = 5  # скільки разів пробуємо, якщо сервер каже "забагато запитів"

# папка для результату (створюємо, якщо її ще немає)
out = Path("data/wiki")
out.mkdir(parents=True, exist_ok=True)


def fetch(title):
    """Завантажує текст однієї статті. Повертає рядок (може бути порожнім)."""
    # параметри запиту до API Вікіпедії: хочемо чистий текст без розмітки
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "redirects": 1,
        }
    )
    url = "https://uk.wikipedia.org/w/api.php?" + params
    # User-Agent просить Вікіпедія: треба представитися
    req = urllib.request.Request(url, headers={"User-Agent": "findex-lab/1.0 (student project)"})
    # пробую кілька разів
    for attempt in range(1, RETRIES + 1):
        try:
            data = json.load(urllib.request.urlopen(req, timeout=30))
            # у відповіді одна сторінка, беремо її
            page = next(iter(data["query"]["pages"].values()))
            return page.get("extract", "")
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < RETRIES:
                # 429 = забагато запитів, чекаємо довше з кожною спробою
                wait = 10 * attempt
                print(f"   429, чекаю {wait} с (спроба {attempt})...")
                time.sleep(wait)
                continue
            # це інша помилка, її не ховаю
            raise
    return ""


# проходжу по списку статей
for title in TITLES:
    # робимо з назви безпечне ім'я файлу
    name = title.replace(" ", "_").replace("(", "").replace(")", "")
    path = out / (name + ".txt")
    # якщо файл уже є і не порожній, не качаємо вдруге
    if path.exists() and path.stat().st_size > 0:
        print("вже є", title)
        continue
    try:
        # качаю текст статті
        text = fetch(title)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        print("помилка", title, exc)
        continue
    if text:
        # зберігаю статтю в файл
        path.write_text(text, encoding="utf-8")
        print("ok", title, len(text))
    time.sleep(PAUSE)  # пауза між статтями
