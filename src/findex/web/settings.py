"""Налаштування сервісу. Усі значення беруться зі змінних оточення (так вимагає Docker і Render)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфіг зі змінних оточення: INDEX_PATH, CORPUS_PATH, LOG_LEVEL і так далі.

    Імена змінних збігаються з назвами полів (великими літерами), префікса немає."""

    # зайві змінні оточення (PATH, HOME тощо) просто ігнорую
    model_config = SettingsConfigDict(extra="ignore")

    # де лежить файл індексу, який сервіс завантажує один раз при старті
    index_path: Path = Path("demo/index.bin")
    # звідки брати тексти для уривків і перегляду документа: папка з .txt або файл .jsonl
    # (якщо None або такого шляху немає, уривків просто не буде)
    corpus_path: Path | None = Path("demo/corpus")
    # рівень логів: DEBUG, INFO, WARNING
    log_level: str = "INFO"
    # скільки результатів на сторінці за замовчуванням і щонайбільше
    default_page_size: int = 10
    max_page_size: int = 50
    # ширина уривка в символах з кожного боку від слова запиту
    snippet_width: int = 90
    # підпис унизу сторінки про походження текстів
    corpus_note: str = "Тексти: Українська Вікіпедія, ліцензія CC BY-SA 4.0."
