"""маленькі допоміжні декоратори."""

from __future__ import annotations

import functools
import logging
import time

# логер, у який я пишу час роботи
log = logging.getLogger("findex.timing")


def timed(fn):
    """Записує в лог, скільки часу працювала функція `fn`.

    Запис `@timed` над `def f` означає те саме, що `f = timed(f)`.
    `functools.wraps` копіює ім'я та опис функції на обгортку, тому
    f.__name__ лишається 'f', а не 'wrapper'."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # запам'ятовую час перед запуском
        t0 = time.perf_counter()  # час старту
        try:
            # запускаю саму функцію з усіма її аргументами
            return fn(*args, **kwargs)
        finally:
            # finally спрацює навіть якщо функція впала з помилкою
            log.info("%s took %.2f ms", fn.__qualname__, (time.perf_counter() - t0) * 1000)
    # повертаю обгортку замість звичайної функції
    return wrapper
