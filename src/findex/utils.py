"""utils.py — small decorators."""

from __future__ import annotations

import functools
import logging
import time

log = logging.getLogger("findex.timing")


def timed(fn):
    """Log how long `fn` took (INFO on the 'findex.timing' logger).

    `@timed` above `def f` is exactly `f = timed(f)`. `functools.wraps` copies
    __name__ / __doc__ onto the wrapper, so f.__name__ is still 'f'."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.info("%s took %.2f ms", fn.__qualname__, (time.perf_counter() - t0) * 1000)
    return wrapper
