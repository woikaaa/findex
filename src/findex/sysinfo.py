"""відомості про машину й процес, потрібні вимірам (лаби 5 і 6)."""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def peak_rss_mb() -> float:
    """Пік пам'яті цього процесу (RSS) у мегабайтах."""
    if sys.platform == "win32":
        try:
            import psutil  # на Windows немає модуля resource, тому беру psutil
        except ImportError:
            return float("nan")  # psutil немає (наприклад у 3.13t без dev-залежностей)
        return psutil.Process().memory_info().peak_wset / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # на Linux ru_maxrss у кілобайтах, на macOS у байтах
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


def cpu_model() -> str:
    """Назва процесора (щоб записати її в README)."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gil_status() -> str:
    """Чи ввімкнений GIL у цьому інтерпретаторі (3.13t без GIL відповідає False)."""
    check = getattr(sys, "_is_gil_enabled", None)
    return "GIL увімкнено" if check is None or check() else "GIL ВИМКНЕНО"
