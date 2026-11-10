"""
prof_split.py: ділить час профілю cProfile на «Python» і «native» (лаба 8).

    python prof_split.py docs\\search_before.prof docs\\page_before.prof docs\\build_before.prof

УВАГА: час, який регулярний вираз витрачає всередині `for m in re.finditer(...)`, cProfile
окремо не бачить і записує на функцію, де стоїть цикл (тобто на «Python»). Тому для
make_snippet частка native тут занижена: це нижня межа.

Правило просте. cProfile записує вбудовані функції й методи C (re, str.casefold,
unicodedata.normalize, math.log, dict.get ...) з порожнім іменем файлу «~». Їхній власний
час це native. Решта (мій код і бібліотеки на Python) це Python.
"""

from __future__ import annotations

import pstats
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("використання: python prof_split.py файл.prof [ще.prof ...]")
    for path in sys.argv[1:]:
        stats = pstats.Stats(path)
        py = native = 0.0
        native_rows: list[tuple[float, str]] = []
        python_rows: list[tuple[float, str]] = []
        # stats.stats: (файл, рядок, ім'я) -> (cc, nc, власний час, сумарний час, хто викликав)
        for (filename, line, name), (_cc, _nc, tt, _ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
            if filename == "~":
                native += tt
                native_rows.append((tt, name))
            else:
                py += tt
                python_rows.append(
                    (tt, f"{filename.replace(chr(92), '/').split('/')[-1]}:{line}({name})")
                )
        total = py + native
        print(f"\n=== {path} ===")
        print(
            f"Python: {py / total * 100:5.1f}%   native (C): {native / total * 100:5.1f}%"
            f"   усього {total:.2f} с"
        )
        print(
            "  найдорожчі native:",
            ", ".join(
                f"{n} {t / total * 100:.0f}%" for t, n in sorted(native_rows, reverse=True)[:4]
            ),
        )
        print(
            "  найдорожчі Python:",
            ", ".join(
                f"{n} {t / total * 100:.0f}%" for t, n in sorted(python_rows, reverse=True)[:4]
            ),
        )


if __name__ == "__main__":
    main()
