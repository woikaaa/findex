"""
три запити для вимірювань у лабі 8: частотне слово, рідкісне слово і запит із трьох слів.

Запити вибираються з самого індексу за правилом, тому на будь-якому корпусі вони
однакові за змістом, а не «які я придумала». Їх використовують і profile_run.py,
і тести pytest-benchmark, і таблиця в README.
"""

from __future__ import annotations

from findex.model import Index


def pick_cases(index: Index) -> list[tuple[str, str]]:
    """Повертає пари (назва випадку, текст запиту).

    - frequent: слово, що є в найбільшій кількості документів (найважчий випадок);
    - rare: слово з найменшою кількістю документів (найлегший випадок);
    - three: три слова середньої частоти, з'єднані через OR (багато документів,
      у кожного кілька слів: ось де видно вартість підрахунку балів).
    """
    # слова від найчастіших до найрідших; при рівності беру за алфавітом
    by_df = sorted(index.postings, key=lambda t: (-index.df(t), t))
    if not by_df:
        raise ValueError("index is empty")
    frequent = by_df[0]
    rare = by_df[-1]
    n = len(by_df)
    # три слова з різних місць списку (10-те, 30-те, 50-те), але не далі за кінець
    picks = [by_df[min(n - 1, r)] for r in (9, 29, 49)]
    three = " OR ".join(dict.fromkeys(picks))  # dict.fromkeys прибирає повтори, порядок лишається
    return [("frequent", frequent), ("rare", rare), ("three", three)]
