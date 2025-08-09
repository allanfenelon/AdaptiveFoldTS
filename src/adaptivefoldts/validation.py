from typing import List, Tuple
import pandas as pd

def rolling_window_split(
    series: pd.Series,
    window_size: int,
    test_size: int,
    step_size: int,
    max_folds: int | None = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Gera índices para janelas rolling (janela deslizante) de treino e teste.

    Returns lista de tuplas: (train_start, train_end, test_start, test_end)
    """
    folds = []
    n = len(series)
    start = 0

    while True:
        train_start = start
        train_end = train_start + window_size
        test_start = train_end
        test_end = test_start + test_size

        if test_end > n:
            break

        folds.append((train_start, train_end, test_start, test_end))

        start += step_size
        if max_folds is not None and len(folds) >= max_folds:
            break

    return folds