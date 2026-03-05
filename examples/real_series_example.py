"""Exemplo de uso do AdaptiveFoldTS com uma serie temporal realista."""

import numpy as np
import pandas as pd
from adaptivefoldts import AdaptiveFoldTS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def create_realistic_series(n_days=365):
    """Cria uma serie temporal simulando vendas diarias com tendencia.

    Args:
        n_days (int): Numero de dias a simular.

    Returns:
        pd.Series: Serie temporal simulada.
    """
    np.random.seed(42)
    time_idx = np.arange(n_days)

    trend = time_idx * 0.1
    seasonality = np.sin(2 * np.pi * time_idx / 7) * 5
    noise = np.random.normal(0, 1.5, n_days)

    sales = 50 + trend + seasonality + noise

    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

    return pd.Series(sales, index=dates, name="sales")


def main():
    """Funcao principal para rodar o exemplo."""
    series = create_realistic_series(n_days=200)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=50, random_state=42
        ),
    }

    af = AdaptiveFoldTS(
        series=series,
        window_size=30,
        test_size=7,
        step_size=7,
        strategy="rolling",
        metric="mae",
        verbose=True,
    )

    results = af.evaluate_models(models=models, max_iterations_per_fold=3)

    print("\nResultados de MAE agregados por modelo:")
    for model_name, score in results.items():
        print("{}: {:.4f}".format(model_name, score))  # noqa: UP032

    ranked = af.rank_models(ascending=True)
    best_model = ranked[0][0]
    best_score = ranked[0][1]

    print(
        "\nMelhor modelo: {} com MAE de {:.4f}".format(
            best_model, best_score
        )
    )  # noqa: UP032


if __name__ == "__main__":
    main()
