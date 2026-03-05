import unittest

import numpy as np
import pandas as pd
from adaptivefoldts.core import AdaptiveFoldTS
from adaptivefoldts.heap import AdaptiveHeap
from adaptivefoldts.validation import rolling_window_split


class TestAdaptiveHeap(unittest.TestCase):
    """Testes para o AdaptiveHeap."""

    def test_adaptive_heap(self):
        """Testa o comportamento basico do heap ajustado."""
        heap = AdaptiveHeap()
        self.assertTrue(heap.is_empty())
        self.assertEqual(len(heap), 0)

        heap.push(1.5, "item1")
        heap.push(0.5, "item2")
        heap.push(2.0, "item3")

        self.assertFalse(heap.is_empty())
        self.assertEqual(len(heap), 3)

        self.assertEqual(heap.peek(), (0.5, "item2"))

        priority, item = heap.pop()
        self.assertEqual(priority, 0.5)
        self.assertEqual(item, "item2")
        self.assertEqual(len(heap), 2)


class TestValidation(unittest.TestCase):
    """Testes para as funcoes de validacao."""

    def test_rolling_window_split(self):
        """Testa o rolling_window_split."""
        series = pd.Series(range(10))
        folds = rolling_window_split(
            series=series,
            window_size=3,
            test_size=2,
            step_size=1,
            max_folds=None,
        )

        expected_folds = [
            (0, 3, 3, 5),
            (1, 4, 4, 6),
            (2, 5, 5, 7),
            (3, 6, 6, 8),
            (4, 7, 7, 9),
            (5, 8, 8, 10),
        ]
        self.assertEqual(folds, expected_folds)


class TestAdaptiveFoldTS(unittest.TestCase):
    """Testes para o AdaptiveFoldTS."""

    def test_adaptivefoldts_initialization(self):
        """Testa a inicializacao da classe com dados corretos."""
        series = pd.Series([1, 2, 3, 4, 5])
        ts = AdaptiveFoldTS(
            series=series, window_size=2, test_size=1, step_size=1
        )
        self.assertEqual(ts.window_size, 2)
        self.assertEqual(ts.test_size, 1)
        self.assertEqual(ts.step_size, 1)

    def test_adaptivefoldts_invalid_initialization(self):
        """Testa inicializacoes invalidas para devidas excecoes."""
        series = pd.Series([1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            AdaptiveFoldTS(series=series, window_size=-1)

        with self.assertRaises((TypeError, AttributeError)):
            AdaptiveFoldTS(series=[1, 2, 3])

        with self.assertRaises(ValueError):
            AdaptiveFoldTS(series=series, test_size=1.5)


class MockModel:
    """Modelo mock para simular previsao."""

    def __init__(self, constant_pred):
        """Inicializa mock model."""
        self.constant_pred = constant_pred

    def fit(self, x, y):
        """Simula o metodo fit do modelo."""
        pass

    def predict(self, x):
        """Simula o metodo predict do modelo."""
        return np.full(x.shape[0], self.constant_pred)


class TestAdaptiveFoldTSEvaluate(unittest.TestCase):
    """Testes para o metodo evaluate_models do AdaptiveFoldTS."""

    def test_adaptivefoldts_evaluate_models(self):
        """Testa se o modelo avalia corretamente."""
        series = pd.Series([10, 20, 30, 40, 50, 60])
        ts = AdaptiveFoldTS(
            series=series,
            window_size=2,
            test_size=1,
            step_size=1,
            metric="mae",
        )
        models = {
            "model_0": MockModel(constant_pred=0),
            "model_20": MockModel(constant_pred=20),
        }

        results = ts.evaluate_models(models)

        self.assertIn("model_0", results)
        self.assertIn("model_20", results)

        self.assertAlmostEqual(results["model_0"], 45.0)
        self.assertAlmostEqual(results["model_20"], 25.0)

        ranked = ts.rank_models(ascending=True)
        self.assertEqual(ranked[0][0], "model_20")
        self.assertEqual(ranked[1][0], "model_0")

    def test_calculate_metric(self):
        """Testa o calculo das metricas de erro."""
        series = pd.Series([1, 2, 3])
        ts = AdaptiveFoldTS(series=series, test_size=1)

        y_true = np.array([2, 4, 6])
        y_pred = np.array([1, 5, 6])

        ts.metric = "mae"
        self.assertAlmostEqual(ts._calculate_metric(y_true, y_pred), 2 / 3)

        ts.metric = "mse"
        self.assertAlmostEqual(ts._calculate_metric(y_true, y_pred), 2 / 3)

        ts.metric = "rmse"
        self.assertAlmostEqual(
            ts._calculate_metric(y_true, y_pred), np.sqrt(2 / 3)
        )


if __name__ == "__main__":
    unittest.main()
