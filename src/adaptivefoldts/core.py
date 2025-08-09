from typing import Optional, Union, Dict, Any
import pandas as pd
import numpy as np

from .validation import rolling_window_split
from .heap import AdaptiveHeap


class AdaptiveFoldTS:
    def __init__(
        self,
        series: pd.Series,
        window_size: Optional[int] = None,
        test_size: Union[int, float] = 0.2,
        max_folds: Optional[int] = None,
        step_size: int = 1,
        strategy: str = "rolling",
        heap_criterion: str = "error",
        metric: str = "mae",
        min_train_size: Optional[int] = None,
        max_train_size: Optional[int] = None,
        verbose: bool = False,
        random_seed: Optional[int] = None,
    ):
        """
        Inicializa o validador cruzado adaptativo para séries temporais.

        Parâmetros
        ----------
        series : pd.Series
            Série temporal a ser validada. Deve ser indexada temporalmente, mas será resetado o índice internamente.
        window_size : int, opcional
            Tamanho fixo da janela de treino. Se None, pode usar min_train_size e max_train_size para controle dinâmico.
        test_size : int ou float, padrão 0.2
            Tamanho da janela de teste. Se for float, representa a proporção da série (ex: 0.2 = 20% dos dados).
        max_folds : int, opcional
            Número máximo de folds a gerar na validação cruzada. Se None, gera o máximo possível baseado nos parâmetros.
        step_size : int, padrão 1
            Passo para avançar a janela na geração dos folds sequenciais.
        strategy : str, padrão "rolling"
            Estratégia de divisão da série temporal. Atualmente suportado apenas "rolling".
        metric : str, padrão "mae"
            Métrica usada para avaliar a qualidade das previsões e fazer a priorização adaptativa do Heap. Opções comuns: "mae", "mse", "rmse".
        min_train_size : int, opcional
            Tamanho mínimo da janela de treino para estratégias adaptativas.
        max_train_size : int, opcional
            Tamanho máximo da janela de treino para estratégias adaptativas.
        verbose : bool, padrão False
            Se True, habilita mensagens detalhadas durante a execução para acompanhamento.
        random_seed : int, opcional
            Semente para controle de aleatoriedade e reprodutibilidade dos resultados.

        Raises
        ------
        TypeError
            Se `series` não for uma instância de `pd.Series`.
        ValueError
            Se parâmetros numéricos forem inválidos (exemplo: tamanhos negativos, proporções fora do intervalo).
        NotImplementedError
            Se for selecionada uma estratégia de divisão que não seja suportada (exemplo: diferente de "rolling").
        """
        self.series = series.reset_index(drop=True)
        self.window_size = window_size
        self.test_size = test_size
        self.max_folds = max_folds
        self.step_size = step_size
        self.strategy = strategy
        self.heap_criterion = heap_criterion
        self.metric = metric.lower()
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.verbose = verbose
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        self._validate_parameters()
        self.results = {}

    def _validate_parameters(self):
        if not isinstance(self.series, pd.Series):
            raise TypeError("series deve ser um pandas.Series")

        if self.window_size is not None and self.window_size <= 0:
            raise ValueError("window_size deve ser positivo")

        if isinstance(self.test_size, float):
            if not (0 < self.test_size < 1):
                raise ValueError("test_size float deve estar entre 0 e 1")
            self.test_size = int(len(self.series) * self.test_size)
        elif isinstance(self.test_size, int):
            if self.test_size <= 0:
                raise ValueError("test_size int deve ser positivo")
        else:
            raise TypeError("test_size deve ser int ou float")

        if self.max_folds is not None and self.max_folds <= 0:
            raise ValueError("max_folds deve ser positivo ou None")

        if self.step_size <= 0:
            raise ValueError("step_size deve ser positivo")

        if self.strategy != "rolling":
            raise NotImplementedError(f"Estratégia '{self.strategy}' não implementada. Use rolling.")

    def _generate_folds(self):
        match self.strategy:
            case 'rolling':
                return rolling_window_split(
                    self.series,
                    window_size=self.window_size or self.min_train_size,
                    test_size=self.test_size,
                    step_size=self.step_size,
                    max_folds=self.max_folds
                )
            case _:
                raise NotImplementedError(f"Estratégia '{self.strategy}' não implementada.")

    def _calculate_metric(self, y_true, y_pred):
        if self.metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        elif self.metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif self.metric == "rmse":
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        else:
            raise ValueError(f"Métrica '{self.metric}' não implementada.")

    def evaluate_models(self, models: Dict[str, Any], max_iterations_per_fold: int = 3) -> Dict[str, float]:
        folds = self._generate_folds()
        results = {name: [] for name in models.keys()}
        fold_iter_counts = {fold_id: 0 for fold_id in range(len(folds))}
        heap = AdaptiveHeap()
        for fold_id, fold in enumerate(folds):
            heap.push(0.0, (fold_id, fold))

        while not heap.is_empty():
            priority, (fold_id, (train_start, train_end, test_start, test_end)) = heap.pop()
            if self.verbose:
                print(f"Processando fold {fold_id + 1}/{len(folds)} com prioridade {priority:.4f}")

            fold_iter_counts[fold_id] += 1
            if fold_iter_counts[fold_id] > max_iterations_per_fold:
                if self.verbose:
                    print(f"Fold {fold_id + 1} atingiu limite máximo de iterações. Pulando.")
                continue

            train_y = self.series.iloc[train_start:train_end].values
            test_y = self.series.iloc[test_start:test_end].values

            fold_errors = []

            for name, model in models.items():
                X_train = np.arange(len(train_y)).reshape(-1, 1)
                X_test = np.arange(len(train_y), len(train_y) + len(test_y)).reshape(-1, 1)

                model.fit(X_train, train_y)
                preds = model.predict(X_test)

                error = self._calculate_metric(test_y, preds)
                results[name].append(error)
                fold_errors.append(error)

            new_priority = float(np.mean(fold_errors))

            heap.push(-new_priority, (fold_id, (train_start, train_end, test_start, test_end)))

        aggregated = {name: float(np.mean(errors)) for name, errors in results.items()}
        self.results = aggregated
        return aggregated

    def rank_models(self, ascending: bool = True) -> list:
        if not self.results:
            raise RuntimeError("Avalie os modelos antes de ranquear.")

        return sorted(self.results.items(), key=lambda x: x[1], reverse=not ascending)
