# AdaptiveFoldTS

[![PyPI version](https://badge.fury.io/py/adaptivefoldts.svg)](https://badge.fury.io/py/adaptivefoldts)
[![PyPI downloads](https://img.shields.io/pypi/dm/adaptivefoldts)](https://pypistats.org/packages/adaptivefoldts)


**Validação Cruzada Adaptativa para Séries Temporais com Priorização Inteligente de Folds**

AdaptiveFoldTS é uma biblioteca Python inovadora projetada para realizar validação cruzada em séries temporais utilizando uma abordagem adaptativa e priorizada. Utilizando estruturas de heap, a biblioteca organiza janelas de treino e teste com base em critérios dinâmicos — como variância, erro preditivo ou outras métricas customizáveis — permitindo que o processo de validação foque nos segmentos mais críticos ou desafiadores da série temporal.

Durante o processo, as prioridades dos folds são atualizadas iterativamente conforme as métricas dos modelos são calculadas. Isso significa que folds com maiores erros ou maior variabilidade recebem maior atenção, sendo reavaliados, o que possibilita uma melhor identificação do modelo que entrega o melhor desempenho global.

---

## Principais características

- Priorização adaptativa das janelas de validação via heap para maximizar eficiência e insight.  
- Atualização dinâmica das prioridades baseada nas métricas de erro dos modelos, para foco em folds mais relevantes.  
- Suporte a múltiplas estratégias de janelamento para treino e teste (rolling window, expanding window, etc).  
- Integração simples com modelos sklearn-like ou customizados.  
- Avaliação simultânea de múltiplos modelos com métricas configuráveis e ranking automático.  
- Ferramentas para análise detalhada do desempenho por fold e geração de relatórios.

---

## Instalação

```bash
pip install adaptivefoldts
```

## Exemplo básico de uso
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

from adaptivefoldts import AdaptiveFoldTS

np.random.seed(42)
series = pd.Series(np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.5, 100))

models = {
    "linear_regression": LinearRegression(),
    "dummy_mean": DummyRegressor(strategy="mean"),
}

af = AdaptiveFoldTS(
    series=series,
    window_size=30,
    test_size=0.2,
    step_size=5,
    strategy="rolling",
    metric="mse",
    verbose=True,
    max_folds=None,
)

results = af.evaluate_models(models=models, max_iterations_per_fold=1)

print("Resultados agregados por modelo:")
for model_name, score in results.items():
    print(f"{model_name}: {score:.4f}")
```

## Funcionalidades Futuras

- Implementação de grid search e otimização automatizada de hiperparâmetros integrada ao processo adaptativo.
- Suporte a mais métricas customizáveis, incluindo métricas específicas para séries temporais (ex: MASE, SMAPE).
- Inclusão de outras estratégias de janelamento adaptativo, como janelas expansivas e segmentações baseadas em eventos.
- Visualização interativa dos folds, métricas e prioridades ao longo do processo de validação.
- Paralelização do processo de avaliação para acelerar experimentos em grandes conjuntos de dados e múltiplos modelos.
