# AdaptiveFoldTS

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
from adaptivefoldts import AdaptiveFoldTS

# Suponha que você tenha uma série temporal
series = pd.Series(np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100))

# Inicialize o validador adaptativo
cv = AdaptiveFoldTS(series=series, window_size=20, test_size=5, verbose=True)

# Suponha que você tenha um modelo sklearn-like
from sklearn.linear_model import LinearRegression
models = {
    "RegressaoLinear": LinearRegression()
}

# Avalie os modelos
results = cv.evaluate_models(models)

# Imprima o ranking
print(cv.rank_models())

```

## Funcionalidades Futuras

- Implementação de grid search e otimização automatizada de hiperparâmetros integrada ao processo adaptativo.
- Suporte a mais métricas customizáveis, incluindo métricas específicas para séries temporais (ex: MASE, SMAPE).
- Inclusão de outras estratégias de janelamento adaptativo, como janelas expansivas e segmentações baseadas em eventos.
- Visualização interativa dos folds, métricas e prioridades ao longo do processo de validação.
- Paralelização do processo de avaliação para acelerar experimentos em grandes conjuntos de dados e múltiplos modelos.
