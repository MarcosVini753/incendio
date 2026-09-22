"""FuzzyKNN corrigido, zoo de modelos e metricas apropriadas a eventos raros."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# `force_all_finite` virou `ensure_all_finite` no scikit-learn 1.6 e foi removido
# na 1.8. Resolve o nome uma vez, para o modulo rodar nas duas familias de versao.
_FINITO = (
    {"ensure_all_finite": True}
    if "ensure_all_finite" in __import__("inspect").signature(check_array).parameters
    else {"force_all_finite": True}
)


class FuzzyKNN(ClassifierMixin, BaseEstimator):
    """k-NN Fuzzy de Keller (1985), com pertinencias de classe normalizadas.

    Correcoes em relacao a implementacao original do projeto:

    * `predict` devolve apenas os rotulos, como manda o contrato do
      scikit-learn. A versao original devolvia `[rotulos, pertinencias]`, o que
      quebra `cross_val_score`, `GridSearchCV`, `Pipeline` e qualquer metrica
      chamada diretamente. As pertinencias agora saem por `predict_proba`.
    * `fit` devolve `self` e registra `classes_` e `n_features_in_`, sem os
      quais `clone`, `check_is_fitted` e serializacao nao funcionam direito.
    * As pertinencias sao normalizadas dentro do estimador (soma 1), como na
      formulacao de Keller. Na versao original a normalizacao era feita a mao,
      fora do modelo, so no trecho final do mapa.
    * A busca dos vizinhos usa `argpartition` (selecao parcial, O(n)) em vez de
      `argsort` (O(n log n)) e processa o teste em blocos vetorizados em vez de
      um laco Python por amostra.

    Parametros
    ----------
    k : int
        Numero de vizinhos.
    m : float
        Expoente de difusao (o `q` do codigo original). m > 1. Peso do vizinho
        j = 1 / d_j^(2/(m-1)).
    bloco : int
        Quantas amostras de teste processar por vez (controla o pico de memoria:
        o custo e bloco x n_treino floats).
    """

    def __init__(self, k: int = 21, m: float = 2.0, bloco: int = 2048):
        self.k = k
        self.m = m
        self.bloco = bloco

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64, **_FINITO)
        if self.m <= 1:
            raise ValueError(f"m deve ser > 1, recebido {self.m}")
        if self.k < 1:
            raise ValueError(f"k deve ser >= 1, recebido {self.k}")
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        # indice da classe de cada amostra de treino
        self.y_idx_ = np.searchsorted(self.classes_, y)
        self._k_efetivo = min(self.k, X.shape[0])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "X_")
        X = check_array(X, dtype=np.float64, **_FINITO)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"esperava {self.n_features_in_} colunas, recebeu {X.shape[1]}"
            )
        n_cls = len(self.classes_)
        k = self._k_efetivo
        expoente = 2.0 / (self.m - 1.0)
        saida = np.empty((X.shape[0], n_cls), dtype=np.float64)

        # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2
        norma_treino = np.einsum("ij,ij->i", self.X_, self.X_)

        for ini in range(0, X.shape[0], self.bloco):
            lote = X[ini : ini + self.bloco]
            d2 = (
                np.einsum("ij,ij->i", lote, lote)[:, None]
                - 2.0 * lote @ self.X_.T
                + norma_treino[None, :]
            )
            np.maximum(d2, 0.0, out=d2)
            viz = np.argpartition(d2, k - 1, axis=1)[:, :k]
            d_viz = np.sqrt(np.take_along_axis(d2, viz, axis=1))
            peso = 1.0 / (d_viz**expoente + 1e-12)
            classe_viz = self.y_idx_[viz]

            num = np.zeros((lote.shape[0], n_cls))
            for c in range(n_cls):
                num[:, c] = (peso * (classe_viz == c)).sum(axis=1)
            saida[ini : ini + self.bloco] = num / num.sum(axis=1, keepdims=True)
        return saida

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------- zoo de modelos


def zoo(semente: int = 42, k_fuzzy: int = 29) -> dict[str, object]:
    """Modelos comparaveis para o problema. Todos expoem `predict_proba`.

    HistGradientBoosting lida com NaN nativamente; os demais recebem imputacao
    e padronizacao via Pipeline.
    """
    from sklearn.impute import SimpleImputer

    def com_preparo(estimador):
        return Pipeline(
            [
                ("imputar", SimpleImputer(strategy="median")),
                ("escalar", StandardScaler()),
                ("modelo", estimador),
            ]
        )

    return {
        "GradBoost": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=semente,
        ),
        "RandomForest": Pipeline(
            [
                ("imputar", SimpleImputer(strategy="median")),
                (
                    "modelo",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                        random_state=semente,
                    ),
                ),
            ]
        ),
        "RegLogistica": com_preparo(
            LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        f"FuzzyKNN_k{k_fuzzy}": com_preparo(FuzzyKNN(k=k_fuzzy, m=2.0)),
    }


# --------------------------------------------------------------------- metricas


def taxa_deteccao_no_topo(y, p, fracao: float) -> float:
    """Fracao dos incendios capturada nas `fracao` celulas de maior risco.

    Esta e a metrica que importa na operacao: se a brigada consegue vigiar 5%
    do territorio por dia, quantos focos daquele dia caem dentro dos 5% que o
    modelo apontou como mais perigosos.
    """
    y = np.asarray(y)
    n_topo = max(int(round(len(p) * fracao)), 1)
    corte = np.argsort(p)[::-1][:n_topo]
    total = y.sum()
    return float(y[corte].sum() / total) if total else float("nan")


def avaliar(y, p) -> dict[str, float]:
    """Metricas adequadas a classe rara: PR-AUC pesa mais que ROC-AUC aqui."""
    y = np.asarray(y)
    return {
        "prevalencia": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "det@1%": taxa_deteccao_no_topo(y, p, 0.01),
        "det@5%": taxa_deteccao_no_topo(y, p, 0.05),
        "det@10%": taxa_deteccao_no_topo(y, p, 0.10),
        "det@20%": taxa_deteccao_no_topo(y, p, 0.20),
    }


def tabela_resultados(resultados: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(resultados).T
    return df.sort_values("pr_auc", ascending=False)
