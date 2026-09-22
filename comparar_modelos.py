from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from pipeline import dados, modelos
from treinar_perigo_diario import ANOS_TESTE, ANOS_TREINO, ANOS_VALIDACAO, construir

SAIDA = Path("resultados")
N_BOOTSTRAP = 2000
SEMENTE = 7


def treinar_todos(am, nomes):
    cols = dados.colunas_preditoras(am, nomes)
    trn = am[am["ano"].isin(ANOS_TREINO)]
    val = am[am["ano"].isin(ANOS_VALIDACAO)]
    tst = am[am["ano"].isin(ANOS_TESTE)]
    Xtr, ytr = trn[cols].to_numpy(dtype=np.float64), trn["y"].to_numpy()
    Xva, yva = val[cols].to_numpy(dtype=np.float64), val["y"].to_numpy()
    Xte, yte = tst[cols].to_numpy(dtype=np.float64), tst["y"].to_numpy()

    prev_val, prev_tst = {}, {}
    for nome, mdl in modelos.zoo().items():
        mdl.fit(Xtr, ytr)
        prev_val[nome] = mdl.predict_proba(Xva)[:, 1]
        prev_tst[nome] = mdl.predict_proba(Xte)[:, 1]
        print(f"  {nome} treinado")
    return yva, prev_val, yte, prev_tst


def bootstrap(y, previsoes: dict[str, np.ndarray], n=N_BOOTSTRAP):
    """Reamostra as mesmas linhas para todos os modelos (pareado)."""
    rng = np.random.default_rng(SEMENTE)
    nomes = list(previsoes)
    guarda = {nome: {"roc_auc": [], "pr_auc": []} for nome in nomes}
    n_linhas = len(y)
    descartadas = 0
    for _ in range(n):
        idx = rng.integers(0, n_linhas, n_linhas)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            descartadas += 1
            continue
        for nome in nomes:
            pb = previsoes[nome][idx]
            guarda[nome]["roc_auc"].append(roc_auc_score(yb, pb))
            guarda[nome]["pr_auc"].append(average_precision_score(yb, pb))
    if descartadas:
        print(f"  ({descartadas} reamostragens descartadas por falta de positivos)")
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in guarda.items()}


def tabela_ic(y, previsoes, dist) -> pd.DataFrame:
    linhas = {}
    for nome, p in previsoes.items():
        linha = {}
        for met, f in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
            amostras = dist[nome][met]
            lo, hi = np.percentile(amostras, [2.5, 97.5])
            linha[met] = f(y, p)
            linha[f"{met}_ic95"] = f"[{lo:.4f}, {hi:.4f}]"
        linhas[nome] = linha
    return pd.DataFrame(linhas).T


def comparar_pares(dist, metrica="pr_auc") -> pd.DataFrame:
    nomes = list(dist)
    linhas = []
    for i, a in enumerate(nomes):
        for b in nomes[i + 1 :]:
            d = dist[a][metrica] - dist[b][metrica]
            lo, hi = np.percentile(d, [2.5, 97.5])
            empate = lo <= 0 <= hi
            linhas.append(
                {
                    "modelo_A": a,
                    "modelo_B": b,
                    "dif_media": d.mean(),
                    "ic95_inf": lo,
                    "ic95_sup": hi,
                    "P(A>B)": (d > 0).mean(),
                    "conclusao": "EMPATE" if empate else ("A ganha" if d.mean() > 0 else "B ganha"),
                }
            )
    return pd.DataFrame(linhas)


def grafico(dist, caminho):
    nomes = list(dist)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, met, rot in [(a1, "roc_auc", "ROC-AUC"), (a2, "pr_auc", "PR-AUC")]:
        dados_bp = [dist[n][met] for n in nomes]
        ax.boxplot(dados_bp, tick_labels=nomes, showfliers=False)
        ax.set_ylabel(rot)
        ax.set_title(f"{rot} - {N_BOOTSTRAP} reamostragens")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Incerteza das metricas no teste 2014-2015 (bootstrap pareado)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


def main():
    am, _, nomes_clima = construir()
    print("treinando os 4 modelos...")
    yva, prev_val, yte, prev_tst = treinar_todos(am, nomes_clima)

    print(f"\nbootstrap com {N_BOOTSTRAP} reamostragens do teste...")
    dist = bootstrap(yte, prev_tst)

    print("\n=== DESEMPENHO NO TESTE COM IC 95% ===")
    tab = tabela_ic(yte, prev_tst, dist)
    print(tab.to_string())

    print("\n=== COMPARACAO PAREADA (PR-AUC) ===")
    pares = comparar_pares(dist, "pr_auc")
    print(pares.to_string(index=False, float_format="%.4f"))

    print("\n=== COMPARACAO PAREADA (ROC-AUC) ===")
    pares_roc = comparar_pares(dist, "roc_auc")
    print(pares_roc.to_string(index=False, float_format="%.4f"))

    SAIDA.mkdir(exist_ok=True)
    tab.to_csv(SAIDA / "comparacao_ic95.csv")
    pares.to_csv(SAIDA / "comparacao_pareada_prauc.csv", index=False)
    pares_roc.to_csv(SAIDA / "comparacao_pareada_rocauc.csv", index=False)
    grafico(dist, SAIDA / "comparacao_modelos_bootstrap.png")
    print(f"\ntabelas e figura salvas em {SAIDA}")


if __name__ == "__main__":
    main()
