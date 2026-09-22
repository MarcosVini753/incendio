from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from pipeline import dados, geo, modelos
from treinar_perigo_diario import construir

SAIDA = Path("resultados")
PREDITORES = ["veg", "dist_estrada", "dist_agua", "altitude"]
LADO_BLOCO_M = 25_000  


def montar(est: pd.DataFrame, alvo: str | int) -> tuple[pd.DataFrame, np.ndarray]:
    """Monta X, y. `alvo` = 'qualquer' (queimou em algum ano) ou um ano especifico."""
    if alvo == "qualquer":
        y = np.zeros(len(est), dtype=np.int8)
        for a in geo.ANOS_CICATRIZ:
            y |= est[f"queimou_{a}"].values.astype(np.int8)
    else:
        y = est[f"queimou_{int(alvo)}"].values.astype(np.int8)
    return est, y


def bloco_espacial(est: pd.DataFrame, lado_m: float = LADO_BLOCO_M) -> np.ndarray:
    """Identificador de bloco espacial de `lado_m` metros, para GroupKFold."""
    bx = np.floor(est["x_g"].values / lado_m).astype(np.int64)
    by = np.floor(est["y_g2"].values / lado_m).astype(np.int64)
    return bx * 100_000 + by


def equilibrar(y: np.ndarray, rng, razao: int = 1) -> np.ndarray:
    """Indices de um subconjunto com `razao` negativos por positivo."""
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    n = min(len(pos) * razao, len(neg))
    return np.concatenate([pos, rng.choice(neg, size=n, replace=False)])


def comparar_protocolos(est, y, semente=42):
    """Mesma familia de modelos, dois protocolos de validacao cruzada.

    aleatorio : StratifiedKFold(shuffle=True) - o do script original
    espacial  : GroupKFold sobre blocos de 25 km - sem gemeas entre treino e teste
    """
    rng = np.random.default_rng(semente)
    idx = equilibrar(y, rng, razao=1)
    X = est.iloc[idx][PREDITORES].to_numpy(dtype=np.float64)
    yb = y[idx]
    grupos = bloco_espacial(est)[idx]
    print(
        f"conjunto equilibrado: n={len(yb)} positivos={int(yb.sum())} "
        f"blocos espaciais distintos={len(np.unique(grupos))}"
    )

    protocolos = {
        "aleatorio (original)": list(
            StratifiedKFold(5, shuffle=True, random_state=1).split(X, yb)
        ),
        "espacial 25 km": list(GroupKFold(5).split(X, yb, groups=grupos)),
    }

    linhas = {}
    for nome_mdl, base in modelos.zoo(semente).items():
        for nome_prot, dobras in protocolos.items():
            pontos = []
            for trn, tst in dobras:
                mdl = modelos.zoo(semente)[nome_mdl]
                try:
                    mdl.fit(X[trn], yb[trn])
                    p = mdl.predict_proba(X[tst])[:, 1]
                except Exception as erro:
                    print(f"  {nome_mdl}/{nome_prot}: {type(erro).__name__}: {erro}")
                    pontos = []
                    break
                pontos.append(modelos.avaliar(yb[tst], p))
            if pontos:
                linhas[f"{nome_mdl} | {nome_prot}"] = (
                    pd.DataFrame(pontos).mean().to_dict()
                )
                print(
                    f"  {nome_mdl:14s} {nome_prot:22s} "
                    f"roc_auc={linhas[f'{nome_mdl} | {nome_prot}']['roc_auc']:.4f}"
                )
    tab = pd.DataFrame(linhas).T
    print("\n--- VALIDACAO ALEATORIA x ESPACIAL (media de 5 dobras) ---")
    print(tab.to_string(float_format="%.4f"))
    SAIDA.mkdir(exist_ok=True)
    tab.to_csv(SAIDA / "suscetibilidade_protocolos.csv")
    return tab


def buscar_k(est, y, semente=42, ks=range(3, 56, 4)):
    """Busca o k do FuzzyKNN com validacao ESPACIAL, e usa o resultado.

    O script original varre k de 3 a 55 com validacao aleatoria e depois nao
    aplica o vencedor. Aqui o k escolhido volta como valor de retorno.
    """
    rng = np.random.default_rng(semente)
    idx = equilibrar(y, rng, razao=1)
    X = est.iloc[idx][PREDITORES].to_numpy(dtype=np.float64)
    yb = y[idx]
    grupos = bloco_espacial(est)[idx]
    dobras = list(GroupKFold(5).split(X, yb, groups=grupos))

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    resultados = []
    for k in ks:
        t0 = time.time()
        aucs = []
        for trn, tst in dobras:
            mdl = Pipeline(
                [
                    ("imputar", SimpleImputer(strategy="median")),
                    ("escalar", StandardScaler()),
                    ("modelo", modelos.FuzzyKNN(k=k, m=2.0)),
                ]
            )
            mdl.fit(X[trn], yb[trn])
            aucs.append(modelos.avaliar(yb[tst], mdl.predict_proba(X[tst])[:, 1]))
        m = pd.DataFrame(aucs).mean()
        resultados.append({"k": k, **m.to_dict()})
        print(
            f"  k={k:3d} roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f} "
            f"({time.time()-t0:.1f} s)"
        )
    tab = pd.DataFrame(resultados)
    tab.to_csv(SAIDA / "busca_k_fuzzyknn.csv", index=False)
    melhor = int(tab.loc[tab["roc_auc"].idxmax(), "k"])
    print(f"\nk otimo por ROC-AUC com validacao espacial: {melhor}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tab["k"], tab["roc_auc"], "o-", label="ROC-AUC")
    ax.plot(tab["k"], tab["pr_auc"], "s-", label="PR-AUC")
    ax.axvline(melhor, ls="--", c="k", lw=0.8, label=f"k otimo = {melhor}")
    ax.set_xlabel("k (numero de vizinhos)")
    ax.set_ylabel("metrica (media de 5 dobras espaciais)")
    ax.set_title("FuzzyKNN: escolha de k com validacao espacial")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SAIDA / "busca_k_fuzzyknn.png", dpi=140)
    plt.close(fig)
    return melhor


def mapa_suscetibilidade(est, y, k_fuzzy, semente=42):
    """Treina no conjunto equilibrado e projeta em todas as celulas do estado."""
    rng = np.random.default_rng(semente)
    idx = equilibrar(y, rng, razao=1)
    Xtr = est.iloc[idx][PREDITORES].to_numpy(dtype=np.float64)
    ytr = y[idx]
    Xtudo = est[PREDITORES].to_numpy(dtype=np.float64)

    zoo = modelos.zoo(semente, k_fuzzy=k_fuzzy)
    fig, eixos = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, (nome, mdl) in zip(eixos.ravel(), zoo.items()):
        t0 = time.time()
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xtudo)[:, 1]
        g = ax.scatter(
            est["x_g"], est["y_g2"], c=p, cmap="turbo", marker="s", s=0.7,
            edgecolors="none", vmin=0, vmax=1,
        )
        ax.set_title(f"{nome}  ({time.time()-t0:.0f} s)")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(g, ax=ax, label="suscetibilidade", fraction=0.03)
        joblib.dump(
            {"modelo": mdl, "colunas": PREDITORES},
            SAIDA / f"modelo_suscetibilidade_{nome}.joblib",
        )
        print(f"  {nome}: mapa gerado, faixa [{p.min():.3f}, {p.max():.3f}]")
    fig.suptitle("Suscetibilidade a incendio - Acre - 4 modelos, mesmos preditores")
    destino = SAIDA / "mapa_suscetibilidade_comparado.png"
    fig.savefig(destino, dpi=140)
    plt.close(fig)
    print(f"mapa salvo em {destino}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--alvo", default="qualquer",
        help="'qualquer' (queimou em algum ano de 2006-2016) ou um ano, ex. 2010",
    )
    ap.add_argument("--busca-k", action="store_true", help="varre k do FuzzyKNN")
    ap.add_argument("--sem-mapa", action="store_true")
    a = ap.parse_args()

    _, est, _ = construir()
    est, y = montar(est, a.alvo)
    print(
        f"alvo = {a.alvo} | positivos = {int(y.sum())} de {len(y)} "
        f"({100*y.mean():.3f}%)\n"
    )

    comparar_protocolos(est, y)
    k = buscar_k(est, y) if a.busca_k else 29
    if not a.busca_k:
        print(f"\n(sem --busca-k: usando k={k}, o valor fixado no script original)")
    if not a.sem_mapa:
        mapa_suscetibilidade(est, y, k)


if __name__ == "__main__":
    main()
