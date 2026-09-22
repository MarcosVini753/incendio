from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from pipeline import dados, modelos
from treinar_perigo_diario import ANOS_TESTE, ANOS_VALIDACAO, construir

SAIDA = Path("resultados")


def metricas_no_limiar(y, p, limiar: float) -> dict:
    yp = (p >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return {
        "limiar": limiar,
        "VP": int(vp),
        "FP": int(fp),
        "FN": int(fn),
        "VN": int(vn),
        "acuracia": accuracy_score(y, yp),
        "acuracia_balanceada": balanced_accuracy_score(y, yp),
        "precisao": precision_score(y, yp, zero_division=0),
        "recall": recall_score(y, yp, zero_division=0),
        "especificidade": vn / (vn + fp) if (vn + fp) else np.nan,
        "f1": f1_score(y, yp, zero_division=0),
        "kappa": cohen_kappa_score(y, yp),
        "mcc": matthews_corrcoef(y, yp) if len(np.unique(yp)) > 1 else 0.0,
    }


def desenhar_matriz(y, p, limiar, titulo, caminho):
    yp = (p >= limiar).astype(int)
    mc = confusion_matrix(y, yp, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(mc, cmap="Blues")
    rotulos = ["sem fogo", "com fogo"]
    ax.set_xticks([0, 1], rotulos)
    ax.set_yticks([0, 1], rotulos)
    ax.set_xlabel("previsto pelo modelo")
    ax.set_ylabel("observado")
    ax.set_title(titulo)
    total = mc.sum()
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{mc[i, j]}\n{100*mc[i, j]/total:.2f}%",
                ha="center", va="center",
                color="white" if mc[i, j] > mc.max() / 2 else "black",
                fontsize=12,
            )
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


def curvas(y, p, caminho):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

    a1.plot(fpr, tpr, lw=2, label=f"modelo (AUC = {np.trapezoid(tpr, fpr):.3f})")
    a1.plot([0, 1], [0, 1], "k--", lw=0.8, label="aleatorio")
    a1.set_xlabel("taxa de falso positivo")
    a1.set_ylabel("taxa de verdadeiro positivo (recall)")
    a1.set_title("Curva ROC")
    a1.legend()

    a2.plot(rec, prec, lw=2, color="#b5401f", label="modelo")
    a2.axhline(y.mean(), ls="--", c="k", lw=0.8,
               label=f"aleatorio (prevalencia = {100*y.mean():.2f}%)")
    a2.set_xlabel("recall")
    a2.set_ylabel("precisao")
    a2.set_title("Curva Precisao-Recall")
    a2.legend()

    fig.suptitle("Modelo de perigo diario - teste 2014-2015")
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


def calibracao(y, p, caminho):
    """As probabilidades correspondem a frequencias reais?"""
    from sklearn.calibration import calibration_curve

    frac, media = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="calibracao perfeita")
    ax.plot(media, frac, "o-", label="modelo")
    ax.set_xlabel("probabilidade prevista")
    ax.set_ylabel("frequencia observada de fogo")
    ax.set_title("Calibracao (teste 2014-2015)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


def main():
    am, _, nomes = construir()
    pacote = joblib.load(SAIDA / "modelo_perigo_diario.joblib")
    mdl, cols = pacote["modelo"], pacote["colunas"]
    print(f"modelo: {pacote['nome']}\n")

    val = am[am["ano"].isin(ANOS_VALIDACAO)]
    tst = am[am["ano"].isin(ANOS_TESTE)]
    yva = val["y"].to_numpy()
    yte = tst["y"].to_numpy()
    pva = mdl.predict_proba(val[cols].to_numpy(dtype=np.float64))[:, 1]
    pte = mdl.predict_proba(tst[cols].to_numpy(dtype=np.float64))[:, 1]

    # limiar de F1 maximo escolhido na VALIDACAO, nunca no teste
    prec, rec, lim = precision_recall_curve(yva, pva)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    lim_f1 = float(lim[max(f1[:-1].argmax(), 0)])
    lim_top5 = float(np.quantile(pte, 0.95))
    print(f"limiar de F1 maximo (definido em 2013): {lim_f1:.4f}")
    print(f"limiar do top 5% (definido no teste)  : {lim_top5:.4f}\n")

    linhas = {
        'trivial "nunca ha fogo"': metricas_no_limiar(yte, np.zeros_like(pte), 0.5),
    }
    for rotulo, L in [
        ("limiar 0,50 (convencao)", 0.5),
        (f"limiar F1 max ({lim_f1:.3f})", lim_f1),
        (f"limiar top 5% ({lim_top5:.3f})", lim_top5),
    ]:
        linhas[rotulo] = metricas_no_limiar(yte, pte, L)

    tab = pd.DataFrame(linhas).T
    print("=== TESTE 2014-2015 | n =", len(yte), "| positivos =", int(yte.sum()), "===")
    print(tab.to_string(float_format="%.4f"))

    SAIDA.mkdir(exist_ok=True)
    tab.to_csv(SAIDA / "metricas_classificacao.csv")
    desenhar_matriz(yte, pte, lim_f1,
                    f"Matriz de confusao - limiar F1 max = {lim_f1:.3f}",
                    SAIDA / "matriz_confusao_f1.png")
    desenhar_matriz(yte, pte, lim_top5,
                    f"Matriz de confusao - limiar top 5% = {lim_top5:.3f}",
                    SAIDA / "matriz_confusao_top5.png")
    curvas(yte, pte, SAIDA / "curvas_roc_pr.png")
    calibracao(yte, pte, SAIDA / "calibracao.png")

    print("\n--- por que a acuracia engana aqui ---")
    triv = linhas['trivial "nunca ha fogo"']["acuracia"]
    print(f"  o modelo trivial 'nunca ha fogo' tem acuracia de {100*triv:.2f}%")
    print(f"  e recall de 0%: ele nao encontra UM unico incendio.")
    print("  reporte recall, precisao, F1, kappa e MCC. Nao reporte acuracia sozinha.")
    print(f"\nfiguras e tabela salvas em {SAIDA}")


if __name__ == "__main__":
    main()
