from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import dados, geo, modelos

SAIDA = Path("resultados")
ANOS_TREINO = tuple(range(2006, 2013))
ANOS_VALIDACAO = (2013,)
ANOS_TESTE = (2014, 2015)


def construir(cache: bool = True):
    """Monta (ou recupera do cache) a tabela (celula, dia) e a estatica."""
    SAIDA.mkdir(exist_ok=True)
    cache_am = SAIDA / "amostra_celula_dia.parquet"
    cache_est = SAIDA / "estatica.parquet"
    cache_nomes = SAIDA / "nomes_clima.json"

    if cache and cache_am.exists() and cache_est.exists() and cache_nomes.exists():
        print("[cache] lendo tabelas de", SAIDA)
        return (
            pd.read_parquet(cache_am),
            pd.read_parquet(cache_est),
            json.loads(cache_nomes.read_text()),
        )

    t0 = time.time()
    print("[1/4] planilha de centroides + agregacao das cicatrizes...")
    est = dados.tabela_estatica()
    n_cel = len(est)
    print(f"      {n_cel} celulas | descartadas {est.attrs.get('linhas_descartadas', 0)}")
    for ano in geo.ANOS_CICATRIZ:
        print(
            f"      {ano}: {int(est[f'queimou_{ano}'].sum()):5d} celulas com cicatriz "
            f"({int(est[f'npix_{ano}'].sum()):5d} pixels)"
        )

    print(f"[2/4] preditores meteorologicos na grade de 0.1 graus...")
    cubo, nomes = dados.features_climaticas()
    print(f"      cubo {cubo.shape} ({cubo.nbytes/1e6:.0f} MB) | {len(nomes)} features")

    print("[3/4] amostragem (celula, dia)...")
    am = dados.amostrar_celula_dia(est, cubo, nomes)
    print(
        f"      {len(am)} pares | positivos {int(am['y'].sum())} "
        f"| prevalencia {100*am['y'].mean():.3f}%"
    )
    del cubo

    print("[4/4] gravando cache...")
    am.to_parquet(cache_am, index=False)
    est.to_parquet(cache_est, index=False)
    cache_nomes.write_text(json.dumps(nomes))
    print(f"      pronto em {time.time()-t0:.1f} s")
    return am, est, nomes


def treinar(am: pd.DataFrame, nomes_clima: list[str]):
    cols = dados.colunas_preditoras(am, nomes_clima)
    print(f"\n{len(cols)} preditores: {cols}\n")

    trn = am[am["ano"].isin(ANOS_TREINO)]
    val = am[am["ano"].isin(ANOS_VALIDACAO)]
    tst = am[am["ano"].isin(ANOS_TESTE)]
    for nome, parte in [("treino", trn), ("validacao", val), ("teste", tst)]:
        print(
            f"  {nome:10s} n={len(parte):7d}  positivos={int(parte['y'].sum()):5d}  "
            f"prevalencia={100*parte['y'].mean():.3f}%  anos={sorted(parte['ano'].unique())}"
        )

    Xtr, ytr = trn[cols].to_numpy(dtype=np.float64), trn["y"].to_numpy()
    Xva, yva = val[cols].to_numpy(dtype=np.float64), val["y"].to_numpy()
    Xte, yte = tst[cols].to_numpy(dtype=np.float64), tst["y"].to_numpy()

    res_val, res_tst, ajustados = {}, {}, {}
    for nome, mdl in modelos.zoo().items():
        t0 = time.time()
        try:
            mdl.fit(Xtr, ytr)
            pva = mdl.predict_proba(Xva)[:, 1]
            pte = mdl.predict_proba(Xte)[:, 1]
        except Exception as erro:  # falha de um modelo nao derruba a comparacao
            print(f"  {nome}: FALHOU -> {type(erro).__name__}: {erro}")
            continue
        res_val[nome] = modelos.avaliar(yva, pva)
        res_tst[nome] = modelos.avaliar(yte, pte)
        ajustados[nome] = mdl
        print(f"  {nome}: {time.time()-t0:.1f} s")

    print("\n--- VALIDACAO (2013) ---")
    print(modelos.tabela_resultados(res_val).to_string(float_format="%.4f"))
    print("\n--- TESTE (2014-2015) ---")
    tab_tst = modelos.tabela_resultados(res_tst)
    print(tab_tst.to_string(float_format="%.4f"))

    SAIDA.mkdir(exist_ok=True)
    modelos.tabela_resultados(res_val).to_csv(SAIDA / "metricas_validacao.csv")
    tab_tst.to_csv(SAIDA / "metricas_teste.csv")

    melhor = modelos.tabela_resultados(res_val).index[0]
    print(f"\nmelhor por PR-AUC na validacao: {melhor}")
    joblib.dump(
        {"modelo": ajustados[melhor], "colunas": cols, "nome": melhor},
        SAIDA / "modelo_perigo_diario.joblib",
    )
    print(f"modelo salvo em {SAIDA/'modelo_perigo_diario.joblib'}")

    _importancias(ajustados[melhor], Xva, yva, cols, melhor)
    return ajustados[melhor], cols


PAISAGEM = ["veg", "dist_estrada", "dist_agua", "altitude"]
LOCAL = ["lon", "lat"]
HISTORICO = ["taxa_queima_hist", "queimou_ano_anterior"]
CALENDARIO = ["dia_ano", "dia_sin", "dia_cos"]
CONTAGIO = dados.NOMES_CONTAGIO

CLIMA_ENXUTO = [
    "ur_media_7d",
    "ur_min_15d",
    "prec_dia",
    "prec_acum_30d",
    "prec_acum_90d",
    "dias_estiagem",
    "indice_seca",
]


def ablacao(am: pd.DataFrame, nomes_clima: list[str]):
    """Quanto cada bloco de preditores acrescenta, na mesma separacao temporal.

    O ponto sutil: `dia_ano` e uma CLIMATOLOGIA, nao uma previsao. Um modelo que
    so olha o calendario sabe que agosto e perigoso, mas nunca vai saber que
    ESTE agosto esta mais seco que o normal. Por isso a tabela inclui as versoes
    sem calendario: e ali que se ve o que os rasters de clima realmente entregam.
    """
    todos = dados.colunas_preditoras(am, nomes_clima)
    base = PAISAGEM + LOCAL + HISTORICO + CALENDARIO
    conjuntos = {
        "A paisagem (o que o projeto usa hoje)": PAISAGEM,
        "B A + local + historico + calendario": base,
        "C B + clima completo": [c for c in todos if c not in CONTAGIO],
        "D B + clima enxuto": base + CLIMA_ENXUTO,
        "E sem calendario, sem clima": PAISAGEM + LOCAL + HISTORICO,
        "F sem calendario, com clima enxuto": PAISAGEM + LOCAL + HISTORICO + CLIMA_ENXUTO,
        "G so clima enxuto + calendario": CLIMA_ENXUTO + CALENDARIO,
        "H B + contagio": base + CONTAGIO,
        "I B + clima enxuto + contagio": base + CLIMA_ENXUTO + CONTAGIO,
        "J tudo": todos,
        "K so contagio (persistencia)": CONTAGIO,
        "L so dist_fogo_recente_m": ["dist_fogo_recente_m"],
        "M tudo, sem dist_fogo_recente_m": [
            c for c in todos if c != "dist_fogo_recente_m"
        ],
    }

    trn = am[am["ano"].isin(ANOS_TREINO)]
    tst = am[am["ano"].isin(ANOS_TESTE)]
    linhas = {}
    for rotulo, cols in conjuntos.items():
        cols = [c for c in cols if c in am.columns]
        mdl = modelos.zoo()["GradBoost"]
        mdl.fit(trn[cols].to_numpy(dtype=np.float64), trn["y"].to_numpy())
        p = mdl.predict_proba(tst[cols].to_numpy(dtype=np.float64))[:, 1]
        linhas[rotulo] = {"n_preditores": len(cols)} | modelos.avaliar(
            tst["y"].to_numpy(), p
        )
        print(f"  {rotulo}: pr_auc={linhas[rotulo]['pr_auc']:.4f}")
    tab = pd.DataFrame(linhas).T
    print("\n--- ABLACAO (GradBoost, treino 2006-2012, teste 2014-2015) ---")
    print(tab.to_string(float_format="%.4f"))
    SAIDA.mkdir(exist_ok=True)
    tab.to_csv(SAIDA / "ablacao.csv")
    return tab


def _importancias(mdl, X, y, cols, nome):
    """Importancia por permutacao, medida em queda de PR-AUC."""
    from sklearn.inspection import permutation_importance

    print("\ncalculando importancia por permutacao (pode levar 1-2 min)...")
    r = permutation_importance(
        mdl, X, y, n_repeats=5, random_state=0, scoring="average_precision", n_jobs=1
    )
    imp = (
        pd.DataFrame({"preditor": cols, "queda_pr_auc": r.importances_mean})
        .sort_values("queda_pr_auc", ascending=False)
        .reset_index(drop=True)
    )
    print(imp.head(15).to_string(index=False, float_format="%.5f"))
    imp.to_csv(SAIDA / "importancia_preditores.csv", index=False)

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["preditor"], top["queda_pr_auc"], color="#b5401f")
    ax.set_xlabel("queda de PR-AUC ao permutar o preditor")
    ax.set_title(f"Importancia dos preditores - {nome}")
    fig.tight_layout()
    fig.savefig(SAIDA / "importancia_preditores.png", dpi=140)
    plt.close(fig)


def mapa_do_dia(mdl, cols, est: pd.DataFrame, data: str):
    """Gera o mapa de perigo para uma data especifica, celula por celula."""
    d = np.datetime64(data)
    banda = int((d - geo.DATA_INICIO_CLIMA).astype(int))
    if not 0 <= banda < geo.N_DIAS_CLIMA:
        raise SystemExit(
            f"{data} fora da cobertura climatica "
            f"(2006-01-01 a 2015-12-31, banda calculada {banda})"
        )
    ano = int(str(d)[:4])
    dia_ano = banda - int(
        (np.datetime64(f"{ano}-01-01") - geo.DATA_INICIO_CLIMA).astype(int)
    ) + 1

    print(f"\ngerando mapa para {data} (dia {dia_ano} de {ano})")
    cubo, nomes = dados.features_climaticas()

    sel = est[est["cli_id"] >= 0].copy()
    lin = pd.DataFrame(index=sel.index)
    for c in dados.COLUNAS_ESTATICAS:
        lin[c] = sel[c].values

    anos_arr = np.array(geo.ANOS_CICATRIZ)
    ant = anos_arr < ano
    mq = np.column_stack([sel[f"queimou_{a}"].values for a in geo.ANOS_CICATRIZ]).astype(
        np.float32
    )
    lin["taxa_queima_hist"] = mq[:, ant].mean(axis=1) if ant.any() else np.nan
    lin["queimou_ano_anterior"] = (
        mq[:, anos_arr == ano - 1].ravel() if (anos_arr == ano - 1).any() else np.nan
    )
    ang = 2 * np.pi * dia_ano / 366.0
    lin["dia_sin"], lin["dia_cos"], lin["dia_ano"] = np.sin(ang), np.cos(ang), dia_ano
    ids = sel["cli_id"].values.astype(int)
    for k, nome in enumerate(nomes):
        lin[nome] = cubo[ids, banda, k]
    del cubo

    # contagio: fogo detectado nos dias anteriores, dentro do proprio ano
    dia_cel = sel[f"dia_{ano}"].values.astype(np.int64)
    queimou = dia_cel > 0
    cont = dados.contagio(
        sel["x_g"].values, sel["y_g2"].values,
        np.full(len(sel), dia_ano, dtype=np.int64),
        sel["x_g"].values[queimou], sel["y_g2"].values[queimou], dia_cel[queimou],
    )
    for c, v in cont.items():
        lin[c] = v

    p = mdl.predict_proba(lin[cols].to_numpy(dtype=np.float64))[:, 1]

    fig, ax = plt.subplots(figsize=(11, 6))
    g = ax.scatter(
        sel["x_g"], sel["y_g2"], c=p, cmap="turbo", marker="s", s=1.2,
        edgecolors="none", vmin=0, vmax=max(p.max(), 1e-6),
    )
    plt.colorbar(g, ax=ax, label="probabilidade relativa de ignicao")

    real = sel[sel[f"dia_{ano}"].values == dia_ano]
    if len(real):
        ax.scatter(
            real["x_g"], real["y_g2"], facecolors="none", edgecolors="white",
            s=45, linewidths=1.1, label=f"cicatriz observada ({len(real)})",
        )
        ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Perigo diario de incendio - Acre - {data}")
    ax.set_xlabel("UTM 19S leste (m)")
    ax.set_ylabel("UTM 19S norte (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    destino = SAIDA / f"mapa_perigo_{data}.png"
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"mapa salvo em {destino}")
    if len(real):
        lim = np.quantile(p, 0.95)
        acerto = (p[sel[f"dia_{ano}"].values == dia_ano] >= lim).mean()
        print(
            f"das {len(real)} cicatrizes desse dia, {100*acerto:.1f}% cairam nos "
            f"5% de celulas mais perigosas do mapa"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sem-cache", action="store_true", help="recalcula as tabelas")
    ap.add_argument("--mapa", metavar="AAAA-MM-DD", help="gera o mapa de um dia")
    ap.add_argument("--sem-ablacao", action="store_true", help="pula a ablacao")
    a = ap.parse_args()

    am, est, nomes = construir(cache=not a.sem_cache)
    mdl, cols = treinar(am, nomes)
    if not a.sem_ablacao:
        ablacao(am, nomes)
    if a.mapa:
        mapa_do_dia(mdl, cols, est, a.mapa)


if __name__ == "__main__":
    main()
