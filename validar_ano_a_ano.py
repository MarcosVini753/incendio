"""Validacao deixando-um-ano-de-fora, incluindo os anos extremos.

Motivo
------
O teste principal de `treinar_perigo_diario.py` usa 2014-2015 como conjunto de
teste. Sao dois anos de POUCO fogo (239 e 292 celulas queimadas). Isso deixa
uma pergunta aberta: o modelo aguenta um ano de seca extrema, quando o padrao
espacial e temporal do fogo muda de escala?

2010 e o ano extremo com dados de clima disponiveis (1.969 celulas, 8x um ano
normal). Este script treina em todos os outros anos e testa em cada ano
isoladamente, para expor o desempenho ano a ano em vez de escondê-lo numa media.

Ressalva metodologica: ao testar 2010 o treino inclui anos posteriores a 2010,
ou seja usa futuro para prever passado. Isso NAO vale como simulacao
operacional. Vale como teste de robustez: o modelo consegue representar um ano
extremo, ou ele so funciona em anos medianos?

Uso
---
    python validar_ano_a_ano.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import dados, geo, modelos
from treinar_perigo_diario import CLIMA_ENXUTO, construir

SAIDA = Path("resultados")


def leave_one_year_out(am: pd.DataFrame, nomes_clima: list[str]) -> pd.DataFrame:
    cols = dados.colunas_preditoras(am, nomes_clima)
    linhas = {}
    for ano in geo.ANOS_CLIMA:
        trn = am[am["ano"] != ano]
        tst = am[am["ano"] == ano]
        mdl = modelos.zoo()["GradBoost"]
        mdl.fit(trn[cols].to_numpy(dtype=np.float64), trn["y"].to_numpy())
        p = mdl.predict_proba(tst[cols].to_numpy(dtype=np.float64))[:, 1]
        m = modelos.avaliar(tst["y"].to_numpy(), p)
        linhas[ano] = {"n_positivos": int(tst["y"].sum())} | m
        print(
            f"  {ano}: positivos={linhas[ano]['n_positivos']:5d} "
            f"roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f} "
            f"det@5%={m['det@5%']:.4f}"
        )
    return pd.DataFrame(linhas).T


def teste_espacial(am: pd.DataFrame, nomes_clima: list[str], est: pd.DataFrame):
    """O modelo generaliza para regioes que nao viu, ou memoriza coordenadas?

    lon/lat aparecem como 2o preditor mais importante, o que levanta a suspeita
    de memorizacao espacial. Aqui o teste e duplo:
      1. separacao por bloco espacial de 50 km (celulas de teste longe do treino);
      2. o mesmo, mas removendo lon/lat dos preditores.
    Se o desempenho desabar no caso 1 e se manter no caso 2, o modelo estava
    apoiado em coordenada, nao em processo.
    """
    from sklearn.model_selection import GroupKFold

    todos = dados.colunas_preditoras(am, nomes_clima)
    sem_coord = [c for c in todos if c not in ("lon", "lat")]

    bx = np.floor(est["x_g"].values / 50_000).astype(np.int64)
    by = np.floor(est["y_g2"].values / 50_000).astype(np.int64)
    bloco_da_celula = bx * 100_000 + by
    grupos = bloco_da_celula[am["celula"].values]
    print(f"  blocos espaciais de 50 km: {len(np.unique(grupos))}")

    linhas = {}
    for rotulo, cols in [("com lon/lat", todos), ("sem lon/lat", sem_coord)]:
        pontos = []
        for trn, tst in GroupKFold(5).split(am, am["y"], groups=grupos):
            mdl = modelos.zoo()["GradBoost"]
            mdl.fit(
                am.iloc[trn][cols].to_numpy(dtype=np.float64),
                am.iloc[trn]["y"].to_numpy(),
            )
            p = mdl.predict_proba(am.iloc[tst][cols].to_numpy(dtype=np.float64))[:, 1]
            pontos.append(modelos.avaliar(am.iloc[tst]["y"].to_numpy(), p))
        linhas[f"blocos 50 km, {rotulo}"] = pd.DataFrame(pontos).mean().to_dict()
        print(
            f"  {rotulo}: roc_auc={linhas[f'blocos 50 km, {rotulo}']['roc_auc']:.4f} "
            f"pr_auc={linhas[f'blocos 50 km, {rotulo}']['pr_auc']:.4f} "
            f"det@5%={linhas[f'blocos 50 km, {rotulo}']['det@5%']:.4f}"
        )
    return pd.DataFrame(linhas).T


def linha_de_base(am: pd.DataFrame) -> pd.DataFrame:
    """Referencias burras, para dar escala as metricas.

    Sem uma linha de base, PR-AUC 0.35 nao significa nada. Aqui entram:
      aleatorio      : sorteio uniforme
      so_calendario  : usa apenas o dia do ano (a climatologia)
      so_historico   : usa apenas a taxa historica de queima da celula
    """
    tst = am[am["ano"].isin((2014, 2015))]
    y = tst["y"].to_numpy()
    rng = np.random.default_rng(0)
    linhas = {
        "aleatorio": modelos.avaliar(y, rng.random(len(y))),
        "so dia do ano": modelos.avaliar(y, _densidade_sazonal(am, tst)),
        "so historico da celula": modelos.avaliar(
            y, np.nan_to_num(tst["taxa_queima_hist"].to_numpy(), nan=0.0)
        ),
    }
    return pd.DataFrame(linhas).T


def _densidade_sazonal(am: pd.DataFrame, tst: pd.DataFrame) -> np.ndarray:
    """Probabilidade de fogo por dia do ano, estimada no treino (climatologia)."""
    trn = am[am["ano"].isin(range(2006, 2013))]
    taxa = trn.groupby("dia_ano")["y"].mean()
    suave = taxa.reindex(range(1, 367)).interpolate().rolling(15, center=True,
                                                             min_periods=1).mean()
    return suave.reindex(tst["dia_ano"].values).to_numpy()


def main():
    am, est, nomes = construir()

    print("\n=== 1. LINHAS DE BASE (teste 2014-2015) ===")
    base = linha_de_base(am)
    print(base.to_string(float_format="%.4f"))

    print("\n=== 2. DEIXANDO UM ANO DE FORA ===")
    loyo = leave_one_year_out(am, nomes)
    print("\n" + loyo.to_string(float_format="%.4f"))
    print(
        f"\n  ROC-AUC: media {loyo['roc_auc'].mean():.4f} "
        f"min {loyo['roc_auc'].min():.4f} ({loyo['roc_auc'].idxmin()}) "
        f"max {loyo['roc_auc'].max():.4f} ({loyo['roc_auc'].idxmax()})"
    )
    print(
        f"  det@5% : media {loyo['det@5%'].mean():.4f} "
        f"min {loyo['det@5%'].min():.4f} ({loyo['det@5%'].idxmin()})"
    )

    print("\n=== 3. GENERALIZACAO ESPACIAL (blocos de 50 km) ===")
    esp = teste_espacial(am, nomes, est)
    print("\n" + esp.to_string(float_format="%.4f"))

    SAIDA.mkdir(exist_ok=True)
    base.to_csv(SAIDA / "linhas_de_base.csv")
    loyo.to_csv(SAIDA / "validacao_ano_a_ano.csv")
    esp.to_csv(SAIDA / "validacao_espacial_diario.csv")
    print(f"\ntabelas salvas em {SAIDA}")


if __name__ == "__main__":
    main()
