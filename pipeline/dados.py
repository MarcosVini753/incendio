"""Construcao das tabelas de treino a partir da planilha + 3 rasters.

Duas tabelas sao produzidas:

1. `tabela_estatica()`      -> uma linha por celula (307.410). Preditores de
   paisagem que nao mudam no tempo. Base do modelo de SUSCETIBILIDADE.

2. `amostrar_celula_dia()`  -> uma linha por par (celula, dia). Preditores de
   paisagem + preditores meteorologicos acumulados. Base do modelo de
   PERIGO DIARIO.

O elo entre a planilha (UTM19S, celula 893x598 m) e o raster de cicatrizes
(WGS84, pixel ~500 m) e feito por mapeamento INVERSO: cada um dos 1,47 milhao
de pixels do raster e atribuido a celula que o contem. Isso preserva toda a
area queimada, ao contrario da amostragem do vizinho mais proximo, que perde
~55% dos pixels porque a celula e maior que o pixel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geo

# janelas (em dias) dos acumulados meteorologicos
JANELAS_PRECIPITACAO = (3, 7, 15, 30, 60, 90)
JANELAS_UMIDADE = (3, 7, 15, 30)
LIMIAR_DIA_CHUVOSO = 1.0  # mm; abaixo disso o dia conta como seco

# recorte sazonal: 99,8% das cicatrizes ocorrem a partir do dia 121 (1 de maio)
DIA_INICIO_ESTACAO = 121
DIA_FIM_ESTACAO = 334  # 30 de novembro

COLUNAS_ESTATICAS = ["veg", "dist_estrada", "dist_agua", "altitude", "lon", "lat"]


# --------------------------------------------------------------------- estatica


def tabela_estatica(centroides: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabela por celula: preditores de paisagem + indices das grades + cicatrizes.

    Colunas adicionadas:
      cli_lin, cli_col      indices na grade climatica (0.1 deg)
      cli_id                indice na matriz de series climaticas validas (-1 = sem dado)
      queimou_YYYY          bool, celula com pelo menos 1 pixel de cicatriz no ano
      dia_YYYY              int, primeiro dia do ano com deteccao (0 = nao queimou)
      npix_YYYY             int, numero de pixels de cicatriz na celula
    """
    d = geo.ler_centroides() if centroides is None else centroides.copy()

    # ---- indices na grade climatica, com preenchimento pela celula valida mais proxima
    grade_cli = geo.ler_grade(geo.ARQ_UMIDADE)
    _, mascara_cli = geo.clima_valido(geo.ARQ_UMIDADE)
    lin, col = grade_cli.indices(d["lon"].values, d["lat"].values)
    lin, col = _puxar_para_valido(lin, col, mascara_cli)
    d["cli_lin"], d["cli_col"] = lin, col

    # id linear dentro das celulas validas (ordem de mascara_cli achatada, igual
    # a ordem devolvida por geo.clima_valido)
    id_cli = np.full(mascara_cli.shape, -1, dtype=np.int64)
    id_cli[mascara_cli] = np.arange(int(mascara_cli.sum()))
    d["cli_id"] = id_cli[lin, col]

    # ---- cicatrizes agregadas por celula
    cic = _agregar_cicatrizes(d)
    for ano in geo.ANOS_CICATRIZ:
        d[f"queimou_{ano}"] = cic["queimou"][ano]
        d[f"dia_{ano}"] = cic["dia"][ano]
        d[f"npix_{ano}"] = cic["npix"][ano]

    return d


def _puxar_para_valido(lin, col, mascara):
    """Move indices que caem em celula sem dado para a celula valida mais proxima."""
    invalido = ~mascara[lin, col]
    if not invalido.any():
        return lin, col
    vl, vc = np.nonzero(mascara)
    li, ci = lin[invalido], col[invalido]
    # busca exaustiva: |validas| ~ 1375, |invalidos| ~ 16 mil -> 22 M distancias, ok
    d2 = (vl[None, :] - li[:, None]) ** 2 + (vc[None, :] - ci[:, None]) ** 2
    melhor = d2.argmin(axis=1)
    lin = lin.copy()
    col = col.copy()
    lin[invalido] = vl[melhor]
    col[invalido] = vc[melhor]
    return lin, col


def _agregar_cicatrizes(centroides: pd.DataFrame) -> dict:
    """Agrega o raster de cicatrizes na grade de centroides por mapeamento inverso.

    Para cada pixel do raster: lon/lat -> UTM19S -> indice (Y, X) da grade ->
    linha da planilha. Depois reduz por celula (queimou / primeiro dia / n pixels).
    """
    grade = geo.ler_grade(geo.ARQ_CICATRIZES)
    cubo = geo.ler_cubo(geo.ARQ_CICATRIZES)  # (897, 1642, 11) int16

    lat_c, lon_c = grade.centros()
    malha_lon, malha_lat = np.meshgrid(lon_c, lat_c)
    x_utm, y_utm = geo.lonlat_para_utm(malha_lon.ravel(), malha_lat.ravel())
    yi, xi = geo.utm_para_indice_grade(x_utm, y_utm)

    # tabela de consulta (Y, X) -> posicao na planilha
    Y = centroides["Y"].values.astype(np.int64)
    X = centroides["X"].values.astype(np.int64)
    y_min, x_min = Y.min(), X.min()
    n_y, n_x = Y.max() - y_min + 1, X.max() - x_min + 1
    consulta = np.full((n_y, n_x), -1, dtype=np.int64)
    consulta[Y - y_min, X - x_min] = np.arange(len(centroides))

    yi -= y_min
    xi -= x_min
    dentro = (yi >= 0) & (yi < n_y) & (xi >= 0) & (xi < n_x)
    destino = np.full(yi.shape, -1, dtype=np.int64)
    destino[dentro] = consulta[yi[dentro], xi[dentro]]

    n_cel = len(centroides)
    saida = {"queimou": {}, "dia": {}, "npix": {}}
    for k, ano in enumerate(geo.ANOS_CICATRIZ):
        dia_pix = cubo[:, :, k].ravel()
        sel = (destino >= 0) & (dia_pix > 0)
        alvo = destino[sel]
        dias = dia_pix[sel].astype(np.int64)

        npix = np.bincount(alvo, minlength=n_cel).astype(np.int32)
        primeiro = np.zeros(n_cel, dtype=np.int16)
        # menor dia por celula: np.minimum.at sobre vetor inicializado alto
        tmp = np.full(n_cel, 9999, dtype=np.int64)
        np.minimum.at(tmp, alvo, dias)
        tem = tmp < 9999
        primeiro[tem] = tmp[tem].astype(np.int16)

        saida["queimou"][ano] = npix > 0
        saida["dia"][ano] = primeiro
        saida["npix"][ano] = npix
    del cubo
    return saida


# ------------------------------------------------------------ clima acumulado


def features_climaticas() -> tuple[np.ndarray, list[str]]:
    """Calcula os preditores meteorologicos na propria grade climatica.

    Trabalhar na grade de 0.1 deg (1375 celulas x 3652 dias) em vez de na grade
    de centroides (307 mil x 3652) reduz o custo em ~220x. Depois basta indexar
    por (cli_id, dia).

    Retorna (cubo[n_celulas, n_dias, n_features] float32, nomes).
    """
    prec, _ = geo.clima_valido(geo.ARQ_PRECIPITACAO)  # (1375, 3652) mm
    umid, _ = geo.clima_valido(geo.ARQ_UMIDADE)  # (1375, 3652) %
    n_cel, n_dias = prec.shape

    feats: list[np.ndarray] = []
    nomes: list[str] = []

    def somar(arr: np.ndarray) -> np.ndarray:
        """Soma cumulativa com uma coluna zero a esquerda."""
        s = np.zeros((arr.shape[0], arr.shape[1] + 1), dtype=np.float64)
        np.cumsum(arr, axis=1, out=s[:, 1:])
        return s

    cs_prec = somar(prec)
    cs_umid = somar(umid)
    idx = np.arange(n_dias)

    for j in JANELAS_PRECIPITACAO:
        ini = np.maximum(idx - j + 1, 0)
        n = (idx - ini + 1).astype(np.float32)
        acum = (cs_prec[:, idx + 1] - cs_prec[:, ini]).astype(np.float32)
        feats.append(acum)
        nomes.append(f"prec_acum_{j}d")
        feats.append(acum / n)
        nomes.append(f"prec_media_{j}d")

    for j in JANELAS_UMIDADE:
        ini = np.maximum(idx - j + 1, 0)
        n = (idx - ini + 1).astype(np.float32)
        feats.append(((cs_umid[:, idx + 1] - cs_umid[:, ini]) / n).astype(np.float32))
        nomes.append(f"ur_media_{j}d")

    feats.append(umid.copy())
    nomes.append("ur_dia")
    feats.append(prec.copy())
    nomes.append("prec_dia")

    # minimo movel de umidade em 7 e 15 dias (proxy de estresse hidrico agudo)
    for j in (7, 15):
        feats.append(_min_movel(umid, j))
        nomes.append(f"ur_min_{j}d")

    # comprimento da estiagem: dias consecutivos com precipitacao < limiar
    feats.append(_dias_secos_consecutivos(prec, LIMIAR_DIA_CHUVOSO))
    nomes.append("dias_estiagem")

    # indice de seca acumulada estilo KBDI simplificado:
    # soma o deficit diario e zera proporcionalmente a chuva efetiva
    feats.append(_indice_seca(prec, umid))
    nomes.append("indice_seca")

    cubo = np.stack(feats, axis=2).astype(np.float32)
    return cubo, nomes


def _min_movel(arr: np.ndarray, janela: int) -> np.ndarray:
    """Minimo em janela causal [t-janela+1, t], vetorizado por deslocamento."""
    saida = arr.astype(np.float32, copy=True)
    for k in range(1, janela):
        deslocado = np.empty_like(saida)
        deslocado[:, :k] = arr[:, :1]
        deslocado[:, k:] = arr[:, :-k]
        np.minimum(saida, deslocado, out=saida)
    return saida


def _dias_secos_consecutivos(prec: np.ndarray, limiar: float) -> np.ndarray:
    """Numero de dias consecutivos, ate t inclusive, com precipitacao < limiar."""
    seco = prec < limiar
    saida = np.zeros(prec.shape, dtype=np.float32)
    corrente = np.zeros(prec.shape[0], dtype=np.float32)
    for t in range(prec.shape[1]):
        corrente = np.where(seco[:, t], corrente + 1.0, 0.0)
        saida[:, t] = corrente
    return saida


def _indice_seca(prec: np.ndarray, umid: np.ndarray) -> np.ndarray:
    """Indice de seca acumulada, inspirado no KBDI mas sem dados de temperatura.

    Acumula um deficit diario proporcional a (100 - UR) e desconta a chuva
    efetiva do dia. Nao e o KBDI oficial (que exige temperatura maxima e
    precipitacao media anual): e um proxy de memoria hidrica do combustivel,
    limitado a [0, 800] para manter a escala comparavel a do KBDI.
    """
    n_cel, n_dias = prec.shape
    saida = np.zeros((n_cel, n_dias), dtype=np.float32)
    q = np.zeros(n_cel, dtype=np.float32)
    for t in range(n_dias):
        deficit = np.maximum(100.0 - umid[:, t], 0.0) / 100.0 * 8.0
        chuva_efetiva = np.maximum(prec[:, t] - LIMIAR_DIA_CHUVOSO, 0.0) * 10.0
        q = np.clip(q + deficit - chuva_efetiva, 0.0, 800.0)
        saida[:, t] = q
    return saida


# --------------------------------------------------------- amostragem por dia


def amostrar_celula_dia(
    estatica: pd.DataFrame,
    cubo_clima: np.ndarray,
    nomes_clima: list[str],
    anos=geo.ANOS_CLIMA,
    negativos_por_positivo: int = 40,
    exclusao_dias: int = 45,
    semente: int = 42,
) -> pd.DataFrame:
    """Monta a tabela (celula, dia) com rotulo binario de ignicao.

    Positivo : par (celula, dia) em que a celula registrou cicatriz naquele dia.
    Negativo : par (celula, dia) sorteado na estacao de fogo do mesmo ano, com a
               celula sem cicatriz em uma janela de +-`exclusao_dias` daquele dia.
               A exclusao evita rotular como "sem fogo" um dia imediatamente
               vizinho a um incendio real, que e ambiguo por causa da incerteza
               na data de deteccao.

    O historico de fogo entra como TAXA sobre os anos anteriores disponiveis
    (`taxa_queima_hist`), nunca sobre o ano corrente ou futuros: sem vazamento.
    """
    rng = np.random.default_rng(semente)
    n_cel = len(estatica)
    cli_id = estatica["cli_id"].values
    tem_clima = cli_id >= 0

    dias_por_ano = {
        ano: estatica[f"dia_{ano}"].values.astype(np.int64) for ano in geo.ANOS_CICATRIZ
    }
    datas = geo.datas_clima()
    ano_da_banda = datas.astype("datetime64[Y]").astype(int) + 1970
    inicio_do_ano = {a: int(np.argmax(ano_da_banda == a)) for a in anos}

    blocos = []
    for ano in anos:
        dia_ano = dias_por_ano[ano]
        base = inicio_do_ano[ano]

        # ---------- positivos
        pos = np.nonzero(
            tem_clima
            & (dia_ano >= DIA_INICIO_ESTACAO)
            & (dia_ano <= DIA_FIM_ESTACAO)
        )[0]
        dia_pos = dia_ano[pos]

        # ---------- negativos
        n_neg = len(pos) * negativos_por_positivo
        cand_cel = rng.choice(np.nonzero(tem_clima)[0], size=n_neg, replace=True)
        cand_dia = rng.integers(DIA_INICIO_ESTACAO, DIA_FIM_ESTACAO + 1, size=n_neg)
        d_cel = dia_ano[cand_cel]
        ambiguo = (d_cel > 0) & (np.abs(d_cel - cand_dia) <= exclusao_dias)
        cand_cel, cand_dia = cand_cel[~ambiguo], cand_dia[~ambiguo]

        celula = np.concatenate([pos, cand_cel])
        dia = np.concatenate([dia_pos, cand_dia])
        rotulo = np.concatenate(
            [np.ones(len(pos), dtype=np.int8), np.zeros(len(cand_cel), dtype=np.int8)]
        )

        bloco = pd.DataFrame({"celula": celula, "ano": ano, "dia_ano": dia, "y": rotulo})
        bloco["banda"] = base + dia - 1  # dia 1 -> primeira banda do ano
        blocos.append(bloco)

    am = pd.concat(blocos, ignore_index=True)

    # ---------- preditores estaticos
    for c in COLUNAS_ESTATICAS:
        am[c] = estatica[c].values[am["celula"].values]

    # ---------- historico de fogo (somente anos anteriores)
    matriz_queima = np.column_stack(
        [estatica[f"queimou_{a}"].values for a in geo.ANOS_CICATRIZ]
    ).astype(np.float32)
    anos_arr = np.array(geo.ANOS_CICATRIZ)
    taxa = np.full(len(am), np.nan, dtype=np.float32)
    ult1 = np.full(len(am), np.nan, dtype=np.float32)
    for ano in anos:
        m = am["ano"].values == ano
        anteriores = anos_arr < ano
        if anteriores.any():
            sub = matriz_queima[:, anteriores]
            taxa_cel = sub.mean(axis=1)
            taxa[m] = taxa_cel[am["celula"].values[m]]
            ult1[m] = matriz_queima[:, anos_arr == ano - 1].ravel()[
                am["celula"].values[m]
            ]
    am["taxa_queima_hist"] = taxa
    am["queimou_ano_anterior"] = ult1

    # ---------- sazonalidade
    ang = 2 * np.pi * am["dia_ano"].values / 366.0
    am["dia_sin"] = np.sin(ang)
    am["dia_cos"] = np.cos(ang)

    # ---------- preditores climaticos
    ids = cli_id[am["celula"].values]
    bandas = am["banda"].values
    for k, nome in enumerate(nomes_clima):
        am[nome] = cubo_clima[ids, bandas, k]

    # ---------- contagio espacial (somente dias anteriores)
    cont = contagio_da_amostra(am, estatica)
    for c in cont.columns:
        am[c] = cont[c].values

    return am


# ------------------------------------------------------------ contagio espacial

RAIOS_CONTAGIO_M = (5_000, 15_000, 40_000)
JANELAS_CONTAGIO = (3, 7, 21)

NOMES_CONTAGIO = [
    f"fogo_{r//1000}km_{d}d" for r in RAIOS_CONTAGIO_M for d in JANELAS_CONTAGIO
] + ["dist_fogo_recente_m", "dias_desde_fogo_vizinho"]


def contagio(
    x_alvo: np.ndarray,
    y_alvo: np.ndarray,
    dia_alvo: np.ndarray,
    x_fogo: np.ndarray,
    y_fogo: np.ndarray,
    dia_fogo: np.ndarray,
) -> dict[str, np.ndarray]:
    """Preditores de contagio: fogo proximo nos dias ANTERIORES.

    Incendio na Amazonia ocorre em cluster espaco-temporal: uma queimada de
    manejo escapa e propaga para o entorno nos dias seguintes. Estes preditores
    capturam isso contando quantas celulas dentro de um raio R queimaram na
    janela de D dias que antecede o dia consultado.

    Sem vazamento: somente `atraso >= 1`, isto e, apenas deteccoes de dias
    ANTERIORES entram na conta. Fogo do proprio dia fica de fora. Isso e
    realista na operacao, porque as deteccoes de satelite de ontem ja estao
    disponiveis hoje.

    Todas as coordenadas em metros (UTM19S) e os dias em dia-do-ano.
    """
    from scipy.spatial import cKDTree

    n = len(x_alvo)
    saida = {nome: np.zeros(n, dtype=np.float32) for nome in NOMES_CONTAGIO}
    saida["dist_fogo_recente_m"][:] = np.nan
    saida["dias_desde_fogo_vizinho"][:] = np.nan
    if len(x_fogo) == 0:
        return saida

    r_max = max(RAIOS_CONTAGIO_M)
    arvore_fogo = cKDTree(np.column_stack([x_fogo, y_fogo]))
    arvore_alvo = cKDTree(np.column_stack([x_alvo, y_alvo]))
    vizinhos = arvore_alvo.query_ball_tree(arvore_fogo, r=r_max)

    tamanhos = np.fromiter((len(v) for v in vizinhos), dtype=np.int64, count=n)
    total = int(tamanhos.sum())
    if total == 0:
        return saida
    idx_alvo = np.repeat(np.arange(n), tamanhos)
    idx_fogo = np.fromiter(
        (j for v in vizinhos for j in v), dtype=np.int64, count=total
    )

    dist = np.hypot(
        x_alvo[idx_alvo] - x_fogo[idx_fogo], y_alvo[idx_alvo] - y_fogo[idx_fogo]
    )
    atraso = dia_alvo[idx_alvo] - dia_fogo[idx_fogo]
    anterior = atraso >= 1

    for r in RAIOS_CONTAGIO_M:
        no_raio = anterior & (dist <= r)
        for d in JANELAS_CONTAGIO:
            sel = no_raio & (atraso <= d)
            saida[f"fogo_{r//1000}km_{d}d"] = np.bincount(
                idx_alvo[sel], minlength=n
            ).astype(np.float32)

    # distancia ao fogo anterior mais proximo (janela mais larga) e a defasagem
    janela = anterior & (atraso <= max(JANELAS_CONTAGIO))
    if janela.any():
        ia, dd, at = idx_alvo[janela], dist[janela], atraso[janela]
        melhor = np.full(n, np.inf, dtype=np.float64)
        np.minimum.at(melhor, ia, dd)
        tem = np.isfinite(melhor)
        saida["dist_fogo_recente_m"][tem] = melhor[tem]
        lag = np.full(n, np.inf, dtype=np.float64)
        np.minimum.at(lag, ia, at.astype(np.float64))
        saida["dias_desde_fogo_vizinho"][tem] = lag[tem]
    return saida


def contagio_da_amostra(am: pd.DataFrame, estatica: pd.DataFrame) -> pd.DataFrame:
    """Aplica `contagio` ano a ano sobre a tabela (celula, dia)."""
    xg = estatica["x_g"].values
    yg = estatica["y_g2"].values
    cel = am["celula"].values
    blocos = []
    for ano, sub in am.groupby("ano", sort=True):
        dia_cel = estatica[f"dia_{ano}"].values.astype(np.int64)
        queimou = dia_cel > 0
        idx = sub.index.to_numpy()
        f = contagio(
            xg[cel[idx]], yg[cel[idx]], sub["dia_ano"].values.astype(np.int64),
            xg[queimou], yg[queimou], dia_cel[queimou],
        )
        blocos.append(pd.DataFrame(f, index=idx))
    return pd.concat(blocos).reindex(am.index)


def colunas_preditoras(am: pd.DataFrame, nomes_clima: list[str]) -> list[str]:
    """Lista de colunas usadas como X. Exclui identificadores e o rotulo."""
    extras = [
        "taxa_queima_hist",
        "queimou_ano_anterior",
        "dia_sin",
        "dia_cos",
        "dia_ano",
    ]
    return [
        c
        for c in COLUNAS_ESTATICAS + extras + nomes_clima + NOMES_CONTAGIO
        if c in am.columns
    ]
