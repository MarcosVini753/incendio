"""Geometria das grades e leitura dos GeoTIFF do projeto.

Fatos verificados nos arquivos (nao sao suposicoes):

centroides2003a2013.xlsx
    Grade regular em UTM zona 19S (validada contra o raster de cicatrizes:
    100% dos centroides caem dentro do raster e P(cicatriz 2010 | foco 2010)
    fica 21x acima da taxa base). Afim exata, residuo < 1e-4 m:
        x_g  = 892.980533 * X - 91431.715127
        y_g2 = 598.158593 * Y + 8690698.257560
    Celula = 892.98 x 598.16 m = 0.5341 km^2.
    307.410 celulas validas -> 164.295 km^2 (Acre = 164.123 km^2).

Cicatrizes_Incendio_Acre_2006_2016.tif
    897 x 1642, 11 bandas (ano_2006 .. ano_2016), int16, EPSG:4326,
    pixel 0.004491576 deg (~500 m), canto superior esquerdo
    (-73.98973838, -7.11465705).
    Valor 0 = nao queimou; valor > 0 = DIA DO ANO da deteccao (1..366).
    93% das deteccoes caem entre julho e outubro.

Precipitacao_13h_Diaria_* / Umidade_Relativa_Diaria_*
    41 x 75, 3652 bandas diarias de 2006-01-01 a 2015-12-31, float64,
    EPSG:4326, pixel 0.1 deg (~11.1 km), canto superior esquerdo
    (-74.04951513, -7.05044416).
    1700 das 3075 celulas sao NaN em todos os dias (fora do dominio);
    as 1375 celulas restantes nao tem NaN algum.
    ATENCAO: o nome do arquivo diz 2016 mas a ultima banda e 2015-12-31.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tifffile
from pyproj import Transformer

# ------------------------------------------------------------------ constantes

ARQ_CICATRIZES = "Cicatrizes_Incendio_Acre_2006_2016.tif"
ARQ_PRECIPITACAO = "Precipitacao_13h_Diaria_Acre_2006_2016.tif"
ARQ_UMIDADE = "Umidade_Relativa_Diaria_Acre_2006_2016.tif"
ARQ_CENTROIDES = "centroides2003a2013.xlsx"

EPSG_CENTROIDES = 31979  # SIRGAS 2000 / UTM zone 19S
ANOS_CICATRIZ = tuple(range(2006, 2017))
DATA_INICIO_CLIMA = np.datetime64("2006-01-01")
N_DIAS_CLIMA = 3652  # 2006-01-01 .. 2015-12-31
ANOS_CLIMA = tuple(range(2006, 2016))

# afim da grade de centroides em UTM19S (metros)
XG_A, XG_B = 892.980533, -91431.715127
YG_A, YG_B = 598.158593, 8690698.257560

# mapeamento ordinal de tipologia vegetal (mesmo do projeto original,
# ordenado por densidade historica de focos de calor por area)
VEG_ORDINAL = {
    "Áreas Antropizadas": 1,
    "FAP - Aluvial + Vs": 2,
    "FAP - Aluvial": 3,
    "FAP + FD": 4,
    "FD + FAP": 4,
    "FAB - Aluvial": 5,
    "FAP": 6,
    "FAB + FD": 7,
    "FD + FAB": 7,
    "FAB + FAP": 8,
    "FAP + FAB": 8,
    "FAP - Aluvial + Pab": 9,
    "Campinaranas": 10,
    "FAB + FAP + FD": 11,
    "FAP + FAB + FD": 11,
    "FAP + FD + FAB": 11,
    "FABD": 12,
    "FD": 13,
    "FD - Submontana": 14,
    "FAP + Pab": 15,
}


# -------------------------------------------------------------------- geometria


@dataclass(frozen=True)
class GradeRaster:
    """Grade regular norte-acima em graus decimais (EPSG:4326)."""

    n_lin: int
    n_col: int
    lon0: float  # borda oeste
    lat0: float  # borda norte
    passo: float  # graus por pixel

    def indices(self, lon, lat, recortar: bool = True):
        """Converte lon/lat em (linha, coluna). Fora da grade -> -1 se recortar=False."""
        col = np.floor((np.asarray(lon) - self.lon0) / self.passo).astype(np.int64)
        lin = np.floor((self.lat0 - np.asarray(lat)) / self.passo).astype(np.int64)
        if recortar:
            np.clip(col, 0, self.n_col - 1, out=col)
            np.clip(lin, 0, self.n_lin - 1, out=lin)
        else:
            fora = (col < 0) | (col >= self.n_col) | (lin < 0) | (lin >= self.n_lin)
            col[fora] = -1
            lin[fora] = -1
        return lin, col

    def centros(self):
        """Retorna (lat_centros[n_lin], lon_centros[n_col])."""
        lat = self.lat0 - (np.arange(self.n_lin) + 0.5) * self.passo
        lon = self.lon0 + (np.arange(self.n_col) + 0.5) * self.passo
        return lat, lon


def ler_grade(caminho: str) -> GradeRaster:
    """Le a geometria de um GeoTIFF sem decodificar pixels."""
    with tifffile.TiffFile(caminho) as tf:
        p = tf.pages[0]
        passo_x, passo_y, _ = p.tags[33550].value  # ModelPixelScale
        _, _, _, lon0, lat0, _ = p.tags[33922].value  # ModelTiepoint
        n_lin, n_col = p.tags[257].value, p.tags[256].value
    if not np.isclose(passo_x, passo_y):
        raise ValueError(f"{caminho}: pixel nao quadrado ({passo_x}, {passo_y})")
    return GradeRaster(n_lin, n_col, lon0, lat0, passo_x)


def nomes_bandas(caminho: str) -> list[str]:
    """Descricoes das bandas gravadas na tag GDAL_METADATA, em ordem de sample."""
    with tifffile.TiffFile(caminho) as tf:
        tag = tf.pages[0].tags.get(42112)
        bruto = None if tag is None else str(tag.value)
    if bruto is None:
        return []
    achados = re.findall(r'sample="(\d+)"[^>]*>([^<]+)<', bruto)
    return [nome for _, nome in sorted(achados, key=lambda t: int(t[0]))]


@functools.lru_cache(maxsize=1)
def _transformador():
    return Transformer.from_crs(EPSG_CENTROIDES, 4326, always_xy=True)


def utm_para_lonlat(x_g, y_g):
    """UTM19S -> (lon, lat) em graus WGS84."""
    return _transformador().transform(np.asarray(x_g), np.asarray(y_g))


@functools.lru_cache(maxsize=1)
def _transformador_inverso():
    return Transformer.from_crs(4326, EPSG_CENTROIDES, always_xy=True)


def lonlat_para_utm(lon, lat):
    """(lon, lat) WGS84 -> UTM19S."""
    return _transformador_inverso().transform(np.asarray(lon), np.asarray(lat))


def utm_para_indice_grade(x_g, y_g):
    """UTM19S -> indices (Y, X) da grade de centroides (arredondamento ao centro)."""
    xi = np.rint((np.asarray(x_g) - XG_B) / XG_A).astype(np.int64)
    yi = np.rint((np.asarray(y_g) - YG_B) / YG_A).astype(np.int64)
    return yi, xi


# ---------------------------------------------------------------------- leitura


def ler_cubo(caminho: str, dtype=None) -> np.ndarray:
    """Le um GeoTIFF multibanda como array (n_lin, n_col, n_bandas)."""
    arr = tifffile.imread(caminho)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr if dtype is None else arr.astype(dtype)


def ler_centroides(caminho: str = ARQ_CENTROIDES) -> pd.DataFrame:
    """Le a planilha de centroides, limpa e acrescenta lon/lat e veg ordinal.

    Limpezas aplicadas:
      * descarta a linha com x_g = y_g2 = 0 (registro corrompido);
      * RASTERVALU == -9999 -> NaN (sem dado de altitude, 108 celulas);
      * VEG_TIP textual -> escala ordinal; categoria ausente -> NaN.
    """
    d = pd.read_excel(caminho, engine="openpyxl")
    n0 = len(d)
    d = d[(d["x_g"] != 0) | (d["y_g2"] != 0)].copy()
    d = d[d["y_g2"] > 1e6].copy()

    d["veg"] = d["VEG_TIP"].map(VEG_ORDINAL).astype("float64")
    d["altitude"] = d["RASTERVALU"].astype("float64")
    d.loc[d["altitude"] <= -9000, "altitude"] = np.nan
    d = d.rename(columns={"Distance": "dist_estrada", "Distance_1": "dist_agua"})

    lon, lat = utm_para_lonlat(d["x_g"].values, d["y_g2"].values)
    d["lon"], d["lat"] = lon, lat

    d = d.reset_index(drop=True)
    d.attrs["linhas_descartadas"] = n0 - len(d)
    return d


def clima_valido(caminho: str) -> tuple[np.ndarray, np.ndarray]:
    """Le um cubo climatico e devolve (series[n_validas, n_dias], mascara[n_lin, n_col]).

    Celulas fora do dominio sao NaN em todos os dias e ficam de fora.
    """
    cubo = ler_cubo(caminho, dtype=np.float32)
    mascara = np.isfinite(cubo).any(axis=2)
    series = cubo[mascara]  # (n_validas, n_dias)
    del cubo
    return series, mascara


def datas_clima() -> np.ndarray:
    """Vetor de datas (datetime64[D]) das bandas diarias."""
    return DATA_INICIO_CLIMA + np.arange(N_DIAS_CLIMA)
