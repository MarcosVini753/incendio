"""Pipeline de modelagem de risco de incendio florestal no Acre.

Modulos
-------
geo      : geometria das grades (centroides UTM19S <-> rasters WGS84) e leitura dos GeoTIFF
dados    : construcao das tabelas de treino (estatica, cicatrizes, clima, amostragem)
modelos  : FuzzyKNN corrigido, zoo de modelos e metricas de avaliacao
"""

__all__ = ["geo", "dados", "modelos"]
