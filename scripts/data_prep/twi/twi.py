"""TWI (Topographic Wetness Index) e estatística zonal por ottobacia.

TWI = ln(a / tan β), onde:
  a  = área de contribuição específica (m²/m) = flow_acc × cell_size
  β  = declividade local

Em bacia única, o TWI vira atributo distribuído. Em multi-bacia (CABra-H), entra como
CDF discretizada por bacia, alimentando o encoder MLP regional.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

CELL_SIZE_M_DEFAULT = 30.0  # ANADEM 30m
SLOPE_TAN_MIN = 1e-3  # ~0.057°, evita TWI = +inf em terreno plano
N_BINS_DEFAULT = 20
BIN_RANGE_DEFAULT = (-5.0, 25.0)  # cobre quase tudo de TWI físico
DEFAULT_CRS = "EPSG:31983"  # SIRGAS 2000 / UTM 23S (sub-bacia do Rio Preto)


def compute_twi_raster(
    slope_path: Path | str,
    flow_acc_path: Path | str,
    cell_size_m: float = CELL_SIZE_M_DEFAULT,
    slope_unit: str = "degrees",
) -> tuple[np.ndarray, dict]:
    """Calcula TWI por célula a partir de slope e flow_accumulation rasters.

    Args:
        slope_path: caminho do raster de slope.
        flow_acc_path: caminho do raster de flow_accumulation (em células).
        cell_size_m: tamanho da célula em metros (default 30 para ANADEM).
        slope_unit: 'degrees', 'percent', ou 'tan' (m/m).

    Returns:
        (twi_array, profile) — TWI float32 e perfil rasterio compatível com nodata=NaN.
    """
    with rasterio.open(slope_path) as src:
        slope = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        slope_nodata = src.nodata

    with rasterio.open(flow_acc_path) as src:
        flow_acc = src.read(1).astype(np.float32)
        flow_acc_nodata = src.nodata

    if slope_unit == "degrees":
        tan_beta = np.tan(np.deg2rad(slope))
    elif slope_unit == "percent":
        tan_beta = slope / 100.0
    elif slope_unit == "tan":
        tan_beta = slope
    else:
        raise ValueError(f"slope_unit inválido: {slope_unit}")

    tan_beta = np.maximum(tan_beta, SLOPE_TAN_MIN)

    # a = flow_acc × cell_area / cell_width = flow_acc × cell_size (em m)
    a = np.maximum(flow_acc, 1.0) * cell_size_m

    twi = np.log(a / tan_beta).astype(np.float32)

    invalid = np.zeros_like(twi, dtype=bool)
    if slope_nodata is not None:
        invalid |= slope == slope_nodata
    if flow_acc_nodata is not None:
        invalid |= flow_acc == flow_acc_nodata
    invalid |= ~np.isfinite(slope) | ~np.isfinite(flow_acc)

    twi[invalid] = np.nan

    profile.update(dtype=rasterio.float32, nodata=float("nan"), count=1, compress="deflate")
    return twi, profile


def zonal_stats_twi(
    twi_raster: np.ndarray,
    profile: dict,
    ottobacias_gdf: gpd.GeoDataFrame,
    n_bins: int = N_BINS_DEFAULT,
    bin_range: tuple[float, float] = BIN_RANGE_DEFAULT,
    id_col: str | None = None,
    assume_crs: str = DEFAULT_CRS,
) -> pd.DataFrame:
    """Estatística zonal de TWI por ottobacia.

    Para cada bacia retorna estatísticas resumo + histograma normalizado (CDF
    discretizada) que serve de atributo regional para o Topmodel.

    Args:
        twi_raster: TWI em float32 (NaN para nodata).
        profile: perfil rasterio (precisa de transform e crs).
        ottobacias_gdf: GeoDataFrame de ottobacias (será reprojetado se necessário).
        n_bins: número de bins do histograma.
        bin_range: (min, max) do TWI para o histograma.
        id_col: nome da coluna de ID na geodataframe (se None, usa o índice posicional).
        assume_crs: CRS a assumir se o raster vier com LOCAL_CS sem código EPSG.

    Returns:
        DataFrame com 1 linha por ottobacia.
    """
    from pyproj import CRS as PyprojCRS  # noqa: PLC0415  evita import top-level

    transform = profile["transform"]
    height, width = twi_raster.shape
    raster_crs = profile["crs"]

    # Rasters da quali vieram com LOCAL_CS — pyproj não consegue transformar.
    # Assume o CRS conhecido (EPSG:31983, UTM 23S).
    raster_crs_resolved = raster_crs
    if raster_crs is None or "LOCAL_CS" in str(raster_crs.to_wkt()):
        raster_crs_resolved = PyprojCRS.from_string(assume_crs)

    if ottobacias_gdf.crs != raster_crs_resolved:
        ottobacias_gdf = ottobacias_gdf.to_crs(raster_crs_resolved)

    shapes = [(geom, idx) for idx, geom in enumerate(ottobacias_gdf.geometry)]
    otto_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=-1,
        dtype=np.int32,
    )

    bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    valid_mask = ~np.isnan(twi_raster)

    rows = []
    for idx, otto_row in ottobacias_gdf.iterrows():
        mask = (otto_raster == idx) & valid_mask
        twi_values = twi_raster[mask]

        if len(twi_values) == 0:
            continue

        hist, _ = np.histogram(twi_values, bins=bins)
        total = hist.sum()
        hist_norm = hist / total if total > 0 else hist.astype(np.float64)

        record: dict = {
            "ottobacia_idx": idx,
            "twi_mean": float(twi_values.mean()),
            "twi_std": float(twi_values.std()),
            "twi_min": float(twi_values.min()),
            "twi_p10": float(np.percentile(twi_values, 10)),
            "twi_p50": float(np.percentile(twi_values, 50)),
            "twi_p90": float(np.percentile(twi_values, 90)),
            "twi_max": float(twi_values.max()),
            "n_cells": int(len(twi_values)),
        }
        if id_col is not None and id_col in ottobacias_gdf.columns:
            record["ottobacia_id"] = otto_row[id_col]

        for i, h in enumerate(hist_norm):
            record[f"twi_bin_{i:02d}"] = float(h)

        rows.append(record)

    return pd.DataFrame(rows)


def histogram_centers(
    n_bins: int = N_BINS_DEFAULT,
    bin_range: tuple[float, float] = BIN_RANGE_DEFAULT,
) -> np.ndarray:
    """Retorna o centro de cada bin do histograma usado em zonal_stats_twi."""
    edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2.0
