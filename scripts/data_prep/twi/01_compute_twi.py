"""Calcula TWI raster e estatística zonal por ottobacia para o piloto.

Pré-requisitos (já copiados de Neural-HBV-TTD):
  - data/raw/slope.tif
  - data/raw/flow_accumulation.tif
  - data/processed/ottobacias.gpkg

Outputs:
  - data/processed/twi.tif                  raster TWI (30 m)
  - data/processed/twi_per_ottobacia.parquet  CDF discretizada + estatísticas
"""

from __future__ import annotations

# PROJ_LIB fix DEVE vir antes de qualquer import geo (rasterio/geopandas/pyproj).
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    _venv_proj = Path(sys.prefix) / "Lib" / "site-packages" / "pyproj" / "proj_dir" / "share" / "proj"
    if _venv_proj.exists():
        os.environ["PROJ_DATA"] = str(_venv_proj)
        os.environ["PROJ_LIB"] = str(_venv_proj)

import io
import logging

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import geopandas as gpd
import numpy as np
import rasterio
from rich.logging import RichHandler

from topmodel_hud_lstm.physics.twi import compute_twi_raster, zonal_stats_twi

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_path=False)],
)
log = logging.getLogger("compute_twi")


def main() -> None:
    slope_path = DATA_RAW / "slope.tif"
    flow_acc_path = DATA_RAW / "flow_accumulation.tif"
    otto_path = DATA_PROCESSED / "ottobacias.gpkg"

    for p in (slope_path, flow_acc_path, otto_path):
        if not p.exists():
            raise FileNotFoundError(f"Arquivo esperado não existe: {p}")

    log.info("Calculando TWI...")
    twi, profile = compute_twi_raster(
        slope_path=slope_path,
        flow_acc_path=flow_acc_path,
        cell_size_m=30.0,
        slope_unit="degrees",
    )
    valid = ~np.isnan(twi)
    log.info(
        "TWI: shape=%s, valid=%d (%.1f%%), min=%.2f, p50=%.2f, p95=%.2f, max=%.2f",
        twi.shape,
        valid.sum(),
        valid.mean() * 100,
        np.nanmin(twi),
        np.nanpercentile(twi, 50),
        np.nanpercentile(twi, 95),
        np.nanmax(twi),
    )

    twi_out = DATA_PROCESSED / "twi.tif"
    with rasterio.open(twi_out, "w", **profile) as dst:
        dst.write(twi, 1)
    log.info("Raster TWI salvo: %s", twi_out)

    log.info("Carregando ottobacias...")
    otto_gdf = gpd.read_file(otto_path)
    log.info("%d ottobacias carregadas (CRS=%s)", len(otto_gdf), otto_gdf.crs)

    id_col_candidates = ["cocursodag", "cobacia", "id", "OBJECTID"]
    id_col = next((c for c in id_col_candidates if c in otto_gdf.columns), None)
    log.info("Coluna de ID detectada: %s", id_col or "(nenhuma — usando índice posicional)")

    log.info("Calculando estatística zonal...")
    df = zonal_stats_twi(
        twi_raster=twi,
        profile=profile,
        ottobacias_gdf=otto_gdf,
        n_bins=20,
        bin_range=(-5.0, 25.0),
        id_col=id_col,
    )

    log.info(
        "Stats: %d ottobacias com TWI. Mean: p10=%.2f, p50=%.2f, p90=%.2f",
        len(df),
        df["twi_mean"].quantile(0.1),
        df["twi_mean"].median(),
        df["twi_mean"].quantile(0.9),
    )

    out_path = DATA_PROCESSED / "twi_per_ottobacia.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Salvo parquet: %s (%d linhas, %d colunas)", out_path, len(df), len(df.columns))

    # CSV redundante — pyarrow tem segfault com torch+h5py no Windows
    csv_path = DATA_PROCESSED / "twi_per_ottobacia.csv"
    df.to_csv(csv_path, index=False)
    log.info("Salvo csv:     %s", csv_path)

    # NPZ com tensores prontos pro Topmodel (mais robusto que parquet/CSV em runtime)
    bin_cols = sorted([c for c in df.columns if c.startswith("twi_bin_")])
    npz_path = DATA_PROCESSED / "twi_attrs.npz"
    np.savez_compressed(
        npz_path,
        ottobacia_idx=df["ottobacia_idx"].to_numpy(dtype=np.int64),
        twi_dist=df[bin_cols].to_numpy(dtype=np.float32),
        twi_mean=df["twi_mean"].to_numpy(dtype=np.float32),
        twi_centers=np.linspace(-5.0 + (25.0 - -5.0) / (2 * 20), 25.0 - (25.0 - -5.0) / (2 * 20), 20).astype(np.float32),
    )
    log.info("Salvo npz:     %s", npz_path)


if __name__ == "__main__":
    main()
