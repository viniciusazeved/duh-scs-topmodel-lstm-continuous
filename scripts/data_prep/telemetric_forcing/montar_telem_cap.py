#!/usr/bin/env python
"""Copia ISOLADA de montar_dataset_telem.py (md) que salva numa pasta de analise,
SEM tocar no dataset oficial. So muda o destino e parametriza o --cap.

Uso: uv run python outputs/qc_chuva_analise/montar_telem_cap.py --cap 305
"""
from __future__ import annotations
import argparse, shutil, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from shapely.geometry import Point

POSTOS = Path(r"D:\TTD_SCS_LSTM\big_ablacao\data\postos")
OUT = Path(r"D:\TTD_SCS_LSTM\outputs\qc_chuva_analise")   # <- isolado, nao contamina
CFG = {
    "cod": "58585000",
    "src": r"D:\TTD_SCS_LSTM\data\processed\dataset_v2.h5",
    "ottos": r"D:\TTD_SCS_LSTM\data\processed\ottobacias_tc.gpkg",
    "epsg_m": 31983,
}


def norm_cod(x) -> str:
    s = (x.decode() if isinstance(x, bytes) else str(x)).strip()
    return s[:-2] if s.endswith(".0") else s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--dmin", type=float, default=50.0)
    ap.add_argument("--cap", type=float, default=305.0)
    args = ap.parse_args()
    cfg = CFG
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"dataset_{cfg['cod']}_telem_cap{int(args.cap)}.h5"

    ot = gpd.read_file(cfg["ottos"]).to_crs(cfg["epsg_m"])
    colcob = next(c for c in ot.columns if c.lower().startswith("cobacia"))
    ot["cob"] = ot[colcob].map(norm_cod)
    cen = ot.set_index("cob").geometry.centroid

    with h5py.File(cfg["src"], "r") as f:
        cob_order = list(ot["cob"])
        splits = {s: f[f"{s}/timestamps"][:].astype("int64") for s in ("train", "val", "test")}
        n_otto = f["train/precipitation"].shape[1]
    assert len(cob_order) == n_otto, f"cob_order={len(cob_order)} != n_otto={n_otto}"
    C = np.array([[cen[c].x, cen[c].y] for c in cob_order])

    ser = pd.read_parquet(POSTOS / "md" / "postos_telem_series.parquet")
    ser["codigo"] = ser.codigo.astype(str).str.zfill(8)
    ser["data_hora"] = pd.to_datetime(ser.data_hora, utc=True)
    meta = pd.read_parquet(POSTOS / "md" / "postos_meta.parquet")
    meta["codigo"] = meta.codigo.astype(str).str.zfill(8)
    cods = sorted(ser.codigo.unique())
    mm = meta.set_index("codigo").reindex(cods)
    Pxy = gpd.GeoSeries([Point(lo, la) for lo, la in zip(mm.lon, mm.lat)], crs=4326).to_crs(cfg["epsg_m"])
    Px = np.array([p.x for p in Pxy]); Py = np.array([p.y for p in Pxy])
    print(f"md: {len(cob_order)} ottos | {len(cods)} postos telem | cap={args.cap}")

    piv = ser.pivot_table(index="data_hora", columns="codigo", values="chuva_mm", aggfunc="sum").reindex(columns=cods)
    M = piv.to_numpy(np.float64).copy()
    n_out = int(np.nansum(M > args.cap))
    M[M > args.cap] = np.nan
    print(f"  outliers > {args.cap} mm/h -> NaN: {n_out}")
    mask = (~np.isnan(M)).astype(np.float64)
    M0 = np.nan_to_num(M, nan=0.0)

    dx = C[:, 0:1] - Px[None, :]; dy = C[:, 1:2] - Py[None, :]
    D = np.maximum(np.sqrt(dx**2 + dy**2), args.dmin)
    W = 1.0 / D**args.power

    num = M0 @ W.T; den = mask @ W.T
    chuva = np.where(den > 0, num / np.where(den == 0, 1, den), 0.0)
    chuva_df = pd.DataFrame(chuva, index=piv.index, columns=cob_order)

    precip = {}
    for s, ts in splits.items():
        idx = pd.to_datetime(ts, unit="s", utc=True)
        sub = chuva_df.reindex(index=idx)
        arr = np.nan_to_num(sub.to_numpy(np.float32), nan=0.0)
        precip[s] = arr
        print(f"  [{s}] {arr.shape} | mean {arr.mean():.4f} | max {arr.max():.1f} mm/h")

    shutil.copy(cfg["src"], dst)
    with h5py.File(dst, "a") as f:
        for s, arr in precip.items():
            del f[f"{s}/precipitation"]
            f.create_dataset(f"{s}/precipitation", data=arr, compression="gzip")
        f.attrs["precip_fonte"] = f"telem IDW(p={args.power}) dinamico/hora | cap={args.cap} mm/h"
    print(f"  -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
