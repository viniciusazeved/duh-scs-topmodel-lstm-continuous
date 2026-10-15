#!/usr/bin/env python
"""Monta dataset_<cod>_telem.h5 com chuva dos postos TELEMETRICOS interpolada por
ottobacia via IDW DINAMICO POR HORA.

IDW dinamico:
  peso w_op = 1/d_op^2 (centroide da ottobacia o -> posto p, distancia metrica).
  A cada hora t, so os postos COM DADO entram (pesos renormalizados):
    chuva_ot = sum_p(chuva_pt * w_op * presente_pt) / sum_p(w_op * presente_pt)
  Vetorizado: num = M0 @ W.T ; den = mask @ W.T ; chuva = num/den.

Copia o .h5 base (MERGE) e substitui a precipitation; reindexa nos timestamps de
vazao (grade identica). Saida: big_ablacao/data/dataset_<cod>_telem.h5

Uso (env TTD): uv run python big_ablacao/scripts/montar_dataset_telem.py --bacia-id md
"""
from __future__ import annotations

import argparse
import shutil
import sys
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
OUT = Path(r"D:\TTD_SCS_LSTM\big_ablacao\data")

BACIAS = {
    "encruzilhada": {
        "cod": "71350001",
        "src": r"D:\TTD_SCS_LSTM\encruzilhada\data\dataset_71350001_manual.h5",
        "ottos": r"D:\TTD_SCS_LSTM\data\manual_71350001\ottobacias_71350001.shp",
        "epsg_m": 31982, "cob_from": "h5",
    },
    "md": {
        "cod": "58585000",
        "src": r"D:\TTD_SCS_LSTM\data\processed\dataset_v2.h5",
        "ottos": r"D:\TTD_SCS_LSTM\data\processed\ottobacias_tc.gpkg",
        "epsg_m": 31983, "cob_from": "gpkg",
    },
}


def norm_cod(x) -> str:
    s = (x.decode() if isinstance(x, bytes) else str(x)).strip()
    return s[:-2] if s.endswith(".0") else s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bacia-id", choices=list(BACIAS), required=True)
    ap.add_argument("--power", type=float, default=2.0, help="expoente do IDW (default 2)")
    ap.add_argument("--dmin", type=float, default=50.0, help="distancia minima (m) p/ evitar peso infinito")
    ap.add_argument("--cap", type=float, default=130.0, help="chuva horaria > cap (mm/h) -> NaN (outlier de posto)")
    args = ap.parse_args()
    cfg = BACIAS[args.bacia_id]
    dst = OUT / f"dataset_{cfg['cod']}_telem.h5"

    # 1. ottobacias -> centroides por cobacia (metrico)
    ot = gpd.read_file(cfg["ottos"]).to_crs(cfg["epsg_m"])
    colcob = next(c for c in ot.columns if c.lower().startswith("cobacia"))
    ot["cob"] = ot[colcob].map(norm_cod)
    cen = ot.set_index("cob").geometry.centroid

    # 2. ordem das ottos (eixo N da precipitation no .h5)
    with h5py.File(cfg["src"], "r") as f:
        if cfg["cob_from"] == "h5":
            cob_order = [norm_cod(x) for x in f["ottobacia/cobacia"][:]]
        else:
            cob_order = list(ot["cob"])  # ordem gpkg == ordem .h5 (validado)
        splits = {s: f[f"{s}/timestamps"][:].astype("int64") for s in ("train", "val", "test")}
        n_otto = f["train/precipitation"].shape[1]
    assert len(cob_order) == n_otto, f"cob_order={len(cob_order)} != n_otto={n_otto}"
    falta = [c for c in cob_order if c not in cen.index]
    assert not falta, f"{len(falta)} cobacias sem centroide: {falta[:3]}"
    C = np.array([[cen[c].x, cen[c].y] for c in cob_order])  # (O,2)

    # 3. postos telemetricos: serie horaria + coords
    ser = pd.read_parquet(POSTOS / args.bacia_id / "postos_telem_series.parquet")
    ser["codigo"] = ser.codigo.astype(str).str.zfill(8)
    ser["data_hora"] = pd.to_datetime(ser.data_hora, utc=True)
    meta = pd.read_parquet(POSTOS / args.bacia_id / "postos_meta.parquet")
    meta["codigo"] = meta.codigo.astype(str).str.zfill(8)
    cods = sorted(ser.codigo.unique())
    mm = meta.set_index("codigo").reindex(cods)
    Pxy = gpd.GeoSeries([Point(lo, la) for lo, la in zip(mm.lon, mm.lat)], crs=4326).to_crs(cfg["epsg_m"])
    Px = np.array([p.x for p in Pxy]); Py = np.array([p.y for p in Pxy])  # (P,)
    print(f"{args.bacia_id}: {len(cob_order)} ottos | {len(cods)} postos telem")

    # 4. pivot chuva horaria (T, P) na ordem cods
    piv = ser.pivot_table(index="data_hora", columns="codigo", values="chuva_mm", aggfunc="sum").reindex(columns=cods)
    M = piv.to_numpy(np.float64).copy()       # (T,P) com NaN onde falta
    n_out = int(np.nansum(M > args.cap))
    M[M > args.cap] = np.nan                  # outlier de posto -> invalido (IDW ignora na hora)
    print(f"  outliers > {args.cap} mm/h zerados (NaN): {n_out}")
    mask = (~np.isnan(M)).astype(np.float64)  # presenca
    M0 = np.nan_to_num(M, nan=0.0)

    # 5. pesos IDW (O,P) = 1/d^p
    dx = C[:, 0:1] - Px[None, :]; dy = C[:, 1:2] - Py[None, :]
    D = np.sqrt(dx**2 + dy**2)
    D = np.maximum(D, args.dmin)
    W = 1.0 / D**args.power                  # (O,P)

    # 6. IDW dinamico por hora (vetorizado)
    num = M0 @ W.T          # (T,O)
    den = mask @ W.T        # (T,O) soma dos pesos dos postos presentes
    chuva = np.where(den > 0, num / np.where(den == 0, 1, den), 0.0)  # (T,O)
    sem_posto = int((den <= 0).all(axis=1).sum())
    chuva_df = pd.DataFrame(chuva, index=piv.index, columns=cob_order)
    print(f"  IDW p={args.power} dmin={args.dmin}m | serie {piv.index.min()} -> {piv.index.max()} | "
          f"{sem_posto} horas sem nenhum posto")

    # 7. reindex nos timestamps de vazao + grava
    precip = {}
    for s, ts in splits.items():
        idx = pd.to_datetime(ts, unit="s", utc=True)
        sub = chuva_df.reindex(index=idx)
        nanf = float(sub.isna().mean().mean())
        arr = np.nan_to_num(sub.to_numpy(np.float32), nan=0.0)
        precip[s] = arr
        print(f"  [{s}] {arr.shape} | NaN reindex {100*nanf:.1f}% (zerado) | "
              f"mean {arr.mean():.4f} mm/h | max {arr.max():.1f}")

    shutil.copy(cfg["src"], dst)
    with h5py.File(dst, "a") as f:
        for s, arr in precip.items():
            del f[f"{s}/precipitation"]
            f.create_dataset(f"{s}/precipitation", data=arr, compression="gzip")
        f.attrs["precip_fonte"] = f"telemetria ANA IDW(p={args.power}) dinamico/hora por ottobacia ({args.bacia_id})"
    with h5py.File(cfg["src"], "r") as a, h5py.File(dst, "r") as b:
        print(f"\n=== {dst.name} ===")
        for s in ("train", "val", "test"):
            print(f"  [{s}] precip mean: MERGE {np.nanmean(a[f'{s}/precipitation'][:]):.4f} | "
                  f"telem {np.nanmean(b[f'{s}/precipitation'][:]):.4f} mm/h")
    print(f"  -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
