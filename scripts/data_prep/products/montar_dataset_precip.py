#!/usr/bin/env python
"""Consolida CSVs de precip por ottobacia (export GEE) em dataset_<cod>_<produto>.h5,
GENERICO p/ as 2 bacias da big ablacao. Copia o .h5 base (MERGE) da bacia e SUBSTITUI
so a precipitation pelo produto novo => ablacao de forcante limpa (Tc, vazao, PET,
timestamps, ottobacias identicos).

Alinhamento por cobacia:
  encruzilhada -> cobacia vem de ottobacia/cobacia do proprio .h5
  md           -> .h5 nao tem cobacia; ordem identica ao gpkg (validado por area,
                  validar_ottos_md.py: max|dArea|=0) -> cobacia = gpkg na ordem do gpkg

  era5land: total_precipitation_hourly (m) -> *1000 = mm/h | horario
  imerg:    precipitation (mm/hr, 30min)   -> agrega 30min->1h (media = mm/h)

Saida: big_ablacao/data/dataset_<cod>_<produto>.h5

Uso:  uv run python big_ablacao/scripts/montar_dataset_precip.py --bacia-id md --produto era5land
"""

from __future__ import annotations

import argparse
import glob
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

DRIVE = Path(r"D:\TTD_SCS_LSTM\encruzilhada\data\precip_csv")  # onde baixar_csvs_drive.py salva
OUT = Path(r"D:\TTD_SCS_LSTM\big_ablacao\data")

BACIAS = {
    "encruzilhada": {
        "cod": "71350001",
        "src": r"D:\TTD_SCS_LSTM\encruzilhada\data\dataset_71350001_manual.h5",
        "gpkg": None,  # cobacia no proprio .h5
        "suffix": "",  # folder encruz_precip_<prod>, prefixo <prod>_otto
    },
    "md": {
        "cod": "58585000",
        "src": r"D:\TTD_SCS_LSTM\data\processed\dataset_v2.h5",
        "gpkg": r"D:\TTD_SCS_LSTM\data\processed\ottobacias_tc.gpkg",  # ordem == .h5 (validado)
        "suffix": "_md",
    },
}
PRODS = {
    "era5land": {"mult": 1000.0, "agg_hour": False, "n_esp": 12},
    "imerg":    {"mult": 1.0,    "agg_hour": True,  "n_esp": 24},
}


def norm_cod(x) -> str:
    s = (x.decode() if isinstance(x, bytes) else str(x)).strip()
    return s[:-2] if s.endswith(".0") else s


def cob_order_da_bacia(cfg) -> list[str]:
    """Ordem das cobacias correspondente as ottos do .h5 (eixo N da precipitation)."""
    if cfg["gpkg"]:
        g = gpd.read_file(cfg["gpkg"])
        col = next(c for c in g.columns if c.lower().startswith("cobacia"))
        return [norm_cod(x) for x in g[col].values]  # ordem do gpkg == ordem do .h5 (validado)
    with h5py.File(cfg["src"], "r") as f:
        return [norm_cod(x) for x in f["ottobacia/cobacia"][:]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bacia-id", choices=list(BACIAS), required=True)
    ap.add_argument("--produto", choices=list(PRODS), required=True)
    args = ap.parse_args()
    cfg, p = BACIAS[args.bacia_id], PRODS[args.produto]
    OUT.mkdir(parents=True, exist_ok=True)

    folder = f"encruz_precip_{args.produto}{cfg['suffix']}"
    prefix = f"{args.produto}{cfg['suffix']}_otto"
    csv_dir = DRIVE / folder
    dst = OUT / f"dataset_{cfg['cod']}_{args.produto}.h5"

    csvs = sorted(glob.glob(str(csv_dir / f"{prefix}_*.csv")))
    print(f"bacia {args.bacia_id} | produto {args.produto} | CSVs: {len(csvs)}/{p['n_esp']}  ({csv_dir})")
    if not csvs:
        print("  [ERRO] nenhum CSV — baixe com baixar_csvs_drive.py primeiro."); return 1
    if len(csvs) < p["n_esp"]:
        print("  [AGUARDANDO] faltam chunks; consolido parcial p/ diagnostico, NAO gravo o .h5.")

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["cob"] = df["cob"].map(norm_cod)
    df["val"] = pd.to_numeric(df["mean"], errors="coerce").fillna(0.0) * p["mult"]
    if p["agg_hour"]:
        df["ts"] = df["ts"].dt.floor("h")
        df = df.groupby(["ts", "cob"], as_index=False)["val"].mean()
    piv = df.pivot_table(index="ts", columns="cob", values="val", aggfunc="mean")
    print(f"pivot: {piv.shape[0]} h x {piv.shape[1]} ottos | {piv.index.min()} -> {piv.index.max()}")

    cob_order = cob_order_da_bacia(cfg)
    with h5py.File(cfg["src"], "r") as f:
        splits = {s: f[f"{s}/timestamps"][:].astype("int64") for s in ("train", "val", "test")}
        n_otto = f["train/precipitation"].shape[1]
    assert len(cob_order) == n_otto, f"cob_order={len(cob_order)} != n_otto={n_otto}"

    faltam = [c for c in cob_order if c not in piv.columns]
    if faltam:
        print(f"  AVISO: {len(faltam)}/{len(cob_order)} cobacias ausentes no pivot (-> 0): {faltam[:3]}...")

    precip = {}
    for s, ts in splits.items():
        idx = pd.to_datetime(ts, unit="s", utc=True)
        sub = piv.reindex(index=idx, columns=cob_order)
        nanf = float(sub.isna().mean().mean())
        arr = np.nan_to_num(sub.to_numpy(np.float32), nan=0.0)
        precip[s] = arr
        print(f"  [{s}] {arr.shape} | NaN reindex {100*nanf:.1f}% (zerado) | "
              f"mean {arr.mean():.4f} mm/h | max {arr.max():.1f}")

    if len(csvs) < p["n_esp"]:
        print("\n[parcial] nao gravei o .h5."); return 0

    shutil.copy(cfg["src"], dst)
    with h5py.File(dst, "a") as f:
        for s, arr in precip.items():
            del f[f"{s}/precipitation"]
            f.create_dataset(f"{s}/precipitation", data=arr, compression="gzip")
        f.attrs["precip_fonte"] = f"{args.produto} GEE reduceRegions/ottobacia ({args.bacia_id})"
    with h5py.File(cfg["src"], "r") as a, h5py.File(dst, "r") as b:
        print(f"\n=== {dst.name} ===")
        for s in ("train", "val", "test"):
            print(f"  [{s}] precip mean: base {np.nanmean(a[f'{s}/precipitation'][:]):.4f} | "
                  f"{args.produto} {np.nanmean(b[f'{s}/precipitation'][:]):.4f} mm/h")
    print(f"  -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
