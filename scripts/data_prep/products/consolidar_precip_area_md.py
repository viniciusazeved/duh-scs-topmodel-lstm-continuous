"""Consolida os CSVs AREA-PONDERADA (reduceRegions polygon, _md_area) de era5land/imerg
em dataset_58585000_<prod>.h5, MESMO metodo do MERGE (media areal + fallback vizinho).

Base = o .h5 CENTROIDE guardado em data/_reagregar_area/ (ja tem pet/vazao/timestamps/
estaticos). Substitui SO a precip pela area-ponderada; onde a area der NaN (ottobacia
sub-pixel sem subpixel no reduceRegions), usa o valor CENTROIDE da base = fallback do
proprio metodo do merge (numericamente identico, area~=centroide medido r=1.0).

Uso: uv run python big_ablacao/scripts/consolidar_precip_area_md.py --produto era5land
"""
from __future__ import annotations
import argparse, glob, shutil, sys
from pathlib import Path
import geopandas as gpd, h5py, numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DRIVE = Path(r"D:\TTD_SCS_LSTM\encruzilhada\data\precip_csv")
OUT = Path(r"D:\TTD_SCS_LSTM\big_ablacao\data")
HOLD = OUT / "_reagregar_area"                       # .h5 centroide guardados (base + fallback)
GPKG = r"D:\TTD_SCS_LSTM\data\processed\ottobacias_tc.gpkg"
COD = "58585000"
PRODS = {"era5land": {"mult": 1000.0, "agg_hour": False, "n_esp": 20},
         "imerg":    {"mult": 1.0,    "agg_hour": True,  "n_esp": 30}}


def norm_cod(x) -> str:
    s = (x.decode() if isinstance(x, bytes) else str(x)).strip()
    return s[:-2] if s.endswith(".0") else s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--produto", choices=list(PRODS), required=True)
    args = ap.parse_args(); p = PRODS[args.produto]
    base = HOLD / f"dataset_{COD}_{args.produto}.h5"           # centroide (base + fallback)
    dst = OUT / f"dataset_{COD}_{args.produto}.h5"
    csv_dir = DRIVE / f"encruz_precip_{args.produto}_md_area"
    prefix = f"{args.produto}_md_area_otto"
    if not base.exists():
        print(f"[ERRO] base centroide ausente: {base}"); return 1

    csvs = sorted(glob.glob(str(csv_dir / f"{prefix}_*.csv")))
    print(f"{args.produto}: CSVs area {len(csvs)}/{p['n_esp']} ({csv_dir})")
    if len(csvs) < p["n_esp"]:
        print("  [AGUARDANDO] faltam chunks; nao gravo."); return 1

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["cob"] = df["cob"].map(norm_cod)
    df["val"] = pd.to_numeric(df["mean"], errors="coerce") * p["mult"]   # NaN preservado (p/ fallback)
    if p["agg_hour"]:
        df["ts"] = df["ts"].dt.floor("h")
        df = df.groupby(["ts", "cob"], as_index=False)["val"].mean()
    piv = df.pivot_table(index="ts", columns="cob", values="val", aggfunc="mean")
    print(f"pivot area: {piv.shape[0]}h x {piv.shape[1]} ottos | {piv.index.min()} -> {piv.index.max()}")

    g = gpd.read_file(GPKG)
    col = next(c for c in g.columns if c.lower().startswith("cobacia"))
    cob_order = [norm_cod(x) for x in g[col].values]

    with h5py.File(base, "r") as f:
        splits = {s: f[f"{s}/timestamps"][:].astype("int64") for s in ("train", "val", "test")}
        cen = {s: f[f"{s}/precipitation"][:] for s in ("train", "val", "test")}   # centroide (fallback)
        n_otto = f["train/precipitation"].shape[1]
    assert len(cob_order) == n_otto, f"{len(cob_order)} != {n_otto}"

    precip, fb_total = {}, np.zeros(n_otto, bool)
    for s, ts in splits.items():
        idx = pd.to_datetime(ts, unit="s", utc=True)
        area = piv.reindex(index=idx, columns=cob_order).to_numpy(np.float32).copy()   # copy: writable p/ fallback; NaN onde area faltou
        nan_mask = np.isnan(area)
        # fallback: onde a area e NaN, usa o centroide da base
        area[nan_mask] = cen[s][nan_mask]
        fb_cols = nan_mask.any(axis=0)            # ottos que usaram fallback em algum passo
        fb_total |= fb_cols
        d = np.abs(area - cen[s])
        precip[s] = np.nan_to_num(area, nan=0.0)
        print(f"  [{s:5s}] {area.shape} | fallback em {fb_cols.sum()} ottos | "
              f"NaN final {100*np.isnan(precip[s]).mean():.2f}% | mean {precip[s].mean():.4f} mm/h "
              f"| diff vs centroide {d[~nan_mask].mean():.4f}")

    shutil.copy(base, dst)                        # base centroide -> dst (preserva pet/Q/timestamps)
    with h5py.File(dst, "a") as f:
        for s, arr in precip.items():
            del f[f"{s}/precipitation"]
            f.create_dataset(f"{s}/precipitation", data=arr, compression="gzip")
        dist = np.unique(np.round(precip["test"], 4), axis=1).shape[1]
        f.attrs["precip_fonte"] = (f"{args.produto} GEE reduceRegions polygon scale=1000 - MEDIA AREAL "
                                   f"ponderada por ottobacia (mesmo metodo do MERGE; fallback centroide "
                                   f"em {int(fb_total.sum())} ottos sub-pixel)")
    print(f"  -> {dst} | distintas/test={dist} | {int(fb_total.sum())} ottos com fallback centroide (de {n_otto})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
