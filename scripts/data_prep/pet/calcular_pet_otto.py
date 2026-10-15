#!/usr/bin/env python
"""Calcula PET-Hargreaves POR OTTOBACIA do MD (de t2m ERA5 por otto) e injeta nos
datasets MD como pet (T, N) — pros modelos Topmodel/peonly usarem PET distribuida
(amplitude termica intra-bacia; antes era climatologia lumped).

Hargreaves-Samani: ETo = 0.0023 * Ra * (Tmean+17.8) * sqrt(Tmax-Tmin)  [mm/dia]
  Ra (radiacao extraterrestre, FAO-56) pela latitude do centroide + dia juliano.
Diario -> horario: /24, repetido nas 24h. Reindexa na grade de cada split.

Entrada: CSVs t2m por otto (encruz... /precip_csv/era5_t2m_md/), ottobacias_tc.gpkg (lat).
Saida: adiciona {split}/pet (T,N) aos dataset_58585000_*.h5.
Uso (env c/ geopandas+h5py): uv run python <este>
"""
from __future__ import annotations

import glob
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd

T2M_DIR = Path(r"D:\TTD_SCS_LSTM\encruzilhada\data\precip_csv\era5_t2m_md")
OTTOS = r"D:\TTD_SCS_LSTM\data\processed\ottobacias_tc.gpkg"
DATA = Path(r"D:\TTD_SCS_LSTM\big_ablacao\data")
PRODUTOS = ["merge", "era5land", "imerg", "telem", "mswep"]


def norm_cod(x) -> str:
    s = (x.decode() if isinstance(x, bytes) else str(x)).strip()
    return s[:-2] if s.endswith(".0") else s


def ra_mm(lat_deg, j):
    """Radiacao extraterrestre Ra (FAO-56 Eq.21), convertida p/ mm/dia equivalente."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    dec = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec)
                                            + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return 0.408 * ra  # MJ/m2/dia -> mm/dia


def main() -> int:
    # ottobacias: cob -> latitude do centroide (4326)
    ot = gpd.read_file(OTTOS).to_crs(4326)
    colcob = next(c for c in ot.columns if c.lower().startswith("cobacia"))
    ot["cob"] = ot[colcob].map(norm_cod)
    lat_by_cob = dict(zip(ot["cob"], ot.geometry.centroid.y))

    # t2m horario por otto
    csvs = sorted(glob.glob(str(T2M_DIR / "t2m_md_otto_*.csv")))
    if not csvs:
        print(f"[ERRO] sem CSVs t2m em {T2M_DIR} (t2m ainda baixando?)"); return 1
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["cob"] = df["cob"].map(norm_cod)
    df["tc"] = pd.to_numeric(df["mean"], errors="coerce") - 273.15  # K -> C
    df["dia"] = df["ts"].dt.floor("D")
    print(f"t2m: {len(csvs)} CSVs | {df.cob.nunique()} ottos | {df.dia.min().date()}->{df.dia.max().date()}")

    # diario por cob: tmin/tmax/tmean
    g = df.groupby(["cob", "dia"])["tc"].agg(["min", "max", "mean"]).reset_index()
    g["lat"] = g["cob"].map(lat_by_cob)
    g["j"] = g["dia"].dt.dayofyear
    g["ra"] = ra_mm(g["lat"].values, g["j"].values)
    dt_amp = np.clip((g["max"] - g["min"]).values, 0, None)
    g["pet_dia"] = 0.0023 * g["ra"] * (g["mean"] + 17.8) * np.sqrt(dt_amp)  # mm/dia
    g["pet_dia"] = g["pet_dia"].clip(lower=0)
    # pivot dia x cob (PET diaria)
    petd = g.pivot_table(index="dia", columns="cob", values="pet_dia")
    print(f"PET diaria: {petd.shape[0]} dias x {petd.shape[1]} ottos | media {petd.mean().mean():.2f} mm/dia")

    # injeta em cada dataset MD (mesmos timestamps; pet horaria = pet_dia/24 no dia do timestamp)
    for prod in PRODUTOS:
        h5 = DATA / f"dataset_58585000_{prod}.h5"
        if not h5.exists():
            print(f"  [pula] {h5.name} ausente"); continue
        with h5py.File(h5, "r") as f:
            cob_order = None  # MD nao tem cobacia no .h5 -> usa ordem do gpkg (validado)
        cob_order = list(ot["cob"])  # ordem gpkg == ordem .h5 (validado validar_ottos_md)
        petd_ord = petd.reindex(columns=cob_order)
        with h5py.File(h5, "a") as f:
            n_otto = f["train/precipitation"].shape[1]
            assert len(cob_order) == n_otto, f"{prod}: cob_order {len(cob_order)} != {n_otto}"
            for split in ("train", "val", "test"):
                ts = f[f"{split}/timestamps"][:].astype("int64")
                dias = pd.to_datetime(ts, unit="s", utc=True).floor("D")
                peth = petd_ord.reindex(index=dias).to_numpy(np.float32) / 24.0  # mm/dia -> mm/h
                peth = np.nan_to_num(peth, nan=float(np.nanmean(petd.values)) / 24.0)
                if f"{split}/pet" in f:
                    del f[f"{split}/pet"]
                f.create_dataset(f"{split}/pet", data=peth, compression="gzip")
        print(f"  [OK] {h5.name}: pet (T,N) injetada | mean {petd_ord.mean().mean()/24:.4f} mm/h")
    print("\n[FIM] PET por ottobacia injetada nos datasets MD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
