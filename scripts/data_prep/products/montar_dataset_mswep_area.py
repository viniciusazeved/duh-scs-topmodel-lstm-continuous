"""Re-monta dataset_58585000_mswep.h5 por AREA-PONDERADA (interseccao pixel cap ottobacia),
MESMO metodo do MERGE. Substitui a precip centroide pela area-ponderada; preserva
pet/streamflow/timestamps/estaticos. Backup do centroide em mswep_centroide_backup.h5.

Uso: uv run python big_ablacao/scripts/montar_dataset_mswep_area.py [--dry-run]
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import h5py, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import box

BASE = Path(r"D:\TTD_SCS_LSTM")
OTTOS = BASE / "data/processed/ottobacias_tc.gpkg"
PAST = BASE / "big_ablacao/data/mswep_raw/Hourly"
NRT = BASE / "big_ablacao/data/mswep_raw/NRT_2025"
DST = BASE / "big_ablacao/data/dataset_58585000_mswep.h5"          # o que o runner usa
BACKUP = BASE / "big_ablacao/data/mswep_centroide_backup.h5"
GAP = pd.Timestamp("2025-07-01", tz="UTC")
LAT0, LON0, DLL = -89.95, -179.95, 0.1


def build_weights():
    ot = gpd.read_file(OTTOS)
    ot4326 = ot.to_crs(4326)
    minx, miny, maxx, maxy = ot4326.total_bounds
    j0 = int(np.floor((minx - LON0) / DLL)); j1 = int(np.ceil((maxx - LON0) / DLL))
    i0 = int(np.floor((miny - LAT0) / DLL)); i1 = int(np.ceil((maxy - LAT0) / DLL))
    cells, cell_ij = [], []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            clon, clat = LON0 + j * DLL, LAT0 + i * DLL
            cells.append(box(clon - DLL/2, clat - DLL/2, clon + DLL/2, clat + DLL/2))
            cell_ij.append((i, j))
    cellsg = gpd.GeoDataFrame({"cid": range(len(cells))}, geometry=cells, crs=4326)
    ot_m = ot.to_crs(31983)[["geometry"]].reset_index(drop=True)
    cells_m = cellsg.to_crs(31983)
    W = np.zeros((len(ot_m), len(cells)), np.float64)
    for oi, og in enumerate(ot_m.geometry):
        for ci, cg in zip(cellsg.cid, cells_m.geometry):
            if og.intersects(cg):
                W[oi, ci] = og.intersection(cg).area
    rs = W.sum(axis=1, keepdims=True)
    sem = int((rs == 0).sum())
    rs[rs == 0] = 1.0
    W /= rs
    # BUG CORRIGIDO 07/06: o .nc do MSWEP tem LATITUDE DECRESCENTE (lat[0]=+89.95).
    # A geometria (clat = LAT0 + i*DLL) esta certa; mas o indice de LINHA do raster e' 1799 - i,
    # senao le a latitude espelhada (hemisferio norte) -> chuva de outra regiao.
    ii = np.array([(1800 - 1) - ij[0] for ij in cell_ij]); jj = np.array([ij[1] for ij in cell_ij])
    print(f"W: {W.shape[0]} ottos x {W.shape[1]} celulas | ottos sem cobertura={sem} (fallback nao usado se 0)")
    return W.astype(np.float32), ii, jj


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); args = ap.parse_args()
    W, ii, jj = build_weights()
    past = {p.name for p in PAST.glob("*.nc")}
    nrt = {p.name for p in NRT.glob("*.nc")} if NRT.exists() else set()

    def cells_at(ts_int):
        ts = pd.Timestamp(ts_int, unit="s", tz="UTC")
        st = f"{ts.year}{ts.dayofyear:03d}.{ts.hour:02d}.nc"
        fp = (NRT/st) if (ts >= GAP and st in nrt) else (PAST/st if st in past else (NRT/st if st in nrt else None))
        if fp is None: return None
        with h5py.File(fp, "r") as f:
            v = f["precipitation"][0][ii, jj].astype(np.float64)
        v[v < 0] = 0.0
        return v

    with h5py.File(DST, "r") as f:
        splits = {s: f[f"{s}/timestamps"][:].astype("int64") for s in ("train", "val", "test")}
        cen = {s: f[f"{s}/precipitation"][:] for s in ("train", "val", "test")}

    out = {}
    for s, ts in splits.items():
        arr = np.zeros((len(ts), W.shape[0]), np.float32); miss = 0
        for k, t in enumerate(ts):
            g = cells_at(int(t))
            if g is None: miss += 1
            else: arr[k] = (W @ g).astype(np.float32)
        out[s] = arr
        d = np.abs(arr - cen[s])
        print(f"  [{s:5s}] {arr.shape} | faltando {miss} | mean {arr.mean():.4f} mm/h "
              f"| vs centroide: diff_media={d.mean():.4f} max={d.max():.3f}")

    if args.dry_run:
        print("DRY-RUN: nada gravado."); return 0

    shutil.copy(DST, BACKUP)
    print(f"backup centroide -> {BACKUP.name}")
    with h5py.File(DST, "a") as f:
        for s, arr in out.items():
            del f[f"{s}/precipitation"]
            f.create_dataset(f"{s}/precipitation", data=arr, compression="gzip")
        f.attrs["precip_fonte"] = "MSWEP V3.16 0.1deg horario - MEDIA AREAL ponderada por interseccao pixel-ottobacia (mesmo metodo do MERGE)"
    print(f"-> {DST} re-agregado por AREA (pet/Q/timestamps preservados)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
