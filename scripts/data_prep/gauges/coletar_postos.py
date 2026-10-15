#!/usr/bin/env python
"""Coleta e avalia pluviometros ate RAIO km de uma bacia (espacial+temporal), GENERICO.

Mesma logica validada na Encruzilhada (encruzilhada/scripts/coletar_avaliar_postos.py),
parametrizada por bacia via dict BACIAS (--bacia-id). Inventario REST (pluviometricos das
UFs) filtrado por distancia ao poligono; chuva DIARIA historica (SOAP legacy tipo=2);
cobertura 2014-2025.

Saidas (big_ablacao/data/postos/<bacia_id>/):
  postos_meta.parquet   -> 1 linha/posto
  postos_series.parquet -> long: codigo,data,chuva_mm

Roda no env do CaBRA-H:
  cd D:\\CaBRA_H ; uv run python D:\\TTD_SCS_LSTM\\big_ablacao\\scripts\\coletar_postos.py --bacia-id md
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\CaBRA_H\src")

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from cabra_h.ana.client_legacy import TelemetriaLegacyClient
from cabra_h.ana.client_rest import HidroWebClient

# Cada bacia: geometria (shp/gpkg; dissolvida), EPSG metrico p/ distancia, UFs do inventario.
BACIAS = {
    "encruzilhada": {
        "geom": r"D:\TTD_SCS_LSTM\encruzilhada\data\manual\vetor\bacia_.shp",
        "epsg_m": 31982,  # UTM 22S (SC)
        "ufs": ["SC", "RS", "PR"],
    },
    "md": {
        "geom": r"D:\TTD_SCS_LSTM\data\processed\ottobacias_tc.gpkg",  # 245 ottos -> dissolve
        "epsg_m": 31983,  # UTM 23S (Rio Preto, RJ/MG)
        "ufs": ["RJ", "MG", "SP"],
    },
}
OUT_BASE = Path(r"D:\TTD_SCS_LSTM\big_ablacao\data\postos")
RAIO_KM = 20.0
ANO_INI, ANO_FIM = 2014, 2025


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bacia-id", choices=list(BACIAS), required=True)
    ap.add_argument("--raio", type=float, default=RAIO_KM)
    args = ap.parse_args()
    cfg = BACIAS[args.bacia_id]
    out = OUT_BASE / args.bacia_id
    out.mkdir(parents=True, exist_ok=True)

    bac = gpd.read_file(cfg["geom"]).to_crs(4326)
    poly = bac.union_all() if hasattr(bac, "union_all") else bac.unary_union
    bac_m = bac.to_crs(cfg["epsg_m"])
    poly_m = bac_m.union_all() if hasattr(bac_m, "union_all") else bac_m.unary_union
    print(f"bacia {args.bacia_id}: {len(bac)} feicoes | EPSG_m={cfg['epsg_m']} | UFs={cfg['ufs']}")

    # 1. inventario pluviometricos
    print("inventariando pluviometricos...")
    with HidroWebClient() as cli:
        est = []
        for uf in cfg["ufs"]:
            try:
                est += cli.inventario(uf=uf)
            except Exception as e:
                print(f"  inventario {uf} ERRO: {e}")
    pluv = [e for e in est if e.is_pluviometrica and e.latitude and e.longitude]
    print(f"  {len(pluv)} pluviometricos com coord ({'+'.join(cfg['ufs'])})")

    # 2. filtrar por distancia <= RAIO ao poligono (em metros)
    candidatos = []
    for e in pluv:
        pt_m = gpd.GeoSeries([Point(e.longitude, e.latitude)], crs=4326).to_crs(cfg["epsg_m"]).iloc[0]
        dist_km = poly_m.distance(pt_m) / 1000.0
        if dist_km <= args.raio:
            dentro = poly.contains(Point(e.longitude, e.latitude))
            candidatos.append((e, dist_km, "dentro" if dentro else "buffer"))
    print(f"  {len(candidatos)} pluviometricos <= {args.raio:.0f} km da bacia "
          f"({sum(1 for _,_,l in candidatos if l=='dentro')} dentro)")

    # 3. chuva diaria historica de cada
    metas, series = [], []
    with TelemetriaLegacyClient() as leg:
        for e, dist_km, local in sorted(candidatos, key=lambda x: x[1]):
            try:
                pts = leg.hidro_serie_historica(e.codigo, tipo_dados=2)
            except Exception as ex:
                print(f"  {e.codigo} ERRO chuva: {type(ex).__name__}")
                pts = []
            df = pd.DataFrame([(p.data, p.valor) for p in pts if p.valor is not None],
                              columns=["data", "chuva_mm"])
            if not df.empty:
                df["data"] = pd.to_datetime(df["data"])
                df = df[~df["data"].duplicated(keep="last")].sort_values("data")
                df["codigo"] = e.codigo
                series.append(df[["codigo", "data", "chuva_mm"]])
                per = df[(df.data.dt.year >= ANO_INI) & (df.data.dt.year <= ANO_FIM)]
                ndias = len(per)
                n_anos = per.data.dt.year.nunique()
                cob = ndias / (365.25 * (ANO_FIM - ANO_INI + 1))
                ch_anual = per.chuva_mm.sum() / max(n_anos, 1)
                ini, fim = df.data.min(), df.data.max()
            else:
                ndias = n_anos = cob = ch_anual = 0
                ini = fim = pd.NaT
            metas.append(dict(codigo=e.codigo, nome=e.nome, lat=e.latitude, lon=e.longitude,
                              dist_km=round(dist_km, 1), local=local,
                              is_telem=e.is_telemetrica, n_dias_2014_2025=ndias,
                              frac_cob=round(cob, 3), chuva_anual_mm=round(ch_anual, 0),
                              ano_ini=ini.year if pd.notna(ini) else None,
                              ano_fim=fim.year if pd.notna(fim) else None))
            print(f"  {e.codigo} {local:6s} {dist_km:4.1f}km  {ndias:5d}d cob={cob:.2f}  {e.nome[:26]}")

    meta = pd.DataFrame(metas).sort_values(["local", "dist_km"])
    meta.to_parquet(out / "postos_meta.parquet", index=False)
    if series:
        pd.concat(series, ignore_index=True).to_parquet(out / "postos_series.parquet", index=False)

    bons = meta[meta.frac_cob >= 0.7]
    print(f"\n[OK] {len(meta)} postos avaliados | {len(bons)} com cobertura >=70% em 2014-2025")
    print(f"     {(meta.local=='dentro').sum()} dentro, {(meta.local=='buffer').sum()} no buffer {args.raio:.0f}km")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
