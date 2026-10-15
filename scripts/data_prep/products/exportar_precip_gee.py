#!/usr/bin/env python
"""Exporta precipitacao por OTTOBACIA (183) da Encruzilhada via GEE: ERA5-Land e IMERG.

reduceRegions(mean) das 183 ottobacias sobre cada imagem; exporta serie temporal
pro Google Drive (cai em G:\\Meu Drive\\<folder>, que e o Drive desktop montado).
Depois um script consolida os CSVs -> precip_<produto>_71350001.h5 (T, 183).

Unidades (convertidas no consumo, nao aqui):
  ERA5-Land 'total_precipitation_hourly': metros acumulados/hora -> *1000 = mm/h
  IMERG 'precipitation': mm/hr (taxa, 30min) -> agregar 2 p/ hora no consumo

Uso:
  # valida 1 dia (sincrono, rapido) antes de comprometer exports:
  cd D:\\CaBRA_H ; uv run python <este> --produto era5land --teste
  # dispara exports anuais (assincrono no GEE):
  cd D:\\CaBRA_H ; uv run python <este> --produto era5land --export
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import geopandas as gpd
from shapely.geometry import mapping

OTTOS = Path(r"D:\TTD_SCS_LSTM\data\manual_71350001\ottobacias_71350001.shp")

PRODUTOS = {
    "era5land": {
        "asset": "ECMWF/ERA5_LAND/HOURLY",
        "band": "total_precipitation_hourly",
        "scale": 11132,  # ~0.1 graus
        "folder": "encruz_precip_era5land",
        "chunk_meses": 12,  # horario (24/dia) -> ~1.6M feats/ano, ok
    },
    "imerg": {
        "asset": "NASA/GPM_L3/IMERG_V07",
        "band": "precipitation",
        "scale": 11132,
        "folder": "encruz_precip_imerg",
        "chunk_meses": 6,  # 30min (48/dia) -> semestral ~1.6M feats/task (evita timeout)
    },
    "t2m": {  # temperatura 2m horaria (Kelvin) p/ PET-Hargreaves por ottobacia
        "asset": "ECMWF/ERA5_LAND/HOURLY",
        "band": "temperature_2m",
        "scale": 11132,
        "folder": "era5_t2m",
        "chunk_meses": 12,
    },
}


def build_fc(ottos_path=OTTOS, cob_prefix="cobacia_me", centroide=False):
    ot = gpd.read_file(ottos_path).to_crs(4326)
    col = next(c for c in ot.columns if c.lower().startswith(cob_prefix))
    if centroide:  # ottos < pixel -> centroide pega o pixel (sem None)
        cen = ot.to_crs(31983).geometry.centroid.to_crs(4326)
        feats = [ee.Feature(ee.Geometry.Point([p.x, p.y]), {"cob": str(c)})
                 for c, p in zip(ot[col], cen)]
    else:
        ot["geometry"] = ot.geometry.simplify(0.003)  # ~300 m, reduz vertices pro GEE
        feats = [ee.Feature(ee.Geometry(mapping(r.geometry)), {"cob": str(r[col])})
                 for _, r in ot.iterrows()]
    return ee.FeatureCollection(feats), len(ot)


def reduzir(coll, fc, scale):
    def _red(img):
        r = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale)
        return r.map(lambda f: f.set("t", img.get("system:time_start")))
    return coll.map(_red).flatten()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--produto", choices=list(PRODUTOS), required=True)
    ap.add_argument("--teste", action="store_true", help="getInfo de 1 dia (valida)")
    ap.add_argument("--export", action="store_true", help="dispara exports anuais 2014-2025")
    ap.add_argument("--ottos", default=str(OTTOS), help="shapefile/gpkg das ottobacias")
    ap.add_argument("--cob-prefix", default="cobacia_me", help="prefixo da coluna de cobacia")
    ap.add_argument("--tag", default="", help="sufixo p/ folder e nomes (ex: _md)")
    ap.add_argument("--scale", type=int, default=0, help="scale reduceRegions (0=nativo; menor p/ ottos < pixel)")
    ap.add_argument("--centroide", action="store_true", help="usar centroide das ottos (ottos < pixel; sem None)")
    ap.add_argument("--inicio", type=int, default=2014, help="ano inicial do export (default 2014; MD usa 2021)")
    ap.add_argument("--chunk", type=int, default=0, help="meses por task (0=default do produto; menor evita timeout do polygon+escala fina)")
    args = ap.parse_args()

    ee.Initialize(project="ggeantigravity")
    p = PRODUTOS[args.produto]
    sc = args.scale or p["scale"]
    fc, n = build_fc(args.ottos, args.cob_prefix, args.centroide)
    print(f"FeatureCollection: {n} ottobacias | produto {args.produto} ({p['band']}) | tag '{args.tag}'")

    if args.teste:
        coll = ee.ImageCollection(p["asset"]).select(p["band"]).filterDate(
            "2020-01-15", "2020-01-16")
        print(f"imagens no dia: {coll.size().getInfo()}")
        out = reduzir(coll, fc, sc)
        amostra = out.limit(8).getInfo()["features"]
        print("amostra (cob, t, mean nativo):")
        for f in amostra:
            pr = f["properties"]
            print(f"  cob={pr.get('cob')}  t={pr.get('t')}  mean={pr.get('mean')}")
        # sanidade de unidade: ERA5 mean em metros (~1e-4 a 1e-3); IMERG em mm/hr (~0-30)
        return 0

    if args.export:
        import datetime as _dt
        chunk = args.chunk or p.get("chunk_meses", 12)
        cur, end = _dt.date(args.inicio, 1, 1), _dt.date(2026, 1, 1)
        n = 0
        while cur < end:
            mo = cur.month - 1 + chunk
            nxt = _dt.date(cur.year + mo // 12, mo % 12 + 1, 1)
            ini, fim = cur.isoformat(), min(nxt, end).isoformat()
            tag = cur.strftime("%Y%m")
            coll = ee.ImageCollection(p["asset"]).select(p["band"]).filterDate(ini, fim)
            out = reduzir(coll, fc, sc)
            ee.batch.Export.table.toDrive(
                collection=out,
                description=f"{args.produto}{args.tag}_otto_{tag}",
                folder=p["folder"] + args.tag,
                fileFormat="CSV",
                selectors=["cob", "t", "mean"],
            ).start()
            print(f"  task disparada: {args.produto}{args.tag}_otto_{tag} ({ini}->{fim})")
            cur = nxt
            n += 1
        print(f"\n{n} tasks rodando no GEE (assincrono, chunk={chunk} meses).")
        print(f"Saida cai em G:\\Meu Drive\\{p['folder']}{args.tag}\\")
        return 0

    print("nada a fazer: passe --teste ou --export")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
