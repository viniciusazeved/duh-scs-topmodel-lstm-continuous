#!/usr/bin/env python
"""Coleta a serie telemetrica adotada (chuva_mm sub-horaria) dos postos com telemetria
(screening_telem.parquet, n_telem>0), em PARALELO (HidroWebClientAsync) com cache por
estacao. Agrega sub-horario -> horario (soma) e salva a serie pro notebook telemetrico.

Saidas (big_ablacao/data/postos/<bacia>/):
  telem_cache/<codigo>.parquet   (cache bruto por estacao, reusavel)
  postos_telem_series.parquet    (codigo, data_hora[h], chuva_mm)  horaria

Periodo por bacia = periodo do dataset (MD 2021-2026, Encruz 2014-2026).
Roda no env CaBRA-H.
"""
from __future__ import annotations

import asyncio
import sys
sys.path.insert(0, r"D:\CaBRA_H\src")

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from cabra_h.ana.client_rest_async import HidroWebClientAsync

OUT = Path(r"D:\TTD_SCS_LSTM\big_ablacao\data\postos")
PERIODO = {"md": (2021, 2026), "encruzilhada": (2014, 2026)}


async def coletar(bacia: str) -> None:
    scr = pd.read_parquet(OUT / "screening_telem.parquet")
    cods = scr[(scr.bacia == bacia) & (scr.n_telem > 0)].codigo.astype(str).str.zfill(8).tolist()
    a, b = PERIODO[bacia]
    ini = datetime(a, 1, 1, tzinfo=timezone.utc)
    fim = datetime(b, 1, 1, tzinfo=timezone.utc)
    cache = OUT / bacia / "telem_cache"
    print(f"\n=== {bacia}: {len(cods)} postos telemetricos | {a}-{b} ===")

    async with HidroWebClientAsync(max_concurrent=6) as cli:
        res = await cli.baixar_varias(
            cods, ini, fim, cache_dir=cache,
            on_complete=lambda c, n, ch: print(f"  {c}: {n:6d} med {'(cache)' if ch else ''}"),
        )

    rows = []
    for cod, meds in res.items():
        for m in meds:
            if getattr(m, "chuva_mm", None) is not None:
                rows.append((cod, m.data_hora, float(m.chuva_mm)))
    if not rows:
        print(f"  [AVISO] {bacia}: 0 registros de chuva"); return
    df = pd.DataFrame(rows, columns=["codigo", "data_hora", "chuva_mm"])
    df["data_hora"] = pd.to_datetime(df["data_hora"], utc=True)
    # sub-horario -> horario: soma a chuva incremental dentro da hora
    dfh = (df.groupby(["codigo", df["data_hora"].dt.floor("h")])["chuva_mm"].sum()
             .reset_index())
    dfh.to_parquet(OUT / bacia / "postos_telem_series.parquet", index=False)
    npost = dfh.codigo.nunique()
    print(f"  {bacia}: {npost} postos c/ chuva | {len(dfh)} horas-posto | "
          f"{dfh.data_hora.min()} -> {dfh.data_hora.max()}")


async def main() -> None:
    for bacia in ["md", "encruzilhada"]:
        await coletar(bacia)


if __name__ == "__main__":
    asyncio.run(main())
