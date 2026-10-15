#!/usr/bin/env python
"""Screening: quais postos candidatos (das 2 bacias) tem CHUVA TELEMETRICA adotada
(REST Chuva_Adotada, sub-horaria). Sonda 2 janelas de 30 dias (2022 e 2024) por posto;
se vier chuva_mm em alguma, o posto tem telemetria de chuva.

Saida: big_ablacao/data/postos/screening_telem.parquet (bacia,codigo,nome,dist,local,n_telem)
Roda no env CaBRA-H.
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"D:\CaBRA_H\src")

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from cabra_h.ana.client_rest import HidroWebClient

OUT = Path(r"D:\TTD_SCS_LSTM\big_ablacao\data\postos")
PROBES = [
    (datetime(2022, 7, 1, tzinfo=timezone.utc), datetime(2022, 7, 31, tzinfo=timezone.utc)),
    (datetime(2024, 7, 1, tzinfo=timezone.utc), datetime(2024, 7, 31, tzinfo=timezone.utc)),
]

rows = []
with HidroWebClient() as cli:
    for bacia in ["md", "encruzilhada"]:
        meta = pd.read_parquet(OUT / bacia / "postos_meta.parquet")
        print(f"\n=== {bacia}: {len(meta)} candidatos ===")
        for r in meta.itertuples():
            n = 0
            for ini, fim in PROBES:
                try:
                    s = cli.serie_telemetrica_adotada(str(r.codigo).zfill(8), ini, fim)
                    n += sum(1 for m in s if getattr(m, "chuva_mm", None) is not None)
                except Exception:
                    pass
            rows.append(dict(bacia=bacia, codigo=str(r.codigo).zfill(8), nome=r.nome,
                             dist_km=r.dist_km, local=r.local, n_telem=n))
            if n:
                print(f"  TEM  {r.codigo} n={n:5d} {r.dist_km:4.1f}km {r.nome[:26]}")

df = pd.DataFrame(rows)
df.to_parquet(OUT / "screening_telem.parquet", index=False)
print("\n=== RESUMO ===")
for bacia in ["md", "encruzilhada"]:
    sub = df[df.bacia == bacia]
    tem = sub[sub.n_telem > 0]
    print(f"{bacia:13s}: {len(tem)}/{len(sub)} postos com chuva telemetrica "
          f"({(tem.local=='dentro').sum()} dentro)")
print(f"\n  -> {OUT}\\screening_telem.parquet")
