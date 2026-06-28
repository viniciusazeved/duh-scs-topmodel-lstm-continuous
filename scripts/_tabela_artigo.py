#!/usr/bin/env python
"""Tabela COMPLETA p/ o artigo: reusa carrega() do consolida_sim e imprime media+-dp por modelo
de NSE, logNSE, KGE(agregado e decomposto r/alpha/beta), PBIAS, FHV, FLV. So leitura; nao toca GPU.

Uso: uv run --project D:/TTD_SCS_LSTM python scripts/_tabela_artigo.py   (de ablacao_skill/)
"""
import sys
import numpy as np
from consolida_sim import carrega, ORDEM

por = carrega()


def st(rows, k):
    xs = [x[k] for x in rows if k in x and np.isfinite(x[k])]
    if not xs:
        return (float("nan"), 0.0)
    return (float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)


hdr = (f"{'modelo':<30}{'n':>3}  {'NSE':>8}{'dp':>7}  {'logNSE':>8}{'dp':>7}  "
       f"{'KGE':>8}{'dp':>7}  {'r':>7}{'alpha':>7}{'beta':>7}{'PBIAS':>8}{'FHV%':>8}{'FLV%':>8}")
print(hdr)
print("-" * len(hdr))
for m in ORDEM:
    if m not in por:
        continue
    rows = list(por[m].values())
    nse = st(rows, "nse"); lns = st(rows, "lognse"); kge = st(rows, "kge")
    r = st(rows, "kge_r"); al = st(rows, "kge_alpha"); be = st(rows, "kge_beta")
    pb = st(rows, "pbias"); fhv = st(rows, "fhv"); flv = st(rows, "flv")
    print(f"{m:<30}{len(rows):>3}  {nse[0]:>+8.3f}{nse[1]:>7.3f}  {lns[0]:>+8.3f}{lns[1]:>7.3f}  "
          f"{kge[0]:>+8.3f}{kge[1]:>7.3f}  {r[0]:>+7.3f}{al[0]:>7.3f}{be[0]:>7.3f}"
          f"{pb[0]:>+8.1f}{fhv[0]:>8.1f}{flv[0]:>8.1f}")
