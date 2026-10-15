#!/usr/bin/env python
"""Quick table (helper): mean and sample sd over seeds of NSE, logNSE, KGE (with r, alpha and beta),
PBIAS, FHV and FLV per configuration, three decimals, in the order of consolida_sim.ORDEM (rungs 1
to 12 plus the auxiliary contrasts). Read only, no GPU. The published Table 3 (13 rungs, two
decimals, seed ensemble) comes from scripts/analysis/table3.py.

Usage, from the repository root: uv run python scripts/_tabela_artigo.py
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
