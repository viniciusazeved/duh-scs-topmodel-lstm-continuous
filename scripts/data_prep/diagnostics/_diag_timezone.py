#!/usr/bin/env python
"""Diagnostico de fuso: compara o CICLO DIURNO da chuva entre os produtos de MD.
Pico convectivo deve cair na mesma hora-UTC se os produtos estiverem alinhados.
Se a telem estiver 3h deslocada dos satelites (UTC nativo), esta em BRT.
Tambem checa o lag chuva->vazao por correlacao cruzada (sanidade do alinhamento P->Q).
"""
import h5py, numpy as np, pandas as pd
from pathlib import Path

base = Path(r"D:\TTD_SCS_LSTM\big_ablacao\data")
prods = ["telem", "merge", "imerg", "era5land"]

def carrega(p):
    f = base / f"dataset_58585000_{p}.h5"
    if not f.exists():
        return None
    with h5py.File(f, "r") as h:
        ts = np.concatenate([h[f"{s}/timestamps"][:].astype("int64") for s in ("train", "val", "test")])
        pr = np.concatenate([h[f"{s}/precipitation"][:] for s in ("train", "val", "test")])
        q = np.concatenate([h[f"{s}/streamflow"][:] for s in ("train", "val", "test")])
    return ts, pr.mean(axis=1), q

print("=== CICLO DIURNO DA CHUVA (precip media por hora UTC) ===")
print("pico ~tarde local (Sudeste, conveccao). BRT = UTC-3.\n")
diur = {}
ref_ts = None
for p in prods:
    d = carrega(p)
    if d is None:
        print(f"{p:10s}: SEM ARQUIVO"); continue
    ts, pm, q = d
    if ref_ts is None:
        ref_ts, ref_q = ts, q
    hr = pd.to_datetime(ts, unit="s", utc=True).hour
    s = pd.Series(pm).groupby(hr).mean().reindex(range(24))
    diur[p] = s
    pico = int(s.idxmax()); vale = int(s.idxmin())
    print(f"{p:10s}: pico {pico:02d}h UTC ({(pico-3)%24:02d}h BRT) | vale {vale:02d}h UTC | "
          f"media {pm.mean():.4f} mm/h | curva: " + " ".join(f"{v:.2f}" for v in s.to_numpy()))

# alinhamento relativo: shift (horas) que casa cada produto com a telem
if "telem" in diur:
    print("\n=== SHIFT que MAXIMIZA correlacao do ciclo diurno vs TELEM ===")
    print("(0h = alinhado com telem | 3h ou 21h = offset de fuso de 3h)\n")
    t = diur["telem"].to_numpy(); t = t - t.mean()
    for p in prods:
        if p not in diur or p == "telem":
            continue
        m = diur[p].to_numpy(); m = m - m.mean()
        cc = [np.corrcoef(np.roll(t, k), m)[0, 1] for k in range(24)]
        best = int(np.argmax(cc))
        print(f"telem vs {p:10s}: melhor shift = {best:2d}h (corr {max(cc):.3f})")

print("\n=== LAG chuva->vazao (corr cruzada, mascara NaN; pico esperado ~Tc) ===")
for p in prods:
    d = carrega(p)
    if d is None:
        continue
    ts, pm, q = d
    pmf = pm.astype(float); qf = q.astype(float)
    cc = []
    for L in range(0, 37):
        a = pmf[:len(pmf) - L] if L else pmf
        b = qf[L:] if L else qf
        msk = np.isfinite(a) & np.isfinite(b)
        cc.append(np.corrcoef(a[msk], b[msk])[0, 1] if msk.sum() > 100 else np.nan)
    cc = np.array(cc)
    best = int(np.nanargmax(cc))
    print(f"{p:10s}: lag pico chuva->vazao = {best:2d}h (corr {np.nanmax(cc):.3f})")
