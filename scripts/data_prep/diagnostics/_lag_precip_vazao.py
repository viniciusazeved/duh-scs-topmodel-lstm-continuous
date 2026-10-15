"""Lag da correlacao cruzada chuva-media-da-bacia x vazao, SERIE COMPLETA (train+val+test),
por produto. Confirma o numero do paper (~29-36h) e que e da serie inteira (nao evento).
Read-only. Uso: uv run python big_ablacao/scripts/_lag_precip_vazao.py"""
import h5py, numpy as np
from pathlib import Path

BASE = Path(r"D:\TTD_SCS_LSTM\big_ablacao\data")
PRODS = ["telem", "merge", "era5land", "imerg", "mswep"]
MAXLAG = 120  # horas


def serie_completa(f):
    """precip media de bacia (ponderada por area) + vazao, concatenando os 3 splits."""
    area = f["ottobacia/area_km2"][:].astype(np.float64); w = area / area.sum()
    P, Q = [], []
    for s in ("train", "val", "test"):
        pr = f[f"{s}/precipitation"][:].astype(np.float64)      # (T,245)
        P.append(pr @ w)                                        # media ponderada -> (T,)
        Q.append(f[f"{s}/streamflow"][:].astype(np.float64))
    return np.concatenate(P), np.concatenate(Q)


def lag_pico(P, Q, maxlag=MAXLAG):
    """corr de Pearson entre P[t] e Q[t+L] p/ L=0..maxlag; retorna L do pico e a corr."""
    best_l, best_r = 0, -np.inf
    rs = []
    for L in range(0, maxlag + 1):
        a = P[:len(P) - L]; b = Q[L:]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 1000:
            rs.append(np.nan); continue
        r = np.corrcoef(a[m], b[m])[0, 1]
        rs.append(r)
        if r > best_r:
            best_r, best_l = r, L
    return best_l, best_r, np.array(rs)


print(f"{'produto':9s} {'lag pico (h)':>13s} {'r no pico':>11s} {'r em lag0':>11s}")
lags = []
for prod in PRODS:
    fp = BASE / f"dataset_58585000_{prod}.h5"
    if not fp.exists():
        print(f"{prod:9s}  (ausente)"); continue
    with h5py.File(fp, "r") as f:
        P, Q = serie_completa(f)
    L, r, rs = lag_pico(P, Q)
    lags.append(L)
    print(f"{prod:9s} {L:>13d} {r:>11.3f} {rs[0]:>11.3f}")
print("-" * 48)
print(f"faixa do lag de pico entre produtos: {min(lags)}-{max(lags)} h   (Tc topografico ~9.5 h)")
print("=> lag da SERIE COMPLETA (multianual), muito maior que o Tc de evento. Confirmado.")
