#!/usr/bin/env python
"""Consolida a grade de SIMULAÇÃO CONTÍNUA por janela (outputs/grade_sim, formato train.py).

Lê test_metrics.nse (= NSE@1h = NSE de simulação) e predictions.npz (pred/target shape (N,1)) de cada
modelo. Monta a cadeia "de onde vem o skill", contrastes pareados (Wilcoxon por seed) e assinaturas do
hidrograma (logNSE, KGE decomposto r/alpha/beta, PBIAS, FHV picos, FLV baixas).

Uso: uv run --project D:/TTD_SCS_LSTM python scripts/consolida_sim.py
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8")  # terminal Windows (cp1252) nao imprime - / Delta
except Exception:
    pass

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

ROOT = Path(__file__).resolve().parent.parent
GRADE = ROOT / "outputs" / "grade_sim"

ORDEM = [
    "LSTM_Lumped_CalOnly", "LSTM_Lumped_RainOnly", "LSTM_Lumped", "LSTM",
    "LSTM_DUH", "LSTM_DUH_Fixed", "LSTM_DUH_SCS", "LSTM_DUH_SCS_PeOnly",
    "LSTM_DUH_Topmodel", "LSTM_DUH_Topmodel_PeOnly", "LSTM_DUH_Topmodel_Baseflow", "Phys_DUH_SCS",
    "LSTM_DUH_Manning", "LSTM_DUH_Manning_Fixed",   # sensibilidade Tc (roteamento), engatilhada
    "LSTM_SCS", "LSTM_Topmodel",   # SEM DUH (geracao sem roteamento explicito), engatilhada
    "LSTM_PET",   # PET como feature (forcante evaporativa), engatilhada
]
CONTRASTES = [
    ("LSTM", "LSTM_Lumped", "espaço: distribuído > lumped"),
    ("LSTM_DUH_SCS", "LSTM", "física (SCS) > neural"),
    ("LSTM_DUH_Topmodel_Baseflow", "LSTM", "física (TOPMODEL+baseflow) > neural"),
    ("LSTM_DUH_Topmodel_Baseflow", "LSTM_DUH_Topmodel", "baseflow agrega (recessão)"),
    ("LSTM_DUH_SCS_PeOnly", "LSTM_DUH_SCS", "PeOnly no SCS (espera colapso)"),
    ("LSTM_DUH", "LSTM_DUH_Fixed", "aprendível > fixo (espera nulo)"),
    ("LSTM_DUH_Topmodel_Baseflow", "LSTM_DUH_SCS", "TOPMODEL+baseflow > SCS (gerador decide)"),
    ("LSTM_DUH_Manning", "LSTM_DUH", "Manning vs Base (quali achou Manning melhor na simulação)"),
    ("LSTM_DUH_Topmodel", "LSTM_Topmodel", "DUH agrega sobre TOPMODEL sem roteamento?"),
    ("LSTM_Topmodel", "LSTM", "TOPMODEL sem DUH > neural (geração sozinha agrega?)"),
    ("LSTM_PET", "LSTM", "PET agrega sobre o neural (forçante evaporativa > calendário)?"),
]


def _lognse(s, o):
    eps = 0.01 * np.nanmean(o)
    so, oo = np.log(s + eps), np.log(o + eps)
    den = np.sum((oo - oo.mean()) ** 2)
    return float(1 - np.sum((so - oo) ** 2) / den) if den > 0 else float("nan")


def _fdc(s, o):
    n = len(o); k = max(1, int(0.02 * n))
    ss, os_ = np.sort(s)[::-1], np.sort(o)[::-1]
    fhv = 100 * (ss[:k].sum() - os_[:k].sum()) / (os_[:k].sum() + 1e-9)
    lo_s, lo_o = np.sort(s)[: int(0.3 * n)], np.sort(o)[: int(0.3 * n)]
    flv = 100 * (np.log(lo_s + 1e-3).sum() - np.log(lo_o + 1e-3).sum()) / (abs(np.log(lo_o + 1e-3).sum()) + 1e-9)
    return float(fhv), float(flv)


def assinaturas(pred, target):
    p = pred[:, 0] if pred.ndim > 1 else pred
    t = target[:, 0] if target.ndim > 1 else target
    m = np.isfinite(p) & np.isfinite(t)
    s, o = p[m], t[m]
    if len(o) < 2:
        return {}
    r = np.corrcoef(s, o)[0, 1]
    out = {"lognse": _lognse(s, o), "kge_r": float(r),
           "kge_alpha": float(s.std() / (o.std() + 1e-9)),
           "kge_beta": float(s.mean() / (o.mean() + 1e-9)),
           "pbias": float(100 * (s.sum() - o.sum()) / (o.sum() + 1e-9))}
    out["fhv"], out["flv"] = _fdc(s, o)
    return out


def carrega():
    por = {}
    if not GRADE.exists():
        return por
    for sd in sorted(GRADE.glob("seed*")):
        try:
            seed = int(sd.name.replace("seed", ""))
        except ValueError:
            continue
        for dd in sorted(p for p in sd.iterdir() if p.is_dir()):
            cands = list(dd.glob("**/results.json"))
            if not cands:
                continue
            try:
                r = json.loads(cands[0].read_text(encoding="utf-8"))
            except Exception:
                continue
            v = r.get("test_metrics", {}).get("nse")
            if v is None or not np.isfinite(v):
                continue
            rec = {"nse": float(v), "kge": float(r.get("test_metrics", {}).get("kge", np.nan))}
            npz = list(dd.glob("**/predictions.npz"))
            if npz:
                d = np.load(npz[0]); rec.update(assinaturas(d["pred"], d["target"]))
            por.setdefault(dd.name, {})[seed] = rec
    return por


def main():
    argparse.ArgumentParser().parse_args()
    por = carrega()
    if not por:
        print(f"[vazio] {GRADE}"); return 1
    print(f"\n{'='*80}\n  CADEIA DO SKILL — simulação contínua por janela (test, média±dp; n seeds)\n{'='*80}")
    print(f"  {'modelo':<30}{'NSE':>8}{'KGE':>8}{'logNSE':>9}{'FHV%':>8}{'FLV%':>8}{'n':>4}")
    print("  " + "-" * 74)
    med = {}
    for disp in ORDEM:
        if disp not in por:
            print(f"  {disp:<30}{'—':>8}"); continue
        rows = list(por[disp].values())
        def col(k):
            xs = [x[k] for x in rows if k in x and np.isfinite(x[k])]
            return (np.mean(xs), np.std(xs, ddof=1) if len(xs) > 1 else 0.0) if xs else (np.nan, 0)
        nse = col("nse"); med[disp] = nse[0]
        print(f"  {disp:<30}{nse[0]:>+8.3f}{col('kge')[0]:>+8.3f}{col('lognse')[0]:>+9.3f}"
              f"{col('fhv')[0]:>8.1f}{col('flv')[0]:>8.1f}{len(rows):>4}")
    if {"LSTM_Lumped", "LSTM_Lumped_CalOnly", "LSTM_Lumped_RainOnly"} <= med.keys():
        print("\n  decomposição da fonte (lumped):")
        print(f"    valor da CHUVA      (Lumped−CalOnly)  = {med['LSTM_Lumped']-med['LSTM_Lumped_CalOnly']:+.3f}")
        print(f"    valor do CALENDÁRIO (Lumped−RainOnly) = {med['LSTM_Lumped']-med['LSTM_Lumped_RainOnly']:+.3f}")
    print(f"\n{'='*80}\n  CONTRASTES PAREADOS (Wilcoxon por seed comum, NSE de simulação)\n{'='*80}")
    for A, B, desc in CONTRASTES:
        if A not in por or B not in por:
            continue
        sc = sorted(set(por[A]) & set(por[B]))
        if len(sc) < 2:
            continue
        da = np.array([por[A][s]["nse"] for s in sc]); db = np.array([por[B][s]["nse"] for s in sc])
        diff = da - db; wins = int((diff > 0).sum()); n = len(sc)
        if n < 5:
            ps = f" | p OMITIDO (n={n}<5)"
        elif HAS_SCIPY and np.any(diff != 0):
            try:
                _, p = wilcoxon(da, db, alternative="greater"); ps = f" | p={p:.4f}"
            except Exception:
                ps = ""
        else:
            ps = ""
        print(f"  {desc:<44} Δ={diff.mean():+.3f} ({wins}/{n}){ps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
