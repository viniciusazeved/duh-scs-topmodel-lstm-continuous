#!/usr/bin/env python
"""Figures 4 to 7 and Supplementary Figures S3 to S5 of the manuscript.

Figures drawn from the decisive grid in ``outputs/grade_sim`` (telemetric forcing, horizon 1 h,
seeds 42 to 51; seeds 42 and 43 for Phys_DUH_Topmodel), in the Elsevier style of the manuscript
(Times New Roman, 9 pt):

    fig1_cadeia          Figure 4   NSE along the TOPMODEL path of the attribution chain
    fig_fdc              Figure 5   flow-duration curves of the test period (log axis)
    fig_serie            Figure 6   hydrographs of the test period
    fig_parametros       Figure 7   learned physical parameters across seeds
    fig_robustez         Figure S3  sample standard deviation of the NSE across seeds
    fig_recessao         Figure S4  deepest recession of the test period (log axis)
    fig_geradores_puros  Figure S5  the two pure generators, without the LSTM

Inputs: ``results.json`` (``test_metrics.nse`` and ``learned_params``) and ``predictions.npz``
(``pred`` and ``target``, shape (N, 1)) of each run, and the test split of
``data/dataset_58585000_telem.h5`` (dates of the scored hours). The simulated series of Figures 5,
6, S4 and S5 are seed ensembles: the predictions of the seeds averaged step by step.

Outputs, in ``outputs/figures``: ``<name>.png`` (300 dpi) for every figure, plus ``<name>.pdf``
for fig_recessao and fig_geradores_puros, the formats of the submitted LaTeX source (pdflatex
prefers a PDF to a PNG of the same name). With the environment of the runs (Python 3.13.10,
matplotlib 3.10.8, numpy 2.4.1, pandas 3.0.0, h5py 3.15.1, Pillow 12.1.0) and the Times New Roman
font, the PNG files are byte-identical to the submitted ones. The PDF files contain the same
objects as the submitted ones; because matplotlib embeds the fonts in the order of their file
paths, they are byte-identical apart from the CreationDate only when matplotlib is installed under
a path that sorts before the Times New Roman file (on the machine that made the figures, matplotlib
sat in the uv cache of the user profile and the font in the Windows font folder); otherwise the
same fonts appear in another order. Without Times New Roman, matplotlib falls back to DejaVu Serif.

Each figure also prints the numbers it draws, unrounded and rounded to two decimals from the
unrounded value; the labels inside the figures use the same two-decimal rounding.

Origin: ``gerar_figuras_sim_rodada7.py`` (revision round 7, 11 October 2026; SHA-256
229dcca4d698d6ba3c50058bfd76d2a43d6101ac045822d462ee767a79766cc9), the generator of the submitted
figures. Changes for this repository, none of which alters a figure: paths relative to the
repository root, with the output folder created in ``main()``; ``main()`` draws fig_parametros
and fig_recessao again (the round-7 ``main()`` had left them out); command-line options; type
hints; comments and console messages in English; console summaries of the drawn numbers.

Usage, from the repository root:
    python scripts/figures/gerar_figuras_sim.py
    python scripts/figures/gerar_figuras_sim.py --only fig_recessao fig_robustez
    python scripts/figures/gerar_figuras_sim.py --grade outputs/retrain --out outputs/fig_retrain
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
GRADE = ROOT / "outputs" / "grade_sim"  # seedNN/<configuration>/<class>/ for each run
OUT = ROOT / "outputs" / "figures"
TELEM_CANDS = [ROOT / "data" / "dataset_58585000_telem.h5"]  # forcing used by all models

# Elsevier style, the same as in the figures of the forecasting study.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "font.weight": "normal",
    "axes.labelweight": "normal", "axes.titleweight": "normal",
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.4,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "legend.fontsize": 8, "legend.frameon": False,
    "lines.linewidth": 1.0,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
GRAY = "#34495e"
# Colors by type of increment along the chain.
COR = {"info": "#4575b4", "space": "#9ecae1", "route": "#bdbdbd",
       "gen": "#1b9e77", "pe": "#7b3294", "neg": "#b03a2e"}

# Formats of the submitted LaTeX source: PNG for every figure, plus PDF for the two figures that
# the Supplementary Information includes as PDF (pdflatex prefers the PDF when both exist).
COM_PDF = {"fig_geradores_puros", "fig_recessao"}

NsePerSeed = dict[str, dict[int, float]]  # configuration -> {seed: test NSE}


def _num(x: float) -> str:
    """Unrounded value followed by its two-decimal rounding."""
    return f"{x:.6f} ({x:.2f})"


def _save(fig: Figure, name: str) -> None:
    """Save ``fig`` as ``OUT/<name>.png`` and, for the names in COM_PDF, as ``OUT/<name>.pdf``."""
    fig.savefig(OUT / f"{name}.png")
    if name in COM_PDF:
        fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"[OK] {name}" + (" (.png, .pdf)" if name in COM_PDF else " (.png)"))


def _nice_logy(ax: Axes, ticks: list[float]) -> None:
    """Log y axis with plain numeric tick labels (20, 30, 50, ...) and minor ticks."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=2)


def carrega() -> NsePerSeed:
    """Test NSE (``test_metrics.nse``) of every run: ``por[configuration][seed]``."""
    por = {}
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
            por.setdefault(dd.name, {})[seed] = float(v)
    return por


def _mu(por: NsePerSeed, m: str) -> float:
    """Mean over seeds."""
    xs = [v for v in por.get(m, {}).values() if np.isfinite(v)]
    return float(np.mean(xs)) if xs else np.nan


def _sd(por: NsePerSeed, m: str) -> float:
    """Sample standard deviation over seeds (ddof = 1)."""
    xs = [v for v in por.get(m, {}).values() if np.isfinite(v)]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else np.nan


def _nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency over the finite pairs."""
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    den = np.sum((o - o.mean()) ** 2)
    return float(1 - np.sum((s - o) ** 2) / den) if den > 0 else np.nan


# ---------------------------------------------------------------- Figure 4: attribution chain
def fig1_cadeia() -> None:
    """Figure 4: decomposition of the mean test NSE along the TOPMODEL path of the chain.

    Rungs 2, 1, 3, 4, 5, 9 and 10 of Table 3, with rung 13 (Phys_DUH_Topmodel) as an annotation.
    Bars: gain of each rung over the previous one (for the two single-input baselines, the NSE
    itself), labelled with the increment; line with diamonds: ensemble NSE per configuration.
    """
    por = carrega()
    spine = [
        ("LSTM_Lumped_RainOnly",     "Rainfall\nonly",           "neg",   "ref"),
        ("LSTM_Lumped_CalOnly",      "Calendar\nonly",           "info",  "start"),
        ("LSTM_Lumped",              "+ rainfall\n(coupling)",   "info",  "float"),
        ("LSTM",                     "+ spatial\n(distributed)", "space", "float"),
        ("LSTM_DUH",                 "+ DUH\nrouting",           "route", "float"),
        ("LSTM_DUH_Topmodel",        "+ TOPMODEL\ngeneration",   "gen",   "float"),
        ("LSTM_DUH_Topmodel_PeOnly", "+ $P_e$-only",             "pe",    "float"),
    ]
    vals = [_mu(por, m) for m, _, _, _ in spine]
    if any(np.isnan(vals)):
        print("[skip] fig1_cadeia: missing configurations",
              [s[0] for s, v in zip(spine, vals) if np.isnan(v)])
        return
    phys = _mu(por, "Phys_DUH_Topmodel")   # about -0.40: the pure-physics baseline collapses
    ens = [_nse_ensemble(m) for m, _, _, _ in spine]   # ensemble NSE per configuration

    print("[fig1_cadeia] Figure 4: mean test NSE over seeds, bar label and ensemble NSE")
    prev_v = None
    for (m, _, _, kind), v, e in zip(spine, vals, ens):
        d = v if kind in ("ref", "start") else v - prev_v
        if kind != "ref":
            prev_v = v
        print(f"  {m:<25} seeds {len(por[m]):>2}  mean {_num(v)}  label {_num(d)}"
              f"  ensemble {_num(e)}")
    print(f"  {'Phys_DUH_Topmodel':<25} seeds {len(por.get('Phys_DUH_Topmodel', {})):>2}"
          f"  mean {_num(phys)}  (annotation)")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(spine))
    prev = None
    for i, (m, lab, cat, kind) in enumerate(spine):
        v = vals[i]
        bottom = 0.0 if kind in ("ref", "start") else prev
        ax.bar(i, v - bottom, bottom=bottom, width=0.62, color=COR[cat],
               edgecolor="0.3", linewidth=0.4, zorder=3)
        if kind == "float":   # dashed connector from the previous top to the base of this bar
            ax.plot([i - 1 + 0.31, i - 0.31], [prev, prev], color="0.55",
                    lw=0.7, ls=(0, (3, 2)), zorder=2)
        if kind == "ref":     # isolated negative bar (rainfall only): label below it
            ax.annotate(f"{v:+.2f}", (i, v), xytext=(0, -3), textcoords="offset points",
                        ha="center", va="top", fontsize=7.5, color=COR["neg"])
        else:
            d = v if kind == "start" else v - prev
            # Round 7 (layout, 11 Oct 2026): at rungs 1 and 2 the ensemble line crossed the
            # label, which moves to the right of the diamond.
            lado = i in (1, 2)
            ax.annotate(f"{d:+.2f}", (i, max(prev if prev is not None else 0.0, v)),
                        xytext=(8, 3) if lado else (0, 3), textcoords="offset points",
                        ha="left" if lado else "center",
                        va="bottom", fontsize=7.5, color="0.2", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
            prev = v
    # Overlay: ensemble NSE (mean of the seeds' predictions) per configuration.
    ax.plot(x, ens, marker="D", ms=4.5, lw=0.9, color="0.15",
            markerfacecolor="white", markeredgecolor="0.15", zorder=6)
    ax.annotate(f"ensemble {ens[-1]:.2f}", (len(spine) - 1, ens[-1]), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=7.5,
                fontweight="bold", color="0.15")
    ax.axhline(0, color="0.5", lw=0.6)
    # The pure-physics baseline is a note in the corner, off scale, so as not to flatten the chain.
    ax.text(0.02, 0.96, f"Physics only (DUH-TOPMODEL, no LSTM): {phys:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.8, color=COR["neg"])
    ax.legend(handles=[Patch(facecolor="0.6", edgecolor="0.3", label="Mean of 10 seeds (bars)"),
                       Line2D([], [], marker="D", color="0.15", markerfacecolor="white",
                              lw=0.9, label="Ensemble (mean of predictions)")],
              loc="lower right", fontsize=7, frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _, _ in spine], fontsize=7.6)
    ax.set_ylabel("NSE (continuous simulation)")
    ax.set_ylim(min(-0.15, min(vals) - 0.05), max(vals) + 0.12)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig1_cadeia")


# ---------------------------------------------------------------- Figure S3: seed-to-seed spread
def fig_robustez() -> None:
    """Figure S3: sample standard deviation of the test NSE across the 10 seeds.

    Selected configurations, including the routing-only control with fixed parameters
    (LSTM_DUH_Fixed); the remaining rungs are in Table 3.
    """
    por = carrega()
    order = [
        ("LSTM",                      "LSTM (neural)",            "neutral"),
        ("LSTM_DUH",                  "+ DUH",                    "neutral"),
        # Round 7 (11 Oct 2026): the routing-only control, without a generator, has a similar
        # spread (0.06).
        ("LSTM_DUH_Fixed",            "+ DUH (fixed)",            "neutral"),
        ("LSTM_DUH_SCS",              "+ SCS-CN",                 "scs"),
        ("LSTM_DUH_Topmodel",         "+ TOPMODEL",               "top"),
        ("LSTM_DUH_Topmodel_Baseflow", "+ TOPMODEL + baseflow",   "top"),
        ("LSTM_DUH_Topmodel_PeOnly",  "+ TOPMODEL ($P_e$-only)",  "pe"),
    ]
    cor_map = {"neutral": GRAY, "scs": "#2e75b6", "top": "#1b9e77", "pe": "#7b3294"}
    rows = [(lab, _sd(por, m), cor_map[c]) for m, lab, c in order if np.isfinite(_sd(por, m))]
    if not rows:
        print("[skip] fig_robustez: no multi-seed results")
        return
    print("[fig_robustez] Figure S3: sample standard deviation (ddof = 1) of the test NSE")
    for m, _, _ in order:
        if np.isfinite(_sd(por, m)):
            print(f"  {m:<27} seeds {len(por[m]):>2}  sd {_num(_sd(por, m))}")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    y = np.arange(len(rows))[::-1]   # first row (LSTM) on top
    ax.barh(y, [s for _, s, _ in rows], color=[c for _, _, c in rows],
            alpha=0.9, edgecolor="0.3", linewidth=0.4)
    for yi, (_, s, _) in zip(y, rows):
        ax.annotate(f"{s:.2f}", (s, yi), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color="0.2")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _, _ in rows], fontsize=8)
    ax.set_xlabel("Std. of NSE across 10 seeds")
    ax.set_xlim(0, max(s for _, s, _ in rows) * 1.18)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "fig_robustez")


# ---------------------------------------------------------------- seed ensembles
def _pred_mean(model: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Seed-ensemble simulated series (N,) and the observed target (N,), the same for all runs."""
    preds, target = [], None
    for sd in sorted(GRADE.glob("seed*")):
        dd = sd / model
        npz = list(dd.glob("**/predictions.npz")) if dd.exists() else []
        if not npz:
            continue
        d = np.load(npz[0])
        preds.append(d["pred"][:, 0])
        if target is None:
            target = d["target"][:, 0]
    if not preds:
        return None, None
    n = min(len(p) for p in preds)
    return np.mean([p[:n] for p in preds], axis=0), target[:n]


def _nse_ensemble(model: str) -> float:
    """NSE of the seed ensemble: the seeds' predictions averaged step by step into one series."""
    P, target = _pred_mean(model)
    return _nse(P, target) if P is not None else np.nan


def _fdc(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flow-duration curve: exceedance probability (%, Weibull) and flows in decreasing order."""
    x = x[np.isfinite(x)]
    xs = np.sort(x)[::-1]
    p = np.arange(1, len(xs) + 1) / (len(xs) + 1) * 100.0
    return p, xs


# ---------------------------------------------------------------- Figure 5: flow-duration curves
def fig_fdc() -> None:
    """Figure 5: flow-duration curves of the test period on a log axis.

    Observed, best hybrid, pure physics (Phys_DUH_Topmodel) and the worst neural model (rainfall
    only), as seed ensembles. The shaded band marks exceedance above 70%, the range of FLV.
    """
    Pb, obs = _pred_mean("LSTM_DUH_Topmodel_PeOnly")   # best hybrid; obs = target (all runs)
    Pw, _ = _pred_mean("Phys_DUH_Topmodel")                  # pure physics
    Pn, _ = _pred_mean("LSTM_Lumped_RainOnly")          # worst neural (rainfall, no calendar)
    if obs is None or Pb is None or Pw is None or Pn is None:
        print("[skip] fig_fdc: predictions not found")
        return
    nb, nw, nn = _nse(Pb, obs), _nse(Pw, obs), _nse(Pn, obs)
    print(f"[fig_fdc] Figure 5: ensemble NSE over {len(obs)} hours: best hybrid {_num(nb)}, "
          f"pure physics {_num(nw)}, worst neural {_num(nn)}")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.axvspan(70, 100, color="0.93", zorder=0)
    for arr, c, lw, lab, z in [
            (obs, "black", 1.7, "Observed", 5),
            (Pb, "#2e75b6", 1.2,
             f"Best hybrid (LSTM + DUH-TOPMODEL, $P_e$-only): NSE = {nb:.2f}", 4),
            (Pw, "#e67e22", 1.2, f"Physics only (DUH-TOPMODEL, no LSTM): NSE = {nw:.2f}", 3),
            (Pn, "#7f8c8d", 1.1, f"Worst neural (LSTM, rainfall only): NSE = {nn:.2f}", 2)]:
        pp, ss = _fdc(arr)
        ax.plot(pp, ss, color=c, lw=lw, label=lab, zorder=z)
    _nice_logy(ax, [20, 30, 40, 50, 60, 80, 100, 150, 200])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("Discharge (m$^3$ s$^{-1}$)")
    ax.annotate("low flows (FLV)", (85, ax.get_ylim()[0]), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7, color="0.45")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(axis="both", which="both", color="0.9", lw=0.3)
    fig.tight_layout()
    _save(fig, "fig_fdc")


# ---------------------------------------------------------------- Figure 7: learned parameters
def _learned(model: str, keys: list[str]) -> np.ndarray:
    """Values across seeds of the first key of ``keys`` present in ``learned_params``."""
    vals = []
    for sd in sorted(GRADE.glob("seed*")):
        rj = list((sd / model).glob("**/results.json")) if (sd / model).exists() else []
        if not rj:
            continue
        lp = json.loads(rj[0].read_text(encoding="utf-8")).get("learned_params", {}) or {}
        for k in keys:
            if k in lp:
                try:
                    vals.append(float(lp[k]))
                    break
                except (TypeError, ValueError):
                    pass
    return np.array(vals, dtype=float)


def fig_parametros() -> None:
    """Figure 7: learned physical parameters (global scalars) across the 10 seeds.

    LSTM_DUH_SCS (SCS) and LSTM_DUH_Topmodel (TOP): one point per seed, mean and sample standard
    deviation (ddof = 1) as marker and error bar, and reference values as dashed red lines.
    """
    SCS, TOP = "LSTM_DUH_SCS", "LSTM_DUH_Topmodel"
    CSCS, CTOP = "#2e75b6", "#1b9e77"
    panels = [
        (r"(a) $\lambda$ (SCS-CN)", [(SCS, ["lambda_scs", "lambda"], CSCS, "SCS")],
         [(0.20, "classic 0.20"), (0.05, "tropical ~0.05")]),
        (r"(b) $t_c$ scale", [(SCS, ["tc_scale"], CSCS, "SCS"), (TOP, ["tc_scale"], CTOP, "TOP")],
         [(1.0, "init 1.0")]),
        (r"(c) $\sigma$ (h)", [(SCS, ["sigma"], CSCS, "SCS"), (TOP, ["sigma"], CTOP, "TOP")],
         [(3.0, "init 3.0")]),
        (r"(d) TOPMODEL $m$ (mm)", [(TOP, ["topmodel_m"], CTOP, "TOP")], []),
    ]
    if not any(len(_learned(m, ks)) for _, series, _ in panels for m, ks, _, _ in series):
        print("[skip] fig_parametros: learned_params not found")
        return
    print("[fig_parametros] Figure 7: learned parameters, mean and sample standard deviation "
          "(ddof = 1)")
    for _, series, _ in panels:
        for model, ks, _, _ in series:
            v = _learned(model, ks)
            if len(v):
                print(f"  {ks[0]:<11} {model:<18} seeds {len(v):>2}  mean {_num(float(v.mean()))}"
                      f"  sd {_num(float(v.std(ddof=1)))}"
                      f"  min {_num(float(v.min()))}  max {_num(float(v.max()))}")
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5))
    for ax, (title, series, refs) in zip(axes, panels):
        ticks, labels = [], []
        for i, (model, ks, color, lab) in enumerate(series):
            v = _learned(model, ks)
            if not len(v):
                continue
            jit = np.linspace(-0.06, 0.06, len(v)) if len(v) > 1 else np.zeros(1)
            ax.scatter(np.full(len(v), i) + jit, v, s=12, color=color, alpha=0.6,
                       edgecolor="none", zorder=3)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="o", ms=5, color=color,
                        capsize=3, elinewidth=0.8, zorder=4)
            ticks.append(i)
            labels.append(lab)
        for rv, rlab in refs:
            ax.axhline(rv, color="#c0392b", lw=0.8, ls="--", zorder=2)
            ax.annotate(rlab, (1.0, rv), xycoords=("axes fraction", "data"),
                        xytext=(-2, 2), textcoords="offset points", fontsize=6,
                        color="#c0392b", va="bottom", ha="right")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlim(-0.5, max(0, len(series) - 1) + 0.5)
    axes[0].set_ylabel("Learned value")
    fig.tight_layout()
    _save(fig, "fig_parametros")


# ---------------------------------------------------------------- Figure S4: recession
def fig_recessao() -> None:
    """Figure S4: deepest recession of the test period, on a log axis.

    Observed discharge, the pure LSTM (rung 4) and the SCS-CN and TOPMODEL generators with the
    shortcut closed (P_e only; rungs 8 and 10), as seed ensembles. The predictions are aligned
    with dates by rebuilding the valid windows (L = 240 h, H = 1) and checking the observed series
    against the target stored in ``predictions.npz``.
    """
    L, H = 240, 1
    P_top, target = _pred_mean("LSTM_DUH_Topmodel_PeOnly")
    if P_top is None:
        print("[skip] fig_recessao: predictions not found")
        return
    telem = next((c for c in TELEM_CANDS if Path(c).exists()), None)
    if telem is None:
        print("[skip] fig_recessao: telemetric h5 not found")
        return
    with h5py.File(telem, "r") as f:
        Q = f["test/streamflow"][:].astype(float)
        Pp = f["test/precipitation"][:].astype(float)
        ts = f["test/timestamps"][:].astype("int64")
    T = len(Q)
    tt = np.arange(L - 1, T - 1 - H + 1)
    qn = np.concatenate([[0], np.cumsum(np.isnan(Q))])
    pn = np.concatenate([[0], np.cumsum(np.isnan(Pp).any(axis=1))])
    alvo = (qn[tt + 1 + H] - qn[tt + 1]) > 0
    look = (pn[tt + 1] - pn[tt - L + 1]) > 0
    valid = tt[~(alvo | look)]
    Qrec, tsrec = Q[valid + H], ts[valid + H]
    n = min(len(Qrec), len(target))
    diff = np.nanmax(np.abs(Qrec[:n] - target[:n]))
    print(f"[fig_recessao] alignment of the observed series with the npz target: "
          f"max|diff| = {diff:.4g} (n = {n})")
    if diff > 1.0:
        print("[fig_recessao] WARNING: poor alignment (> 1 m3/s); the dates may be wrong")

    t = pd.to_datetime(tsrec[:n], unit="s", utc=True).tz_localize(None)
    obs = pd.Series(target[:n], index=t).sort_index()
    models = {"LSTM": ("Pure LSTM (rung 4)", "#7f8c8d", "--"),
              "LSTM_DUH_SCS_PeOnly": (r"+ SCS-CN, $P_e$ only (rung 8)", "#2e75b6", "-"),
              "LSTM_DUH_Topmodel_PeOnly": (r"+ TOPMODEL, $P_e$ only (rung 10)", "#1b9e77", "-")}
    sims = {}
    for m in models:
        P, _ = _pred_mean(m)
        if P is not None:
            sims[m] = pd.Series(P[:n], index=t).sort_index()
    # Window: the 60-day trailing mean of the daily discharge reaches its minimum on day `centro`;
    # the plot spans 55 days before to 20 days after it.
    daily = obs.resample("1D").mean()
    centro = daily.rolling(60, min_periods=30).mean().idxmin()
    a, b = centro - pd.Timedelta(days=55), centro + pd.Timedelta(days=20)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    o = obs[(obs.index >= a) & (obs.index <= b)]
    ax.plot(o.index, o.values, color="black", lw=1.5, label="Observed", zorder=5)
    # Bias over the 30% lowest observed flows of the window (diagnostic for the caption).
    lim30 = np.nanpercentile(o.values, 30)
    baixo = o.values <= lim30
    print(f"[fig_recessao] Figure S4: window {a.date()} to {b.date()}; observed minimum "
          f"{_num(float(np.nanmin(o.values)))} m3/s; 30th percentile {_num(float(lim30))} m3/s "
          f"({int(baixo.sum())} h)")
    for m, (lab, c2, ls) in models.items():
        if m in sims:
            s = sims[m][(sims[m].index >= a) & (sims[m].index <= b)]
            ax.plot(s.index, s.values, color=c2, lw=1.0, ls=ls, label=lab, zorder=4)
            sv = s.reindex(o.index).values
            vies = (100.0 * (np.nanmean(sv[baixo]) - np.nanmean(o.values[baixo]))
                    / np.nanmean(o.values[baixo]))
            print(f"  {m:<25} minimum {_num(float(np.nanmin(sv)))} m3/s; bias over the 30% "
                  f"lowest observed flows {_num(float(vies))} %")
    _nice_logy(ax, [15, 20, 30, 40, 50, 60, 70])
    ax.set_ylim(15, 75)   # fixed range: every tick from 15 to 70 visible
    ax.set_ylabel("Discharge (m$^3$ s$^{-1}$)")
    ax.set_xlabel("Date (test period)")
    # Round 7 (layout, 11 Oct 2026): legend above the axes, in one row; inside them it covered the
    # peaks of about 70 m3/s.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=4, frameon=False)
    ax.grid(axis="both", which="both", color="0.9", lw=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "fig_recessao")


# ---------------------------------------------------------------- Figure 6: test-period series
def _obs_e_datas() -> tuple[np.ndarray | None, pd.DatetimeIndex | None]:
    """Observed discharge (N,) and dates (N,) of the valid test windows (L = 240 h, H = 1)."""
    L, H = 240, 1
    telem = next((c for c in TELEM_CANDS if Path(c).exists()), None)
    if telem is None:
        return None, None
    with h5py.File(telem, "r") as f:
        Q = f["test/streamflow"][:].astype(float)
        Pp = f["test/precipitation"][:].astype(float)
        ts = f["test/timestamps"][:].astype("int64")
    T = len(Q)
    tt = np.arange(L - 1, T - 1 - H + 1)
    qn = np.concatenate([[0], np.cumsum(np.isnan(Q))])
    pn = np.concatenate([[0], np.cumsum(np.isnan(Pp).any(axis=1))])
    valid = tt[~(((qn[tt + 1 + H] - qn[tt + 1]) > 0) | ((pn[tt + 1] - pn[tt - L + 1]) > 0))]
    obs = Q[valid + H]
    datas = pd.to_datetime(ts[valid + H], unit="s", utc=True).tz_localize(None)
    return obs, datas


def fig_serie() -> None:
    """Figure 6: test-period hydrographs, as seed ensembles.

    Observed, best hybrid, pure physics (Phys_DUH_Topmodel) and the worst neural model (rainfall
    only).
    """
    obs, datas = _obs_e_datas()
    if obs is None:
        print("[skip] fig_serie: telemetric h5 not found")
        return
    Pb, tgt = _pred_mean("LSTM_DUH_Topmodel_PeOnly")
    Pw, _ = _pred_mean("Phys_DUH_Topmodel")
    Pn, _ = _pred_mean("LSTM_Lumped_RainOnly")
    if Pb is None or Pw is None or Pn is None:
        print("[skip] fig_serie: predictions not found")
        return
    n = min(len(obs), len(datas), len(Pb), len(Pw), len(Pn))
    # Sanity check: the rebuilt observed series must match the target stored in the npz.
    d = np.nanmax(np.abs(obs[:n] - tgt[:n]))
    print(f"[fig_serie] alignment of the observed series with the npz target: max|diff| = {d:.4g}"
          f" (n = {n}, {datas[0]} to {datas[n - 1]} UTC)")
    nb, nw, nn = _nse(Pb[:n], obs[:n]), _nse(Pw[:n], obs[:n]), _nse(Pn[:n], obs[:n])
    print(f"[fig_serie] Figure 6: ensemble NSE: best hybrid {_num(nb)}, pure physics {_num(nw)}, "
          f"worst neural {_num(nn)}")
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.plot(datas[:n], obs[:n], color="black", lw=0.7, label="Observed", zorder=4)
    ax.plot(datas[:n], Pb[:n], color="#2e75b6", lw=0.7, zorder=3,
            label=f"Best hybrid (LSTM + DUH-TOPMODEL, $P_e$-only): NSE = {nb:.2f}")
    ax.plot(datas[:n], Pw[:n], color="#e67e22", lw=0.7, zorder=2,
            label=f"Physics only (DUH-TOPMODEL, no LSTM): NSE = {nw:.2f}")
    ax.plot(datas[:n], Pn[:n], color="#7f8c8d", lw=0.7, zorder=1,
            label=f"Worst neural (LSTM, rainfall only): NSE = {nn:.2f}")
    ax.set_ylabel("Discharge (m$^3$ s$^{-1}$)")
    ax.set_xlabel("Date (test period)")
    # Round 7 (layout, 11 Oct 2026): in the upper right corner the legend covered the December peak.
    ax.legend(loc="upper center", fontsize=7.5)
    ax.grid(axis="both", color="0.9", lw=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "fig_serie")


# ---------------------------------------------------------------- Figure S5: pure generators
def fig_geradores_puros() -> None:
    """Figure S5: the two pure generators, run without the LSTM, over the test period.

    Linear axis. Phys_DUH_SCS (ensemble of 10 seeds) and Phys_DUH_Topmodel (2 seeds) against the
    observed series; both have a negative ensemble NSE (rungs 12 and 13 of Table 3).
    """
    obs, datas = _obs_e_datas()
    if obs is None:
        print("[skip] fig_geradores_puros: telemetric h5 not found")
        return
    Ps, tgt = _pred_mean("Phys_DUH_SCS")
    Pt, _ = _pred_mean("Phys_DUH_Topmodel")
    if Ps is None or Pt is None:
        print("[skip] fig_geradores_puros: predictions of the pure generators not found")
        return
    n = min(len(obs), len(datas), len(Ps), len(Pt), len(tgt))
    d = np.nanmax(np.abs(obs[:n] - tgt[:n]))
    print(f"[fig_geradores_puros] alignment of the observed series with the npz target: "
          f"max|diff| = {d:.4g} (n = {n})")
    ns, nt = _nse(Ps[:n], obs[:n]), _nse(Pt[:n], obs[:n])
    n_scs = sum(1 for sd in sorted(GRADE.glob("seed*"))
                if list((sd / "Phys_DUH_SCS").glob("**/predictions.npz")))
    n_top = sum(1 for sd in sorted(GRADE.glob("seed*"))
                if list((sd / "Phys_DUH_Topmodel").glob("**/predictions.npz")))
    print(f"[fig_geradores_puros] Figure S5: ensemble NSE: Phys_DUH_SCS ({n_scs} seeds) "
          f"{_num(ns)}, Phys_DUH_Topmodel ({n_top} seeds) {_num(nt)}")

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.plot(datas[:n], obs[:n], color="black", lw=0.7, label="Observed", zorder=4)
    ax.plot(datas[:n], Ps[:n], color="#e67e22", lw=0.7, zorder=3,
            label=f"DUH-SCS-CN, no LSTM ({n_scs} seeds): NSE = {ns:.2f}")
    ax.plot(datas[:n], Pt[:n], color="#1b9e77", lw=0.7, zorder=2,
            label=f"DUH-TOPMODEL, no LSTM ({n_top} seeds): NSE = {nt:.2f}")
    ax.set_ylabel("Discharge (m$^3$ s$^{-1}$)")
    ax.set_xlabel("Date (test period)")
    # Round 7 (layout, 11 Oct 2026): in the upper right corner the legend covered the December peak.
    ax.legend(loc="upper center", fontsize=7.5)
    ax.grid(axis="both", color="0.9", lw=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "fig_geradores_puros")


FIGURES: dict[str, Callable[[], None]] = {
    "fig1_cadeia": fig1_cadeia,                  # Figure 4
    "fig_robustez": fig_robustez,                # Figure S3
    "fig_fdc": fig_fdc,                          # Figure 5
    "fig_parametros": fig_parametros,            # Figure 7
    "fig_recessao": fig_recessao,                # Figure S4
    "fig_serie": fig_serie,                      # Figure 6
    "fig_geradores_puros": fig_geradores_puros,  # Figure S5
}


def main(argv: list[str] | None = None) -> None:
    global GRADE, OUT
    parser = argparse.ArgumentParser(
        description="Figures 4 to 7 and Supplementary Figures S3 to S5 of the manuscript.")
    parser.add_argument("--grade", type=Path, default=GRADE,
                        help="folder with the seedNN runs (default: outputs/grade_sim)")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output folder (default: outputs/figures)")
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), metavar="NAME",
                        help="draw only these figures: " + ", ".join(FIGURES))
    args = parser.parse_args(argv)
    GRADE, OUT = args.grade, args.out
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES.items():
        if args.only is None or name in args.only:
            draw()


if __name__ == "__main__":
    main()
