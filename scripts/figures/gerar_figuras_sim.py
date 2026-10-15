#!/usr/bin/env python
# RODADA 7 (2026-10-10), decisao D3 do Vinicius: metricas com duas casas, formatadas a partir
# do valor bruto (float com '.2f'; nunca arredondar um numero de tres casas ja impresso).
# Copia de D:/Artigo_JOH/scripts/gerar_figuras_sim.py (sha256 e7afa37aa1416bbd...,
# manifesto externo do congelamento de 29/09/2026); o original NAO foi alterado.
# Alteracoes (todas marcadas "rodada 7"): formatos '.3f' -> '.2f' nos rotulos de metrica
# dentro das imagens; OUT no staging da rodada 7; _save grava so os formatos que o LaTeX usa;
# main() roda so as 5 figuras com rotulos de tres casas. Nada de dado, cor, layout ou estatistica.
# Rodar (sem tocar no projeto congelado):
#   uv run --no-project --python 3.13.10 --with matplotlib==3.10.8 --with numpy==2.4.1 \
#     --with pandas==3.0.0 --with h5py==3.15.1 python gerar_figuras_sim_rodada7.py
"""Figuras do artigo de SIMULACAO continua (estilo Elsevier herdado do forecasting),
a partir da grade_sim (telem, horizonte 1, 10 sementes).

  fig1_cadeia      -- waterfall de Delta-NSE: de onde vem o skill (peca central) [Fig.1, RQ1]
  fig_robustez     -- dp do NSE entre sementes: fisica como regularizador        [Fig., RQ3 pilar 1]
  (proximas: fig_fdc, fig_recessao, fig_parametros; backbones/produtos esperam o ABC)

Fonte: D:/TTD_SCS_LSTM/ablacao_skill/outputs/grade_sim/seed<N>/<MODELO>/**/results.json
       (test_metrics.nse = NSE de simulacao @1h) e predictions.npz (pred/target (N,1)).
Saida: D:/Artigo_JOH/artigo_simulacao/figuras/*.{png,pdf}

Rodar: uv run --project D:/TTD_SCS_LSTM python scripts/gerar_figuras_sim.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

GRADE = Path(r"D:\TTD_SCS_LSTM\ablacao_skill\outputs\grade_sim")
OUT = Path(r"D:\Artigo_JOH\artigo_simulacao\revisao_paulo\rodada7_2026-10-09\staging\figuras")  # rodada 7
OUT.mkdir(parents=True, exist_ok=True)

# ---- estilo Elsevier (identico ao gerar_figuras.py do forecasting) ----
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
# cores por tipo de incremento na cadeia
COR = {"info": "#4575b4", "space": "#9ecae1", "route": "#bdbdbd",
       "gen": "#1b9e77", "pe": "#7b3294", "neg": "#b03a2e"}


# rodada 7: so os formatos presentes em overleaf_rev02/figuras (o pdflatex prefere o .pdf
# quando ha os dois); das figuras regeradas, so fig_geradores_puros tem .pdf la
COM_PDF = {"fig_geradores_puros", "fig_recessao"}  # rodada 7: o SI carrega estas duas em PDF


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    if name in COM_PDF:  # rodada 7
        fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"[OK] {name}")


def _nice_logy(ax, ticks):
    """Eixo y log com divisoes NUMERICAS (20, 30, 50...) em vez do '10^2' nu, + ticks menores."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=2)


def carrega():
    """por[modelo][seed] = nse de simulacao (@1h)."""
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


def _mu(por, m):
    xs = [v for v in por.get(m, {}).values() if np.isfinite(v)]
    return float(np.mean(xs)) if xs else np.nan


def _sd(por, m):
    xs = [v for v in por.get(m, {}).values() if np.isfinite(v)]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else np.nan


def _nse(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    den = np.sum((o - o.mean()) ** 2)
    return float(1 - np.sum((s - o) ** 2) / den) if den > 0 else np.nan


# ---------------------------------------------------------------- Fig.1 -- cadeia (waterfall)
def fig1_cadeia():
    """Waterfall de NSE ao longo da cadeia de atribuicao: o skill nasce do acoplamento
    chuva x calendario; espaco modesto; roteamento ~0; geracao pequena. Controles fora
    da espinha (chuva sozinha, fisica pura) falham -> mostrados como referencia."""
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
        print("[skip] fig1: faltam modelos", [s[0] for s, v in zip(spine, vals) if np.isnan(v)])
        return
    phys = _mu(por, "Phys_DUH_Topmodel")   # ~ -0.40 (colapsa; gerador casado com o melhor)
    ens = [_nse_ensemble(m) for m, _, _, _ in spine]   # NSE do ensemble por configuracao

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(spine))
    prev = None
    for i, (m, lab, cat, kind) in enumerate(spine):
        v = vals[i]
        bottom = 0.0 if kind in ("ref", "start") else prev
        ax.bar(i, v - bottom, bottom=bottom, width=0.62, color=COR[cat],
               edgecolor="0.3", linewidth=0.4, zorder=3)
        if kind == "float":   # conector tracejado do topo anterior a base deste
            ax.plot([i - 1 + 0.31, i - 0.31], [prev, prev], color="0.55",
                    lw=0.7, ls=(0, (3, 2)), zorder=2)
        if kind == "ref":     # barra negativa isolada (chuva sozinha): rotulo abaixo
            ax.annotate(f"{v:+.2f}", (i, v), xytext=(0, -3), textcoords="offset points",
                        ha="center", va="top", fontsize=7.5, color=COR["neg"])
        else:
            d = v if kind == "start" else v - prev
            # rodada 7 (layout, 11/10): nos degraus 1 e 2 a linha do ensemble cruzava o
            # rotulo; ele passa para a direita do losango
            lado = i in (1, 2)
            ax.annotate(f"{d:+.2f}", (i, max(prev if prev is not None else 0.0, v)),
                        xytext=(8, 3) if lado else (0, 3), textcoords="offset points",
                        ha="left" if lado else "center",
                        va="bottom", fontsize=7.5, color="0.2", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
            prev = v
    # overlay: NSE do ENSEMBLE (media das previsoes das seeds) por configuracao
    ax.plot(x, ens, marker="D", ms=4.5, lw=0.9, color="0.15",
            markerfacecolor="white", markeredgecolor="0.15", zorder=6)
    ax.annotate(f"ensemble {ens[-1]:.2f}", (len(spine) - 1, ens[-1]), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=7.5,
                fontweight="bold", color="0.15")
    ax.axhline(0, color="0.5", lw=0.6)
    # fisica pura colapsa: nota no canto (fora de escala p/ nao achatar a espinha)
    ax.text(0.02, 0.96, f"Physics only (DUH-TOPMODEL, no LSTM): {phys:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.8, color=COR["neg"])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
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


# ---------------------------------------------------------------- Fig. robustez (pilar 1)
def fig_robustez():
    """dp do NSE entre as 10 sementes: a fisica com estado (TOPMODEL) corta ~metade do
    ruido de inicializacao do LSTM puro -- a fisica como regularizador (pilar 1)."""
    por = carrega()
    order = [
        ("LSTM",                      "LSTM (neural)",            "neutral"),
        ("LSTM_DUH",                  "+ DUH",                    "neutral"),
        # rodada 7 (loop 2, 11/10): o controle so de roteamento, sem gerador, tem DP parecida (0,06)
        ("LSTM_DUH_Fixed",            "+ DUH (fixed)",            "neutral"),
        ("LSTM_DUH_SCS",              "+ SCS-CN",                 "scs"),
        ("LSTM_DUH_Topmodel",         "+ TOPMODEL",               "top"),
        ("LSTM_DUH_Topmodel_Baseflow","+ TOPMODEL + baseflow",    "top"),
        ("LSTM_DUH_Topmodel_PeOnly",  "+ TOPMODEL ($P_e$-only)",  "pe"),
    ]
    cor_map = {"neutral": GRAY, "scs": "#2e75b6", "top": "#1b9e77", "pe": "#7b3294"}
    rows = [(lab, _sd(por, m), cor_map[c]) for m, lab, c in order if np.isfinite(_sd(por, m))]
    if not rows:
        print("[skip] fig_robustez: sem multiseed")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    y = np.arange(len(rows))[::-1]   # primeiro (LSTM) no topo
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


# ---------------------------------------------------------------- Fig.2 -- FDC por arquitetura
def _pred_mean(model):
    """Serie simulada media sobre as sementes (N,) e a obs (N,) do alvo (identica entre modelos)."""
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


def _nse_ensemble(model):
    """NSE do ENSEMBLE: media das previsoes das seeds -> 1 serie -> 1 NSE (> media das seeds)."""
    P, target = _pred_mean(model)
    return _nse(P, target) if P is not None else np.nan


def _fdc(x):
    x = x[np.isfinite(x)]
    xs = np.sort(x)[::-1]
    p = np.arange(1, len(xs) + 1) / (len(xs) + 1) * 100.0
    return p, xs


def fig_fdc():
    """Curva de permanencia observada vs simulada por arquitetura (escala log). A faixa de
    baixas (excedencia >70%, FLV) e onde a fisica com estado (TOPMODEL) acompanha melhor."""
    Pb, obs = _pred_mean("LSTM_DUH_Topmodel_PeOnly")   # melhor; obs = target (identico)
    Pw, _ = _pred_mean("Phys_DUH_Topmodel")                  # pior fisico (fisica pura)
    Pn, _ = _pred_mean("LSTM_Lumped_RainOnly")          # pior neural (so chuva, sem calendario)
    if obs is None or Pb is None or Pw is None or Pn is None:
        print("[skip] fig_fdc: sem predictions")
        return
    nb, nw, nn = _nse(Pb, obs), _nse(Pw, obs), _nse(Pn, obs)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.axvspan(70, 100, color="0.93", zorder=0)
    for arr, c, lw, lab, z in [
            (obs, "black", 1.7, "Observed", 5),
            (Pb, "#2e75b6", 1.2, f"Best hybrid (LSTM + DUH-TOPMODEL, $P_e$-only): NSE = {nb:.2f}", 4),
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


# ---------------------------------------------------------------- Fig.5 -- parametros aprendidos
def _learned(model, keys):
    """Vetor (sobre sementes) do 1o `key` presente em learned_params."""
    vals = []
    for sd in sorted(GRADE.glob("seed*")):
        rj = list((sd / model).glob("**/results.json")) if (sd / model).exists() else []
        if not rj:
            continue
        lp = json.loads(rj[0].read_text(encoding="utf-8")).get("learned_params", {}) or {}
        for k in keys:
            if k in lp:
                try:
                    vals.append(float(lp[k])); break
                except (TypeError, ValueError):
                    pass
    return np.array(vals, dtype=float)


def fig_parametros():
    """Parametros fisicos aprendidos (escalares globais) vs referencia, dispersao entre seeds."""
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
        print("[skip] fig_parametros: learned_params nao encontrados")
        return
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
            ticks.append(i); labels.append(lab)
        for rv, rlab in refs:
            ax.axhline(rv, color="#c0392b", lw=0.8, ls="--", zorder=2)
            ax.annotate(rlab, (1.0, rv), xycoords=("axes fraction", "data"),
                        xytext=(-2, 2), textcoords="offset points", fontsize=6,
                        color="#c0392b", va="bottom", ha="right")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlim(-0.5, max(0, len(series) - 1) + 0.5)
    axes[0].set_ylabel("Learned value")
    fig.tight_layout()
    _save(fig, "fig_parametros")


# ---------------------------------------------------------------- Fig.3 -- evento de recessao
TELEM_CANDS = [r"D:\TTD_SCS_LSTM\ablacao_v2\data\dataset_58585000_telem.h5",
               r"D:\TTD_SCS_LSTM\ablacao_skill\data\dataset_58585000_telem.h5"]


def fig_recessao():
    """Janela de recessao/estiagem (a mais seca do teste, log-y): com o ATALHO FECHADO
    (variantes PeOnly) o gerador com estado (TOPMODEL) acompanha melhor a drenagem/baseflow
    que o neural puro e que o gerador de evento (SCS-CN). Elenco: obs / LSTM puro (degrau 4)
    / +SCS-CN PeOnly (degrau 8) / +TOPMODEL PeOnly (degrau 10). Alinha as predicoes as datas
    reconstruindo as janelas (L=240,H=1) e validando obs vs target do npz."""
    import h5py
    import pandas as pd
    L, H = 240, 1
    P_top, target = _pred_mean("LSTM_DUH_Topmodel_PeOnly")
    if P_top is None:
        print("[skip] fig_recessao: sem predictions")
        return
    telem = next((c for c in TELEM_CANDS if Path(c).exists()), None)
    if telem is None:
        print("[skip] fig_recessao: telem nao encontrado")
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
    print(f"[recessao] alinhamento obs vs npz target: max|diff|={diff:.4g} (n={n})")
    if diff > 1.0:
        print("[recessao] AVISO: alinhamento ruim (>1 m3/s); datas podem nao bater")

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
    # janela: periodo de ~60d com menor vazao media (estiagem/recessao)
    daily = obs.resample("1D").mean()
    centro = daily.rolling(60, min_periods=30).mean().idxmin()
    a, b = centro - pd.Timedelta(days=55), centro + pd.Timedelta(days=20)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    o = obs[(obs.index >= a) & (obs.index <= b)]
    ax.plot(o.index, o.values, color="black", lw=1.5, label="Observed", zorder=5)
    # vies nas 30% menores vazoes da janela (ressalva da legenda da Fig. S4)
    lim30 = np.nanpercentile(o.values, 30)
    baixo = o.values <= lim30
    print(f"[recessao] janela {a.date()} a {b.date()} | obs min={np.nanmin(o.values):.1f} "
          f"m3/s | p30={lim30:.1f} m3/s (n={int(baixo.sum())} h)")
    for m, (lab, c2, ls) in models.items():
        if m in sims:
            s = sims[m][(sims[m].index >= a) & (sims[m].index <= b)]
            ax.plot(s.index, s.values, color=c2, lw=1.0, ls=ls, label=lab, zorder=4)
            sv = s.reindex(o.index).values
            vies = 100.0 * (np.nanmean(sv[baixo]) - np.nanmean(o.values[baixo])) / np.nanmean(o.values[baixo])
            print(f"[recessao] {m}: min={np.nanmin(sv):.1f} m3/s | vies nas 30% menores = {vies:+.1f}%")
    _nice_logy(ax, [15, 20, 30, 40, 50, 60, 70])
    ax.set_ylim(15, 75)   # faixa fixa: todos os ticks 15..70 visiveis
    ax.set_ylabel("Discharge (m$^3$ s$^{-1}$)")
    ax.set_xlabel("Date (test period)")
    # rodada 7 (layout, 11/10): legenda acima do eixo, numa linha; dentro dele cobria os picos de ~70 m3/s
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=4, frameon=False)
    ax.grid(axis="both", which="both", color="0.9", lw=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "fig_recessao")


# ---------------------------------------------------------------- Fig. serie completa (melhor x pior)
def _obs_e_datas():
    """obs (N,) e DatetimeIndex (N,) das janelas validas do teste (L=240, H=1), validado contra o npz."""
    import h5py
    import pandas as pd
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


def fig_serie():
    """Serie completa do teste com DATAS: observada vs melhor (ensemble) vs pior (fisica pura)."""
    obs, datas = _obs_e_datas()
    if obs is None:
        print("[skip] fig_serie: telem nao encontrado")
        return
    Pb, tgt = _pred_mean("LSTM_DUH_Topmodel_PeOnly")
    Pw, _ = _pred_mean("Phys_DUH_Topmodel")
    Pn, _ = _pred_mean("LSTM_Lumped_RainOnly")
    if Pb is None or Pw is None or Pn is None:
        print("[skip] fig_serie: sem predictions")
        return
    n = min(len(obs), len(datas), len(Pb), len(Pw), len(Pn))
    # sanidade: a obs reconstruida deve casar com o target do npz
    d = np.nanmax(np.abs(obs[:n] - tgt[:n]))
    print(f"[serie] alinhamento obs vs npz target: max|diff|={d:.4g}")
    nb, nw, nn = _nse(Pb[:n], obs[:n]), _nse(Pw[:n], obs[:n]), _nse(Pn[:n], obs[:n])
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
    # rodada 7 (layout, 11/10): no canto direito a legenda cobria o pico de dezembro
    ax.legend(loc="upper center", fontsize=7.5)
    ax.grid(axis="both", color="0.9", lw=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "fig_serie")


# ------------------------------------------------- Fig. S5 -- geradores puros (sem LSTM)
def fig_geradores_puros():
    """Serie completa do teste, escala linear: observado vs os DOIS geradores puros rodados
    sem a LSTM (Phys_DUH_SCS, ensemble de 10 sementes; Phys_DUH_Topmodel, 2 sementes).
    Sustenta a §5.2 e os degraus 12-13 da Tabela 3 (NSE de ensemble negativo nos dois)."""
    obs, datas = _obs_e_datas()
    if obs is None:
        print("[skip] fig_geradores_puros: telem nao encontrado")
        return
    Ps, tgt = _pred_mean("Phys_DUH_SCS")
    Pt, _ = _pred_mean("Phys_DUH_Topmodel")
    if Ps is None or Pt is None:
        print("[skip] fig_geradores_puros: sem predictions dos geradores puros")
        return
    n = min(len(obs), len(datas), len(Ps), len(Pt), len(tgt))
    d = np.nanmax(np.abs(obs[:n] - tgt[:n]))
    print(f"[geradores] alinhamento obs vs npz target: max|diff|={d:.4g} (n={n})")
    ns, nt = _nse(Ps[:n], obs[:n]), _nse(Pt[:n], obs[:n])
    n_scs = sum(1 for sd in sorted(GRADE.glob("seed*")) if list((sd / "Phys_DUH_SCS").glob("**/predictions.npz")))
    n_top = sum(1 for sd in sorted(GRADE.glob("seed*")) if list((sd / "Phys_DUH_Topmodel").glob("**/predictions.npz")))
    print(f"[geradores] Phys_DUH_SCS: {n_scs} seeds, ensemble NSE = {ns:.3f} | "
          f"Phys_DUH_Topmodel: {n_top} seeds, ensemble NSE = {nt:.3f}")

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.plot(datas[:n], obs[:n], color="black", lw=0.7, label="Observed", zorder=4)
    ax.plot(datas[:n], Ps[:n], color="#e67e22", lw=0.7, zorder=3,
            label=f"DUH-SCS-CN, no LSTM ({n_scs} seeds): NSE = {ns:.2f}")
    ax.plot(datas[:n], Pt[:n], color="#1b9e77", lw=0.7, zorder=2,
            label=f"DUH-TOPMODEL, no LSTM ({n_top} seeds): NSE = {nt:.2f}")
    ax.set_ylabel("Discharge (m$^3$ s$^{-1}$)")
    ax.set_xlabel("Date (test period)")
    # rodada 7 (layout, 11/10): no canto direito a legenda cobria o pico de dezembro
    ax.legend(loc="upper center", fontsize=7.5)
    ax.grid(axis="both", color="0.9", lw=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "fig_geradores_puros")


def main():
    # rodada 7: so as figuras com rotulos de metrica (Figuras 4, 5, 6, S3 e S5);
    # fig_parametros e fig_recessao nao imprimem metrica e nao sao regeradas
    fig1_cadeia()
    fig_robustez()
    fig_fdc()
    fig_serie()
    fig_geradores_puros()


if __name__ == "__main__":
    main()
