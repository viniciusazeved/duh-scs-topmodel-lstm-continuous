#!/usr/bin/env python
"""Figura do experimento ABC (simulacao continua, h=1): dois paineis A/B/C.

  (a) arquitetura x A/B/C  (produto = telemetrico)  -> arquitetura secundaria (topo comprime)
  (b) produto    x A/B/C   (arquitetura = LSTM)      -> telem e o teto, produtos de grade perdem

Os dois paineis se cruzam no campeao compartilhado (LSTM x telemetrico), teto de ambos,
destacado com estrela. Le os MESMOS logs do tabela_abc.py (mesma fonte das Tabelas 3 e 4).
Estilo Elsevier identico ao gerar_figuras_sim.py.

Saida: D:/Artigo_JOH/artigo_simulacao/figuras/fig_abc.{png,pdf}
Rodar: uv run --project D:/TTD_SCS_LSTM python scripts/fig_abc.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabela_abc as T   # parse, LOGS_ARCH, LOG_PROD, BB_LABEL, PROD_LABEL

OUT = Path(r"D:\Artigo_JOH\artigo_simulacao\figuras")
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "font.weight": "normal",
    "axes.labelweight": "normal", "axes.titleweight": "normal",
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.4, "axes.axisbelow": True,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "legend.fontsize": 8, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
COR = {"A": "#e08214", "C": "#8073ac", "B": "#2166ac"}   # rain / calendar / coupled
GRADE_B_LUMPED = 0.665   # LSTM_Lumped da grade (10 seeds), referencia de sanidade


def _coletar(data, fixar_key, fixar_val, varia_idx, label_map):
    """[(label, key, {A,C,B:(mu,sd)})] ordenado por B desc; so itens que ja tem B."""
    rows = []
    for chave, d in data.items():
        if chave[fixar_key] != fixar_val or "B" not in d:
            continue
        var = chave[varia_idx]
        rows.append((label_map.get(var, var), var, d))
    rows.sort(key=lambda r: -r[2]["B"][0])
    return rows


def _painel(ax, rows, titulo, vencedor_key, ref_annot=False):
    n = len(rows)
    x = np.arange(n)
    w = 0.27
    for modo, lab, off in (("A", "Rainfall only", -w), ("C", "Calendar only", 0.0),
                           ("B", "Rainfall + calendar", w)):
        mus = [r[2].get(modo, (np.nan, 0.0))[0] for r in rows]
        sds = [r[2].get(modo, (np.nan, 0.0))[1] for r in rows]
        ax.bar(x + off, mus, w, yerr=sds, label=lab, color=COR[modo], edgecolor="0.3",
               linewidth=0.4, error_kw=dict(lw=0.6, capsize=2, ecolor="0.45"), zorder=3)
    ax.axhline(GRADE_B_LUMPED, color="0.4", lw=0.7, ls=(0, (4, 2)), zorder=2)
    if ref_annot:
        ax.annotate("grade lumped LSTM (B)", (n - 0.5, GRADE_B_LUMPED), xytext=(0, 2),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6.6, color="0.4")
    ax.axhline(0, color="0.5", lw=0.6, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=18, ha="right")
    for i, r in enumerate(rows):
        if r[1] == vencedor_key:
            b, sdb = r[2]["B"]
            ax.plot([i + w], [b + sdb + 0.045], marker="*", ms=11, color="#c0392b",
                    markeredgecolor="#7b1f12", markeredgewidth=0.4, clip_on=False, zorder=7)
            ax.get_xticklabels()[i].set_fontweight("bold")
    ax.set_title(titulo, fontsize=9)


def main():
    arch = {}
    for lg in T.LOGS_ARCH:
        arch.update(T.parse(lg))
    prod = T.parse(T.LOG_PROD)

    rows_a = _coletar(arch, 0, "telem", 1, T.BB_LABEL)
    rows_b = _coletar(prod, 1, "lstm", 0, T.PROD_LABEL)
    if not rows_a or not rows_b:
        print(f"[warn] dados parciais: arch={len(rows_a)} prod={len(rows_b)} (gerando mesmo assim)")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    _painel(axa, rows_a, "(a) Architecture  (telemetric)", "lstm", ref_annot=True)
    _painel(axb, rows_b, "(b) Rainfall product  (LSTM)", "telem", ref_annot=False)
    axa.set_ylabel("NSE (1 h simulation)")
    h, l = axa.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_abc.{ext}")
    plt.close(fig)
    print(f"[OK] fig_abc  (a: {len(rows_a)} arquiteturas, b: {len(rows_b)} produtos)")


if __name__ == "__main__":
    main()
