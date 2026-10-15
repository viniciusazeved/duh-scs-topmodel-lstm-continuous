#!/usr/bin/env python
"""Supplementary Figure S1: the lumped ABC sweeps in continuous simulation (horizon 1 h).

Mean test NSE (bars) and population standard deviation across the 5 seeds (error bars) in modes A
(rainfall only), C (calendar only) and B (rainfall plus calendar, coupled):

  (a) architectures with the telemetric forcing (``outputs/abc/backbones_telem_gpu1.log``);
  (b) rainfall products with the LSTM (``outputs/abc/products_lstm_gpu1.log``).

The two panels meet at the configuration they share (LSTM with the telemetric forcing), marked
with a star. The dashed line is the mean test NSE of the coupled lumped LSTM of the decisive grid
(rung 3, LSTM_Lumped, 10 seeds). The bars use the three-decimal summaries that ``lumped_lstm.py``
printed in the logs, read by ``tabela_abc.parse``; ``tabela_abc.py`` recomputes Tables S5 and S6
from the per-seed values of the same logs.

Output: ``outputs/figures/fig_abc.png`` (300 dpi; the file the manuscript includes) and
``fig_abc.pdf``. With the environment of the runs (Python 3.13.10, matplotlib 3.10.8, numpy
2.4.1, Pillow 12.1.0) and the Times New Roman font, the PNG is byte-identical to the submitted one.

Origin: ``fig_abc.py`` of the authors' working folder (4 September 2026; SHA-256
85e9da09a521dcc851d125daaa526c5b9914152ba60d382ad066160b058fcaf6), the generator of the submitted
figure. Changes for this repository, none of which alters the figure: paths relative to the
repository, with the output folder created in ``main()``; argparse; type hints; comments and
console messages in English.

Usage, from the repository root:
    python scripts/figures/fig_abc.py [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabela_abc as T  # noqa: E402  (parse, LOGS_ARCH, LOG_PROD, BB_LABEL, PROD_LABEL)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures"

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
COR = {"A": "#e08214", "C": "#8073ac", "B": "#2166ac"}   # rainfall / calendar / coupled
# Mean test NSE of rung 3 (LSTM_Lumped) of the decisive grid over seeds 42 to 51, from the
# results.json files in outputs/grade_sim: unrounded 0.66502, drawn at three decimals.
GRADE_B_LUMPED = 0.665

Row = tuple[str, str, dict[str, tuple[float, float]]]   # (label, key, {mode: (mean, sd)})


def _coletar(data: dict[tuple[str, str], dict[str, tuple[float, float]]], fixar_key: int,
             fixar_val: str, varia_idx: int, label_map: dict[str, str]) -> list[Row]:
    """Rows of one panel, sorted by decreasing mode-B mean; only items that have mode B."""
    rows = []
    for chave, d in data.items():
        if chave[fixar_key] != fixar_val or "B" not in d:
            continue
        var = chave[varia_idx]
        rows.append((label_map.get(var, var), var, d))
    rows.sort(key=lambda r: -r[2]["B"][0])
    return rows


def _painel(ax: Axes, rows: list[Row], titulo: str, vencedor_key: str,
            ref_annot: bool = False) -> None:
    """Grouped bars (modes A, C, B) of one panel, with the grid reference and the star."""
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
        ax.annotate("lumped LSTM baseline", (n - 0.5, GRADE_B_LUMPED), xytext=(0, 2),
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Supplementary Figure S1 (lumped ABC sweeps).")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output folder (default: outputs/figures)")
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    arch = {}
    for lg in T.LOGS_ARCH:
        arch.update(T.parse(lg))
    prod = T.parse(T.LOG_PROD)

    rows_a = _coletar(arch, 0, "telem", 1, T.BB_LABEL)
    rows_b = _coletar(prod, 1, "lstm", 0, T.PROD_LABEL)
    if not rows_a or not rows_b:
        print(f"[warn] partial data: {len(rows_a)} architectures, {len(rows_b)} products "
              "(drawing anyway)")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    _painel(axa, rows_a, "(a) Architecture  (telemetric)", "lstm", ref_annot=True)
    _painel(axb, rows_b, "(b) Rainfall product  (LSTM)", "telem", ref_annot=False)
    axa.set_ylabel("NSE (1 h simulation)")
    handles, labels = axa.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_abc.{ext}")
    plt.close(fig)
    print(f"[OK] fig_abc (.png, .pdf): (a) {len(rows_a)} architectures, (b) {len(rows_b)} products")
    print("Bars as drawn: three-decimal summaries of the logs, mean +/- population sd "
          "(two-decimal tables: tabela_abc.py)")
    for panel, rows in (("(a)", rows_a), ("(b)", rows_b)):
        for lab, _, d in rows:
            cells = "  ".join(f"{m}: {d[m][0]:+.3f} +/- {d[m][1]:.3f}"
                              for m in ("A", "C", "B") if m in d)
            print(f"  {panel} {lab:<20} {cells}")
    print(f"  dashed line (rung 3, LSTM_Lumped, 10 seeds): {GRADE_B_LUMPED:.3f}")


if __name__ == "__main__":
    main()
