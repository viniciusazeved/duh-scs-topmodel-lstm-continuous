#!/usr/bin/env python
"""Figure 3 of the manuscript: architecture of the DUH-SCS/TOPMODEL-LSTM in continuous simulation.

Matplotlib drawing in the Elsevier style, laid out in two columns: the differentiable physics on
the left (rainfall, runoff generation by SCS-CN or TOPMODEL, DUH routing, area-weighted sum) and
the sequence model on the right (concatenation, LSTM, decoder, next-step discharge). Potential
evapotranspiration reaches only TOPMODEL, and a note beside the LSTM records that it was selected
in the preparatory sweep of five architectures (Supplementary Section S5).

Coordinates are in inches (1 data unit = 1 in), so the font sizes set here are close to the
printed ones: the manuscript includes the figure at 0.9 linewidth (about 4.3 in) and the tight box
of the drawing is about 4.6 in wide (factor of about 0.93; 9 pt in the drawing print at about
8.4 pt).

Requires the Times New Roman font. Without it matplotlib falls back to DejaVu Serif and the output
is no longer identical to the submitted figure. Mathematical text uses the STIX fonts shipped with
matplotlib, and the PDF embeds TrueType (Type 42) fonts.

Output: ``<stem>.png`` (300 dpi) and ``<stem>.pdf``; the manuscript includes the PDF. With the
environment of the runs (Python 3.13.10, matplotlib 3.10.8, Pillow 12.1.0, fontTools 4.61.1) and
Times New Roman, the PNG is byte-identical to the submitted one. The PDF contains the same objects
as the submitted one; because matplotlib embeds the fonts in the order of their file paths, it is
byte-identical apart from the CreationDate only when matplotlib (whose STIX fonts set the
mathematical text) is installed under a path that sorts before the Times New Roman file, as on the
machine that made the figure (matplotlib in the uv cache of the user profile, the font in the
Windows font folder); otherwise the same fonts appear in another order.

Origin: ``fig_arquitetura_sim_rodada7.py`` (revision round 7, 10 October 2026; SHA-256
f8d7dd6cfb3c013182b1d2be0024049dbb1440d23ab1c5d910513d613a155f0c), the generator of the submitted
figure. Changes for this repository, none of which alters the drawing: default output path
relative to the repository, with its folder created if missing; drawing wrapped in functions, so
that importing the module has no side effects; argparse; type hints; comments in English.

Usage, from the repository root:
    python scripts/figures/fig_arquitetura_sim.py [STEM]

STEM defaults to outputs/figures/fig_arquitetura; a suffix, if any, is replaced by .png and .pdf.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEM = ROOT / "outputs" / "figures" / "fig_arquitetura"

STYLE = {"font.family": "serif",
         "font.serif": ["Times New Roman", "DejaVu Serif"],
         "font.size": 9, "mathtext.fontset": "stix",
         "pdf.fonttype": 42}

Color = str | tuple[float, ...]

PHYS = (38/255, 84/255, 124/255)
PHYSF = (232/255, 240/255, 247/255)
NEU = (176/255, 96/255, 30/255)
NEUF = (251/255, 242/255, 231/255)
GRAY = (70/255, 70/255, 70/255)

TS, SS, HS = 10, 9, 9.5        # box title, box subtitle and block header (pt)

# ---------- geometry (inches) ----------
BW, BH = 1.80, 0.46            # standard box (two lines)
G = 0.15                       # gap between boxes (arrow length)
PADB = 0.12                    # inner margin of the blocks
HDR = 0.30                     # header band inside a block
FR_H = 0.80                    # frame "runoff generation"
IN_W, IN_H = 0.76, 0.42        # inner boxes SCS-CN / TOPMODEL
BLK_W = BW + 0.30              # block width
GAPB = 0.20                    # gap between the two blocks

XP_L = 0.05                    # physics block
XP_R = XP_L + BLK_W
CXP = (XP_L + XP_R) / 2
XS_L = XP_R + GAPB             # sequence block
XS_R = XS_L + BLK_W
CXS = (XS_L + XS_R) / 2
CX_SCS, CX_TOP = CXP - 0.47, CXP + 0.47                           # inner boxes
CX_PBOX, PBOX_W = 1.85, 1.50           # input P (its stem descends right of the block header)
SIDE_W = 0.96                          # side inputs (PET, calendar)
CX_PET = XS_L + 0.03 + SIDE_W/2
CX_CAL = CX_PET + SIDE_W + 0.12
X_RAIL = XS_R + 0.13                   # rail P -> concatenate

y_qhat = 0.05 + PADB + BH/2
y_dec = y_qhat + BH + G
y_lstm = y_dec + BH + G
y_cat = y_lstm + BH + G
seq_bot, seq_top = 0.05, y_cat + BH/2 + HDR
y_qr = y_cat                                # Q_routed level with concatenate
y_area = y_qr + BH + G
y_duh = y_area + BH + G
fr_bot = y_duh + BH/2 + G
fr_top = fr_bot + FR_H
y_in = fr_top - 0.07 - IN_H/2               # inner boxes (and side inputs)
y_frt = fr_bot + 0.12                       # frame title
y_bar = fr_top + 0.22                       # bar of the P fork
y_hdr = y_bar + 0.22                        # header of the physics block
phys_bot, phys_top = y_qr - BH/2 - PADB, y_hdr + 0.16
y_p = phys_top + 0.18 + BH/2                # input box P

# Framing: the axes span the whole figure, 1 unit = 1 in, margin of 0.05 in.
X0, X1 = XP_L - 0.05, X_RAIL + 0.05
Y0, Y1 = 0.0, y_p + BH/2 + 0.05


def caixa(ax: Axes, x: float, y: float, w: float, h: float, titulo: str, sub: str, ec: Color,
          fc: Color = "white", tcol: Color | None = None, lw: float = 1.1, ts: float = TS,
          ss: float = SS, rs: float = 0.06) -> None:
    """Rounded box centred at (x, y) with a title and an optional italic subtitle."""
    tcol = tcol or ec
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle=f"round,pad=0,rounding_size={rs}", ec=ec, fc=fc, lw=lw))
    if sub:
        ax.text(x, y + 0.085, titulo, ha="center", va="center", color=tcol, fontsize=ts)
        ax.text(x, y - 0.105, sub, ha="center", va="center", color=tcol, fontsize=ss,
                style="italic")
    else:
        ax.text(x, y, titulo, ha="center", va="center", color=tcol, fontsize=ts)


def seta(ax: Axes, x0: float, y0: float, x1: float, y1: float, col: Color, lw: float = 1.2) -> None:
    """Arrow from (x0, y0) to (x1, y1)."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=9, color=col, lw=lw, shrinkA=0, shrinkB=0))


def bloco(ax: Axes, x0: float, y0: float, x1: float, y1: float, col: tuple[float, ...],
          titulo: str) -> None:
    """Translucent block with a bold header."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0,rounding_size=0.09",
                 ec=col, fc=(col[0], col[1], col[2], 0.05), lw=1.2))
    ax.text(x0 + 0.10, y1 - 0.16, titulo, color=col, fontsize=HS, fontweight="bold", va="center")


def build_figure() -> Figure:
    """Draw the diagram; the order of the calls is the drawing order of the submitted figure."""
    fig = plt.figure(figsize=(X1 - X0, Y1 - Y0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.axis("off")

    # ---------- inputs ----------
    caixa(ax, CX_PBOX, y_p, PBOX_W, BH, "precipitation $P$", "245 sub-catchments, hourly", GRAY)
    # Round 7: with the hyphen, "per sub-catchment" at 9 pt came within 0.7 pt of the edge of the
    # box (0.96 in); the subtitle of this box drops to 8.6 pt, the largest size that restores its
    # original clearance (2.2 pt on each side). The title "PET" and the other boxes are unchanged.
    caixa(ax, CX_PET, y_in, SIDE_W, BH, "PET", "per sub-catchment", GRAY, ss=8.6)
    caixa(ax, CX_CAL, y_in, SIDE_W, BH, "calendar", "hour, month", GRAY)

    # ---------- physics block ----------
    bloco(ax, XP_L, phys_bot, XP_R, phys_top, PHYS, "Differentiable physics")
    # Frame of the runoff generation, with the two options side by side.
    ax.add_patch(FancyBboxPatch((CXP - BW/2, fr_bot), BW, FR_H,
                 boxstyle="round,pad=0,rounding_size=0.06", ec=PHYS, fc="white", lw=1.1))
    ax.text(CXP, y_frt, "runoff generation (one of the two)", ha="center", va="center",
            color=PHYS, fontsize=SS, style="italic")
    caixa(ax, CX_SCS, y_in, IN_W, IN_H, "SCS-CN", "$P$ only", PHYS,
          fc=PHYSF, ts=9, ss=SS, lw=0.9, rs=0.04)
    caixa(ax, CX_TOP, y_in, IN_W, IN_H, "TOPMODEL", "$P$ and PET", PHYS,
          fc=PHYSF, ts=9, ss=SS, lw=0.9, rs=0.04)
    ax.text(CXP, y_in, "or", ha="center", va="center", color=PHYS, fontsize=SS, style="italic")
    caixa(ax, CXP, y_duh, BW, BH, "DUH routing ($T_c$, $\\sigma$)", "Gaussian unit hydrograph",
          PHYS)
    caixa(ax, CXP, y_area, BW, BH, "area-weighted sum", "over the 245 sub-catchments", PHYS)
    caixa(ax, CXP, y_qr, BW, BH, "$Q_{\\mathrm{routed}}$", "routed flow, single series", PHYS,
          fc=PHYSF)

    # ---------- sequence block ----------
    bloco(ax, XS_L, seq_bot, XS_R, seq_top, NEU, "Sequence model")
    caixa(ax, CXS, y_cat, BW, BH, "concatenate",
          "$P$ or $P_e$ $\\cdot$ $Q_{\\mathrm{routed}}$ $\\cdot$ calendar", NEU)
    caixa(ax, CXS, y_lstm, BW, BH, "LSTM", "2 layers $\\cdot$ hidden size 64", NEU)
    caixa(ax, CXS, y_dec, BW, BH, "decoder", "single step, not autoregressive", NEU)
    caixa(ax, CXS, y_qhat, BW, BH, "$\\hat{Q}_{t+1}$ (next-step discharge)", "window slid by 1 h",
          NEU, fc=NEUF, lw=1.8)

    # ---------- arrows: P -> generation (the fork reaches both options) ----------
    ax.plot([CX_PBOX, CX_PBOX, CX_SCS], [y_p - BH/2, y_bar, y_bar], color=PHYS, lw=1.2,
            solid_capstyle="round")
    seta(ax, CX_SCS, y_bar, CX_SCS, fr_top, PHYS)
    seta(ax, CX_TOP, y_bar, CX_TOP, fr_top, PHYS)
    # PET -> TOPMODEL only
    seta(ax, CX_PET - SIDE_W/2, y_in, CXP + BW/2, y_in, GRAY)
    # physics chain
    seta(ax, CXP, fr_bot, CXP, y_duh + BH/2, PHYS)
    for yt, yb in [(y_duh, y_area), (y_area, y_qr)]:
        seta(ax, CXP, yt - BH/2, CXP, yb + BH/2, PHYS)
    # Q_routed -> concatenate
    seta(ax, CXP + BW/2, y_qr, CXS - BW/2, y_cat, PHYS)
    # sequence chain
    for yt, yb in [(y_cat, y_lstm), (y_lstm, y_dec), (y_dec, y_qhat)]:
        seta(ax, CXS, yt - BH/2, CXS, yb + BH/2, NEU)
    # Co-inputs of concatenate: calendar (from above) and raw P (rail on the right).
    seta(ax, CX_CAL, y_in - BH/2, CX_CAL, y_cat + BH/2, GRAY)
    ax.plot([CX_PBOX + PBOX_W/2, X_RAIL, X_RAIL], [y_p, y_p, y_cat], color=GRAY, lw=1.2)
    seta(ax, X_RAIL, y_cat, CXS + BW/2, y_cat, GRAY)

    # ---------- note on the architecture sweep ----------
    note = ("LSTM selected in a preparatory\n"
            "sweep of five architectures:\n"
            "LSTM, GRU, Transformer, MLP\n"
            "and DLinear (lumped ABC\n"
            "experiment, SI Section S5)")
    nx0, nx1, ny0, ny1 = XP_L + 0.05, XP_R - 0.05, phys_bot - 0.97, phys_bot - 0.08
    ax.add_patch(FancyBboxPatch((nx0, ny0), nx1 - nx0, ny1 - ny0,
                 boxstyle="round,pad=0,rounding_size=0.06", ec=GRAY, fc="white", lw=0.8,
                 ls=(0, (3, 2))))
    ax.text((nx0 + nx1)/2, (ny0 + ny1)/2, note, ha="center", va="center", color=GRAY,
            fontsize=SS, style="italic", linespacing=1.25)
    # dashed link from the note to the LSTM box
    ax.plot([nx1, CXS - BW/2], [y_lstm, y_lstm], color=GRAY, lw=0.8, ls=(0, (3, 2)))
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Figure 3: architecture diagram (PNG and PDF).")
    parser.add_argument("stem", nargs="?", type=Path, default=DEFAULT_STEM,
                        help="output path without suffix "
                             "(default: outputs/figures/fig_arquitetura)")
    args = parser.parse_args(argv)
    out = args.stem
    out.parent.mkdir(parents=True, exist_ok=True)

    matplotlib.use("Agg")
    plt.rcParams.update(STYLE)
    fig = build_figure()
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    try:
        shown = out.resolve().relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"[OK] {shown.with_suffix('.png')} and .pdf ({bb.width:.2f} x {bb.height:.2f} in)")


if __name__ == "__main__":
    main()
