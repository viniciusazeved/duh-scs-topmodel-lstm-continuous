#!/usr/bin/env python
"""Diagrama de arquitetura (matplotlib, estilo Elsevier) do DUH-SCS/TOPMODEL-LSTM
para SIMULACAO CONTINUA. Mesma estrutura do diagrama do forecasting; muda a cabeca
de saida (passo unico deslizante, nao bloco de 24 h).

Saida: artigo_simulacao/figuras/fig_arquitetura.{png,pdf}
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "mathtext.fontset": "dejavuserif"})

PHYS = (38/255, 84/255, 124/255)
PHYSF = (232/255, 240/255, 247/255)
NEU = (176/255, 96/255, 30/255)
NEUF = (251/255, 242/255, 231/255)
GRAY = (70/255, 70/255, 70/255)

fig, ax = plt.subplots(figsize=(6.9, 9.0))
ax.set_xlim(0, 10.4); ax.set_ylim(0.5, 14.0); ax.axis("off")

BW, BH = 4.7, 0.86  # largura/altura padrao das caixas internas


def caixa(x, y, w, h, titulo, sub, ec, fc="white", tcol=None, lw=1.3, ts=9, ss=7):
    tcol = tcol or ec
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10", ec=ec, fc=fc, lw=lw))
    if sub:
        ax.text(x, y + 0.15, titulo, ha="center", va="center", color=tcol, fontsize=ts)
        ax.text(x, y - 0.19, sub, ha="center", va="center", color=tcol, fontsize=ss, style="italic")
    else:
        ax.text(x, y, titulo, ha="center", va="center", color=tcol, fontsize=ts)


def seta(x0, y0, x1, y1, col, lw=1.7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=12, color=col, lw=lw, shrinkA=0, shrinkB=0))


# ---------- entradas ----------
caixa(3.2, 13.2, 3.7, 0.9, "raw precipitation $P$", "245 sub-catchments, hourly", GRAY)
caixa(7.5, 13.2, 3.0, 0.9, "time features", "hour, month", GRAY)

# ---------- bloco fisico ----------
ax.add_patch(FancyBboxPatch((1.3, 6.85), 5.9, 4.7, boxstyle="round,pad=0.02,rounding_size=0.15",
             ec=PHYS, fc=(PHYS[0], PHYS[1], PHYS[2], 0.04), lw=1.4))
ax.text(1.55, 11.82, "Differentiable physics", color=PHYS, fontsize=9.5, fontweight="bold", va="center")
ax.text(5.35, 11.82, "(per sub-catchment, $\\times$245)", color=PHYS, fontsize=7.5, style="italic", va="center")
caixa(4.25, 11.0, BW, BH, "runoff generation", "SCS-CN $or$ TOPMODEL (both differentiable)", PHYS)
caixa(4.25, 9.9, BW, BH, "DUH routing", "Gaussian unit hydrograph ($T_c,\\,\\sigma$)", PHYS)
caixa(4.25, 8.8, BW, BH, "area-weighted sum", "over the 245 sub-catchments", PHYS)
caixa(4.25, 7.7, BW, BH, "$Q_{\\mathrm{routed}}$", "routed flow (single series)", PHYS, fc=PHYSF)

# ---------- bloco sequencial ----------
ax.add_patch(FancyBboxPatch((1.3, 0.95), 5.9, 4.7, boxstyle="round,pad=0.02,rounding_size=0.15",
             ec=NEU, fc=(NEU[0], NEU[1], NEU[2], 0.04), lw=1.4))
ax.text(1.55, 5.92, "Sequence model", color=NEU, fontsize=9.5, fontweight="bold", va="center")
caixa(4.25, 5.0, BW, BH, "concatenate", "$P$ $or$ $P_e$ (245) $\\cdot$ $Q_{\\mathrm{routed}}$ $\\cdot$ time", NEU)
caixa(4.25, 3.9, BW, BH, "LSTM", "2 layers $\\cdot$ hidden 64 $\\cdot$ sequence-to-one", NEU)
caixa(4.25, 2.8, BW, BH, "decoder", "single next step (not autoregressive)", NEU)
caixa(4.25, 1.7, BW, BH, "$\\hat{Q}_{t+1}$", "next-step estimate $\\cdot$ window slid 1 h", NEU, fc=NEUF, lw=2.2)

# ---------- setas internas ----------
seta(3.2, 12.75, 4.0, 11.43, PHYS)               # P -> runoff
for yt, yb in [(11.0, 9.9), (9.9, 8.8), (8.8, 7.7)]:
    seta(4.25, yt - BH/2, 4.25, yb + BH/2, PHYS)
seta(4.25, 7.7 - BH/2, 4.25, 5.0 + BH/2, PHYS)   # Q_routed -> concatenate
ax.text(4.55, 6.35, "$Q_{\\mathrm{routed}}$", color=PHYS, fontsize=7.5, va="center")
for yt, yb in [(5.0, 3.9), (3.9, 2.8), (2.8, 1.7)]:
    seta(4.25, yt - BH/2, 4.25, yb + BH/2, NEU)

# ---------- co-entradas (P bruto e tempo) -> concatenate ----------
ax.plot([1.35, 0.7, 0.7], [13.2, 13.2, 5.0], color=GRAY, lw=1.4)
seta(0.7, 5.0, 4.25 - BW/2, 5.0, GRAY)
ax.plot([9.0, 9.7, 9.7], [13.2, 13.2, 5.0], color=GRAY, lw=1.4)
seta(9.7, 5.0, 4.25 + BW/2, 5.0, GRAY)

out = Path(r"D:\Artigo_JOH\artigo_simulacao\figuras\fig_arquitetura")
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"OK -> {out}.png/.pdf")
