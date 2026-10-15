#!/usr/bin/env python
# RODADA 7 (2026-10-10), decisao D4 do Vinicius: "sub-catchment(s)" com hifen.
# Copia de D:/Artigo_JOH/scripts/fig_arquitetura_sim.py (sha256 89c93b54628ecf03..., manifesto
# externo do congelamento de 29/09/2026); o original NAO foi alterado. Alteracoes (marcadas
# "rodada 7"): tres subtitulos de caixa ("245 sub-catchments, hourly", "per sub-catchment",
# "over the 245 sub-catchments") e saida padrao no staging da rodada 7 (.png e .pdf; o LaTeX
# usa o .pdf); subtitulo da caixa PET em 8,6 pt para caber (ver la). Nada de dado, cor ou desenho.
# Rodar: uv run --no-project --python 3.13.10 --with matplotlib==3.10.8 --with numpy==2.4.1 \
#          python fig_arquitetura_sim_rodada7.py
"""Diagrama de arquitetura (matplotlib, estilo Elsevier) do DUH-SCS/TOPMODEL-LSTM
para SIMULACAO CONTINUA (Figura 3 do Paper 1).

Layout em duas colunas (fisica a esquerda, modelo sequencial a direita), com as
coordenadas em POLEGADAS (1 unidade = 1 in), de modo que o tamanho de fonte
definido aqui e' praticamente o tamanho impresso: em elsarticle[review] a figura
sai com 0.9\\linewidth ~ 4.3 in e a caixa apertada deste desenho tem ~4.6 in
(fator ~0.93; 9 pt no desenho = ~8.4 pt no papel).

Revisao de 15/09/2026, a partir da leitura do Paulo:
  - P12: a PET aparece como entrada propria, chegando SO ao TOPMODEL; o SCS-CN
    recebe apenas P (caixas internas "P only" / "P and PET", e o garfo de P
    chega as duas opcoes).
  - P14: nota junto ao bloco LSTM dizendo que ele foi escolhido numa varredura
    preparatoria de cinco arquiteturas (experimento ABC concentrado, SI S5).
  - Fonte pequena: o layout vertical antigo (10 linhas) imprimia titulos a
    ~6 pt e subtitulos a ~5 pt; agora tudo >= 9 pt no desenho. A duas colunas
    e' o que permite fonte legivel sem estourar a altura da pagina.

Saida: artigo_simulacao/figuras/fig_arquitetura.{png,pdf}
"""
from __future__ import annotations
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "mathtext.fontset": "stix",
                     "pdf.fonttype": 42})

PHYS = (38/255, 84/255, 124/255)
PHYSF = (232/255, 240/255, 247/255)
NEU = (176/255, 96/255, 30/255)
NEUF = (251/255, 242/255, 231/255)
GRAY = (70/255, 70/255, 70/255)

TS, SS, HS = 10, 9, 9.5        # titulo da caixa, subtitulo, cabecalho do bloco

# ---------- geometria (polegadas) ----------
BW, BH = 1.80, 0.46            # caixa padrao (2 linhas)
G = 0.15                       # vao entre caixas (comprimento das setas)
PADB = 0.12                    # folga interna dos blocos
HDR = 0.30                     # faixa do cabecalho dentro do bloco
FR_H = 0.80                    # moldura "runoff generation"
IN_W, IN_H = 0.76, 0.42        # caixas internas SCS-CN / TOPMODEL
BLK_W = BW + 0.30              # largura dos blocos
GAPB = 0.20                    # vao entre os dois blocos

XP_L = 0.05; XP_R = XP_L + BLK_W; CXP = (XP_L + XP_R) / 2         # bloco fisico
XS_L = XP_R + GAPB; XS_R = XS_L + BLK_W; CXS = (XS_L + XS_R) / 2  # bloco sequencial
CX_SCS, CX_TOP = CXP - 0.47, CXP + 0.47                           # caixas internas
CX_PBOX, PBOX_W = 1.85, 1.50           # entrada P (haste desce a direita do cabecalho)
SIDE_W = 0.96                          # entradas laterais (PET, calendario)
CX_PET = XS_L + 0.03 + SIDE_W/2; CX_CAL = CX_PET + SIDE_W + 0.12
X_RAIL = XS_R + 0.13                   # trilho P -> concatenate

y_qhat = 0.05 + PADB + BH/2
y_dec = y_qhat + BH + G
y_lstm = y_dec + BH + G
y_cat = y_lstm + BH + G
seq_bot, seq_top = 0.05, y_cat + BH/2 + HDR
y_qr = y_cat                                # Q_routed alinhado com concatenate
y_area = y_qr + BH + G
y_duh = y_area + BH + G
fr_bot = y_duh + BH/2 + G; fr_top = fr_bot + FR_H
y_in = fr_top - 0.07 - IN_H/2               # caixas internas (e entradas laterais)
y_frt = fr_bot + 0.12                       # titulo da moldura
y_bar = fr_top + 0.22                       # barra do garfo de P
y_hdr = y_bar + 0.22                        # cabecalho do bloco fisico
phys_bot, phys_top = y_qr - BH/2 - PADB, y_hdr + 0.16
y_p = phys_top + 0.18 + BH/2                # caixa de entrada P

# enquadramento: eixo ocupa a figura inteira, 1 unidade = 1 in, margem de 0.05 in
X0, X1 = XP_L - 0.05, X_RAIL + 0.05
Y0, Y1 = 0.0, y_p + BH/2 + 0.05
fig = plt.figure(figsize=(X1 - X0, Y1 - Y0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.axis("off")


def caixa(x, y, w, h, titulo, sub, ec, fc="white", tcol=None, lw=1.1, ts=TS, ss=SS, rs=0.06):
    tcol = tcol or ec
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle=f"round,pad=0,rounding_size={rs}", ec=ec, fc=fc, lw=lw))
    if sub:
        ax.text(x, y + 0.085, titulo, ha="center", va="center", color=tcol, fontsize=ts)
        ax.text(x, y - 0.105, sub, ha="center", va="center", color=tcol, fontsize=ss, style="italic")
    else:
        ax.text(x, y, titulo, ha="center", va="center", color=tcol, fontsize=ts)


def seta(x0, y0, x1, y1, col, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=9, color=col, lw=lw, shrinkA=0, shrinkB=0))


def bloco(x0, y0, x1, y1, col, titulo):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0,rounding_size=0.09",
                 ec=col, fc=(col[0], col[1], col[2], 0.05), lw=1.2))
    ax.text(x0 + 0.10, y1 - 0.16, titulo, color=col, fontsize=HS, fontweight="bold", va="center")


# ---------- entradas ----------
caixa(CX_PBOX, y_p, PBOX_W, BH, "precipitation $P$", "245 sub-catchments, hourly", GRAY)
# rodada 7: com o hifen, "per sub-catchment" a 9 pt ficava a 0,7 pt da borda da caixa
# (0,96 pol); o subtitulo DESTA caixa cai para 8,6 pt, o minimo que devolve a folga que ela
# tinha no original (2,2 pt de cada lado). Titulo "PET" e demais caixas inalterados.
caixa(CX_PET, y_in, SIDE_W, BH, "PET", "per sub-catchment", GRAY, ss=8.6)
caixa(CX_CAL, y_in, SIDE_W, BH, "calendar", "hour, month", GRAY)

# ---------- bloco fisico ----------
bloco(XP_L, phys_bot, XP_R, phys_top, PHYS, "Differentiable physics")
# moldura da geracao de escoamento, com as duas opcoes lado a lado (P12)
ax.add_patch(FancyBboxPatch((CXP - BW/2, fr_bot), BW, FR_H,
             boxstyle="round,pad=0,rounding_size=0.06", ec=PHYS, fc="white", lw=1.1))
ax.text(CXP, y_frt, "runoff generation (one of the two)", ha="center", va="center",
        color=PHYS, fontsize=SS, style="italic")
caixa(CX_SCS, y_in, IN_W, IN_H, "SCS-CN", "$P$ only", PHYS, fc=PHYSF, ts=9, ss=SS, lw=0.9, rs=0.04)
caixa(CX_TOP, y_in, IN_W, IN_H, "TOPMODEL", "$P$ and PET", PHYS, fc=PHYSF, ts=9, ss=SS, lw=0.9, rs=0.04)
ax.text(CXP, y_in, "or", ha="center", va="center", color=PHYS, fontsize=SS, style="italic")
caixa(CXP, y_duh, BW, BH, "DUH routing ($T_c$, $\\sigma$)", "Gaussian unit hydrograph", PHYS)
caixa(CXP, y_area, BW, BH, "area-weighted sum", "over the 245 sub-catchments", PHYS)
caixa(CXP, y_qr, BW, BH, "$Q_{\\mathrm{routed}}$", "routed flow, single series", PHYS, fc=PHYSF)

# ---------- bloco sequencial ----------
bloco(XS_L, seq_bot, XS_R, seq_top, NEU, "Sequence model")
caixa(CXS, y_cat, BW, BH, "concatenate", "$P$ or $P_e$ $\\cdot$ $Q_{\\mathrm{routed}}$ $\\cdot$ calendar", NEU)
caixa(CXS, y_lstm, BW, BH, "LSTM", "2 layers $\\cdot$ hidden size 64", NEU)
caixa(CXS, y_dec, BW, BH, "decoder", "single step, not autoregressive", NEU)
caixa(CXS, y_qhat, BW, BH, "$\\hat{Q}_{t+1}$ (next-step discharge)", "window slid by 1 h", NEU, fc=NEUF, lw=1.8)

# ---------- setas: P -> geracao (garfo: chega as duas opcoes) ----------
ax.plot([CX_PBOX, CX_PBOX, CX_SCS], [y_p - BH/2, y_bar, y_bar], color=PHYS, lw=1.2, solid_capstyle="round")
seta(CX_SCS, y_bar, CX_SCS, fr_top, PHYS)
seta(CX_TOP, y_bar, CX_TOP, fr_top, PHYS)
# PET -> so TOPMODEL (P12)
seta(CX_PET - SIDE_W/2, y_in, CXP + BW/2, y_in, GRAY)
# cadeia fisica
seta(CXP, fr_bot, CXP, y_duh + BH/2, PHYS)
for yt, yb in [(y_duh, y_area), (y_area, y_qr)]:
    seta(CXP, yt - BH/2, CXP, yb + BH/2, PHYS)
# Q_routed -> concatenate
seta(CXP + BW/2, y_qr, CXS - BW/2, y_cat, PHYS)
# cadeia sequencial
for yt, yb in [(y_cat, y_lstm), (y_lstm, y_dec), (y_dec, y_qhat)]:
    seta(CXS, yt - BH/2, CXS, yb + BH/2, NEU)
# co-entradas do concatenate: calendario (de cima) e P bruto (trilho a direita)
seta(CX_CAL, y_in - BH/2, CX_CAL, y_cat + BH/2, GRAY)
ax.plot([CX_PBOX + PBOX_W/2, X_RAIL, X_RAIL], [y_p, y_p, y_cat], color=GRAY, lw=1.2)
seta(X_RAIL, y_cat, CXS + BW/2, y_cat, GRAY)

# ---------- nota da varredura de arquiteturas (P14) ----------
NOTE = ("LSTM selected in a preparatory\n"
        "sweep of five architectures:\n"
        "LSTM, GRU, Transformer, MLP\n"
        "and DLinear (lumped ABC\n"
        "experiment, SI Section S5)")
nx0, nx1, ny0, ny1 = XP_L + 0.05, XP_R - 0.05, phys_bot - 0.97, phys_bot - 0.08
ax.add_patch(FancyBboxPatch((nx0, ny0), nx1 - nx0, ny1 - ny0,
             boxstyle="round,pad=0,rounding_size=0.06", ec=GRAY, fc="white", lw=0.8, ls=(0, (3, 2))))
ax.text((nx0 + nx1)/2, (ny0 + ny1)/2, NOTE, ha="center", va="center", color=GRAY,
        fontsize=SS, style="italic", linespacing=1.25)
ax.plot([nx1, CXS - BW/2], [y_lstm, y_lstm], color=GRAY, lw=0.8, ls=(0, (3, 2)))   # liga a nota ao LSTM

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\Artigo_JOH\artigo_simulacao\revisao_paulo\rodada7_2026-10-09\staging\figuras\fig_arquitetura")  # rodada 7
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
bb = fig.get_tightbbox(fig.canvas.get_renderer())
print(f"OK -> {out}.png/.pdf  ({bb.width:.2f} x {bb.height:.2f} in)")
