"""Figura S2: campo espacial de chuva por produto (5 mapas coropleticos das 245 ottobacias)
na hora de pico do evento mais forte + hietograma medio de bacia do dia. Mostra a diferenca
de resolucao efetiva (telem ~sub-pixel; grades borram) e a subestimacao do pico pelas grades.

Saida: D:\\Artigo_JOH\\figuras\\fig_produtos_evento.{png,pdf}
Uso:   D:\\TTD_SCS_LSTM\\.venv\\Scripts\\python.exe D:\\Artigo_JOH\\scripts\\fig_produtos_evento.py
       (precisa de geopandas, h5py, matplotlib >= 3.6)

Layout revisto em 15/09/2026 (pedido P10 do Paulo: "nao da pra ver muita coisa, principalmente
no grafico"). A bacia e muito larga (h/w ~ 0,41), entao a grade 2x3 antiga deixava buracos
verticais, mapas pequenos e hietograma com legenda de 5 pt. Agora: 3 linhas x 2 colunas com
posicoes calculadas em polegadas, hietograma ocupando a altura inteira da ultima linha,
colorbar sob o mapa do MSWEP e fontes >= 9 pt na figura. O SI inclui a figura com
width=\\linewidth = 16 cm (A4, margem 2,5 cm), o que da escala ~0,94 no papel (tick de 9 pt
vira ~8,5 pt impresso).
"""
from pathlib import Path
import h5py, numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Tamanhos pensados para o papel (PNG ~6,7 in de largura, com o pad, impresso em 16 cm => escala
# ~0,94; 9 pt aqui vira ~8,5 pt impresso, acima do minimo de 8 pt). P10, 15/09/2026.
FS_TICK, FS_LABEL, FS_TITLE = 9.0, 9.5, 10.0
mpl.rcParams.update({"font.size": FS_TICK, "font.family": "DejaVu Sans"})
DATA = Path(r"D:\TTD_SCS_LSTM\big_ablacao\data")
GPKG = Path(r"D:\TTD_SCS_LSTM\data\processed\ottobacias_tc.gpkg")
OUT = Path(r"D:\Artigo_JOH\figuras")
OUT.mkdir(parents=True, exist_ok=True)
PRODS = [("telem", "Telemetric (IDW)"), ("merge", "MERGE"),
         ("era5land", "ERA5-Land"), ("imerg", "IMERG"), ("mswep", "MSWEP")]
# Paleta Okabe-Ito (segura para daltonismo) + marcador distinto por produto no hietograma.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D", "v"]


def load(prod):
    with h5py.File(DATA / f"dataset_58585000_{prod}.h5", "r") as f:
        P = np.concatenate([f[f"{s}/precipitation"][:] for s in ("train", "val", "test")])  # (T,245)
        ts = np.concatenate([f[f"{s}/timestamps"][:] for s in ("train", "val", "test")]).astype("int64")
        area = f["ottobacia/area_km2"][:].astype(np.float64)
    return P, ts, area


# 1. carrega tudo; acha o dia de pico pela chuva media (ponderada por area) do telem
prods_data = {p: load(p) for p, _ in PRODS}
P0, ts0, area = prods_data["telem"]
w = area / area.sum()
tindex = pd.to_datetime(ts0, unit="s", utc=True)
dia = pd.Series(P0 @ w, index=tindex).resample("D").sum()
dpico = dia.idxmax()
print(f"dia de pico (chuva media telem): {dpico:%Y-%m-%d}  ({dia.max():.1f} mm/dia)")
mask_dia = (tindex >= dpico) & (tindex < dpico + pd.Timedelta(days=1))

# 2. campo na HORA DE PICO (teste de resolucao: blocos das grades saltam) + hietograma
Pmd = {}; hieto = {}
for p, _ in PRODS:
    P, ts, _ = prods_data[p]
    ti = pd.to_datetime(ts, unit="s", utc=True)
    md = (ti >= dpico) & (ti < dpico + pd.Timedelta(days=1))
    Pmd[p] = P[md]                           # (24,245) mm/h no dia
    hieto[p] = P[md] @ w                     # (24,) mm/h media de bacia
hpico = int(np.argmax(hieto["telem"]))       # hora de pico (chuva media telem)
campo = {p: Pmd[p][hpico] for p, _ in PRODS} # (245,) mm/h no instante de pico
vmax = float(np.nanpercentile(np.concatenate([campo[p] for p, _ in PRODS]), 96))  # capa linear: telem satura, mas as grades aparecem (blocos)
print(f"hora de pico: {hpico:02d}:00 UTC")

# 3. geometria (lat/lon) + exutorio (a ottobacia de MENOR Tc escoa mais rapido -> e a de saida)
g = gpd.read_file(GPKG).to_crs(4326)
assert len(g) == 245
with h5py.File(DATA / "dataset_58585000_telem.h5", "r") as f:
    tc_base = f["ottobacia/tc_base_h"][:]
outlet = g.geometry.centroid.iloc[int(np.argmin(tc_base))]   # exutorio = Manuel Duarte (58585000)

# 4. layout em polegadas: 3 linhas x 2 colunas (P10, 15/09/2026)
#    linhas 1-2: 4 mapas; linha 3: mapa MSWEP + colorbar (esq.) e hietograma alto (dir.)
b = g.total_bounds
asp = 1.0 / np.cos(np.radians((b[1] + b[3]) / 2))            # mesmo fator que o geopandas usa em lat/lon
xlim = (b[0] - 0.02 * (b[2] - b[0]), b[2] + 0.02 * (b[2] - b[0]))
ylim = (b[1] - 0.04 * (b[3] - b[1]), b[3] + 0.04 * (b[3] - b[1]))
MAP_W = 2.7                                                  # largura de cada mapa (in)
MAP_H = MAP_W * (ylim[1] - ylim[0]) * asp / (xlim[1] - xlim[0])  # caixa casa exatamente com o aspecto
X_L, GAP = 0.55, 0.55                                        # margem p/ rotulos de latitude; vao entre colunas (cabe o eixo y do hietograma)
X_R = X_L + MAP_W + GAP
TOP, T_ROW2, T_ROW3, CB_DROP, CB_H, BOTTOM = 0.30, 0.32, 0.55, 0.58, 0.13, 0.48
H = TOP + 3 * MAP_H + T_ROW2 + T_ROW3 + CB_DROP + BOTTOM
W = X_R + MAP_W + 0.10
fig = plt.figure(figsize=(W, H))


def ax_in(x0, y0, w_, h_):
    """Cria eixo por posicao em polegadas (origem no canto inferior esquerdo)."""
    return fig.add_axes([x0 / W, y0 / H, w_ / W, h_ / H])


y1 = H - TOP - MAP_H
y2 = y1 - T_ROW2 - MAP_H
y3 = y2 - T_ROW3 - MAP_H
y_cb = y3 - CB_DROP
slots = [(X_L, y1), (X_R, y1), (X_L, y2), (X_R, y2), (X_L, y3)]

cmap = "YlGnBu"
norm = mpl.colors.Normalize(vmin=0, vmax=vmax)   # linear: mais intuitiva e mostra a real subestimacao do pico pelas grades
fmt_lon = FuncFormatter(lambda x, _: f"{abs(x):.1f}\u00b0W")
fmt_lat = FuncFormatter(lambda y, _: f"{abs(y):.1f}\u00b0S")
for k, (p, nome) in enumerate(PRODS):
    ax = ax_in(slots[k][0], slots[k][1], MAP_W, MAP_H)
    g2 = g.copy(); g2["mm"] = campo[p]
    g2.plot(column="mm", cmap=cmap, norm=norm, ax=ax, edgecolor="0.55", linewidth=0.25)
    ax.set_aspect(asp); ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(nome, fontsize=FS_TITLE, weight="bold", pad=4)
    # media de bacia na hora de pico, dentro do painel (canto superior esquerdo fica vazio)
    ax.text(0.02, 0.95, f"mean {hieto[p][hpico]:.1f} mm/h", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS_TICK)
    ax.xaxis.set_major_locator(MultipleLocator(0.4)); ax.xaxis.set_major_formatter(fmt_lon)   # 0.4: evita tick colado na borda direita (rotulo invadia o hietograma)
    ax.yaxis.set_major_locator(MultipleLocator(0.2)); ax.yaxis.set_major_formatter(fmt_lat)
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    ax.tick_params(labelsize=FS_TICK, length=2.5,
                   labelleft=(k % 2 == 0), labelbottom=(k >= 3))   # lat so na coluna esq.; lon so nos mapas de baixo
    ax.plot(outlet.x, outlet.y, marker="v", ms=7, color="red", mec="black", mew=0.6, zorder=6)

# colorbar horizontal sob o mapa do MSWEP (mesma escala nos 5 mapas)
cax = ax_in(X_L + 0.15, y_cb, MAP_W - 0.30, CB_H)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")   # extend: telem satura acima do vmax
cb.set_label("precipitation at peak hour (mm/h)", fontsize=FS_LABEL)
cb.ax.xaxis.set_major_locator(MultipleLocator(3))
cb.ax.tick_params(labelsize=FS_TICK, length=2.5)

# hietograma: ocupa a altura inteira da linha 3 (mapa + colorbar), coluna direita
axh = ax_in(X_R, y_cb, MAP_W, y3 + MAP_H - y_cb)
for (p, nome), c, m in zip(PRODS, COLORS, MARKERS):
    axh.plot(range(24), hieto[p], marker=m, ms=3, lw=1.4, color=c, label=nome)
ymax = max(float(np.max(hieto[p])) for p, _ in PRODS)
axh.set_xlim(0, 23); axh.set_ylim(0, 1.45 * ymax)          # folga em cima para a legenda nao cobrir as curvas
axh.xaxis.set_major_locator(MultipleLocator(3))
axh.set_xlabel("hour of day (UTC)", fontsize=FS_LABEL); axh.set_ylabel("precipitation (mm/h)", fontsize=FS_LABEL)
axh.tick_params(labelsize=FS_TICK, length=2.5)
axh.axvline(hpico, color="0.35", ls="--", lw=1.0)
axh.text(hpico + 0.4, 1.40 * ymax, f"{hpico:02d}:00 UTC", ha="left", va="top", fontsize=FS_TICK, color="0.35")
axh.set_title(f"Mean hyetograph, {dpico.day} {dpico:%b %Y}", fontsize=FS_TITLE, weight="bold", pad=4)   # curto p/ caber na largura do painel; "basin-mean" fica na caption
axh.legend(fontsize=FS_TICK, frameon=False, loc="upper left", borderaxespad=0.3,
           handlelength=1.2, handletextpad=0.5, labelspacing=0.25)
axh.grid(alpha=0.3)
# Sem suptitle: padrao Elsevier/JoH, a descricao fica na legenda (caption), nao no corpo da figura.
# bbox tight: sem isso o eixo direito/rotulos saiam cortados (correcao de 13/09/2026, incorporada aqui).
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_produtos_evento.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.12)
print(f"-> {OUT / 'fig_produtos_evento.png'}  (figsize {W:.2f} x {H:.2f} in)")
