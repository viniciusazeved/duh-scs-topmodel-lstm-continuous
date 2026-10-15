# RODADA 7 (2026-10-10), decisao D4 do Vinicius: "catchment" para a unidade estudada e
# "sub-catchment(s)" com hifen; "basin" so em nome proprio e bacia regional ("Preto River basin"
# como nome da bacia inteira, "Paraiba do Sul River basin"); "Rio Preto" unificado em "Preto River".
# Copia de D:/TTD_SCS_LSTM/scripts/mapa_localizacao.py (identica, a menos do fim de linha, a
# versao do commit e680826 do congelamento de 29/09/2026); o original NAO foi alterado.
# Alteracoes (todas marcadas "rodada 7"): rotulos de texto EN (TXT["en"]); ROOT fixo em
# D:/TTD_SCS_LSTM (so leitura); saida no staging da rodada 7 com o nome usado no LaTeX
# (fig_area_estudo.png, layout canonico main_inset("en") de 15/09/2026). Nada de dado ou cor.
# Layout: unica mudanca e a troca de posicao de 'Drainage network' e 'Catchment boundary' na
# legenda (proposta_legenda_mapa/, adotada pelo coordenador em 10/10/2026 ~23:30), porque com
# "catchment" a legenda passava da largura do linewidth e o LaTeX encolheria a figura ~9%.
# Rodar (sem tocar no projeto congelado; GDAL_PAM_ENABLED=NO impede gravar .aux.xml junto do DEM):
#   GDAL_PAM_ENABLED=NO uv run --no-project --python 3.13.10 --with geopandas==1.1.2 \
#     --with pyogrio==0.12.1 --with shapely==2.1.2 --with pyproj==3.7.2 --with rasterio==1.5.0 \
#     --with pandas==3.0.0 --with numpy==2.4.1 --with matplotlib==3.10.8 --with pyarrow==24.0.0 \
#     python mapa_localizacao_rodada7.py inset en
"""Figura 1 - Mapa de localizacao da bacia do rio Preto a montante de Manuel Duarte.

Layout canonico do artigo (main_inset, desde 15/09/2026, item P07 do Paulo):
Painel (a): hipsometria ANADEM (UTM 23S) + 245 ottobacias + rede hidrografica do rio
            Preto + POSTOS PLUVIOMETRICOS do produto de chuva TELEM (IDW dinamico/hora,
            principal do artigo) + estacao fluviometrica MD + divisa RJ/MG.
Inset (b):  America do Sul / Brasil (geograficas), encaixado no canto vazio de (a).

Layouts anteriores, mantidos para reproducao: main() = tres paineis empilhados
(a + Paraiba do Sul + America do Sul); main_inset(inset_rps=True) = rodada 4, com o
Paraiba do Sul como inset (b) e a America do Sul como inset (c).

IMPORTANTE: a rede mostrada NAO e o shapefile fluviometrico generico. Sao os postos
PLUVIOMETRICOS (prefixo ANA 02) usados na interpolacao IDW do produto TELEM -- o
produto de chuva principal do artigo MD-only. Fonte canonica:
  big_ablacao/data/postos/md/postos_telem_series.parquet (serie usada no IDW)
  big_ablacao/data/postos/md/postos_meta.parquet         (coords, local, cobertura)

Projecao: painel (a) EPSG:31983 (UTM 23S), ticks em km sem titulo de eixo (o datum
vai na legenda do .tex); insets EPSG:4326.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
ROOT = Path(r"D:\TTD_SCS_LSTM")  # rodada 7: a copia mora fora do projeto; leitura apenas
DEM = ROOT / "data/raw/rasters/dem.tif"
BACIA_MD = ROOT / "data/raw/bacia/bacia_manuel_duarte.shp"
OTTOS = ROOT / "data/raw/ottobacias/ottobacias_cn_2022.gpkg"
GDB = ROOT / "data/raw/mapa_localizacao/BH.gdb"
UF = ROOT / "data/raw/mapa_localizacao/BR_UF_2025/BR_UF_2025.shp"
POSTOS_META = ROOT / "big_ablacao/data/postos/md/postos_meta.parquet"
POSTOS_SERIE = ROOT / "big_ablacao/data/postos/md/postos_telem_series.parquet"
NE = ROOT / ".venv/Lib/site-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp"
OUTDIR = Path(r"D:\Artigo_JOH\artigo_simulacao\revisao_paulo\rodada7_2026-10-09\staging\figuras")  # rodada 7
OUTDIR.mkdir(parents=True, exist_ok=True)

CRS_UTM = 31983
CRS_GEO = 4326
EST_MD_LONLAT = (-43.5589, -22.0844)  # estacao fluviometrica 58585000
COB_MIN = 0.05  # frac_cob minimo p/ considerar "com registro no periodo"

# Cidades de referencia (nome, lon, lat, ha, va) -- offsets ajustados p/ nao sobrepor
CIDADES = [
    ("Rio de Janeiro", -43.1729, -22.9068, "center", "top"),
    ("São Paulo", -46.6333, -23.5505, "left", "top"),
    ("Juiz de Fora", -43.3434, -21.7642, "left", "bottom"),
    ("Volta Redonda", -44.1041, -22.5231, "right", "bottom"),
    ("Campos dos Goytacazes", -41.3300, -21.7547, "right", "top"),
]

# Rotulos PT (avaliacao com o Hugo) e EN (submissao JoH)
TXT = {
    "pt": {
        "title_a": "(a) Bacia em Manuel Duarte: hipsometria, ottobacias, drenagem e pluviômetros (produto IDW)",
        "cbar": "Altitude (m)",
        "leg_est": "Estação fluviométrica MD (58585000)",
        "leg_p_in": "Pluviômetro telem. na bacia (n={n})",
        "leg_p_out": "Pluviômetro telem. no entorno (n={n})",
        "leg_rede": "Rede de drenagem",
        "leg_divisa": "Divisa RJ/MG",
        "leg_ottos": "245 ottobacias",
        "leg_md": "Bacia em Manuel Duarte (3.117 km²)",
        "leg_rps": "Bacia do Paraíba do Sul",
        "leg_rp": "Bacia do rio Preto",
        "title_b": "(b) Contexto na bacia do rio Paraíba do Sul",
        "title_c": "(c) América do Sul",
        "area_estudo": "área de\nestudo",
        # layout inset (P07): rotulos curtos, o resto vai na legenda do .tex
        "title_a_in": "(a) Bacia do rio Preto a montante de Manuel Duarte",
        "title_rps_in": "(b) Bacia do Paraíba do Sul",
        "title_sa_in": "({k}) América do Sul",
        "leg_est_in": "Estação fluviométrica Manuel Duarte (58585000)",
        "leg_p_in_in": "Pluviômetro telem. dentro da bacia (n={n})",
        "leg_p_out_in": "Pluviômetro telem. fora da bacia (n={n})",
        "leg_divisa_in": "Divisa RJ/MG",
        "leg_md_in": "Limite da bacia (3.117 km²)",
        # titulos de eixo: so no modo inset_rps (rodada 4); o canonico nao os usa
        "xlabel_in": "Easting (km), UTM 23S / SIRGAS 2000",
        "ylabel_in": "Northing (km)",
    },
    "en": {
        "title_a": "(a) Catchment at Manuel Duarte: hypsometry, sub-catchments, drainage and rain gauges (IDW product)",
        "cbar": "Elevation (m)",
        "leg_est": "MD streamflow gauge (58585000)",
        "leg_p_in": "Telemetric rain gauge in catchment (n={n})",
        "leg_p_out": "Telemetric rain gauge in surroundings (n={n})",
        "leg_rede": "Drainage network",
        "leg_divisa": "RJ/MG state boundary",
        "leg_ottos": "245 sub-catchments",
        "leg_md": "Catchment at Manuel Duarte (3,117 km²)",
        "leg_rps": "Paraíba do Sul River basin",
        "leg_rp": "Preto River basin",
        "title_b": "(b) Setting within the Paraíba do Sul River basin",
        "title_c": "(c) South America",
        "area_estudo": "study\narea",
        # layout inset (P07): rotulos curtos, o resto vai na legenda do .tex.
        # "gauging station" segue o texto do artigo (secao 2.1); evita "streamflow".
        "title_a_in": "(a) Preto River basin upstream of Manuel Duarte",
        "title_rps_in": "(b) Paraíba do Sul basin",
        "title_sa_in": "({k}) South America",
        "leg_est_in": "Manuel Duarte gauging station (58585000)",
        # "inside/outside the basin" (P07, 15/09/2026): mais curto que "in the
        # surroundings", o que afasta "(n=21)" da 2a coluna da legenda sem alargar
        # o quadro; o texto (secao 2.2) fala em "6 within the basin and 21 in the
        # surrounding area", todos a menos de 20 km da divisa.
        # rodada 7 (D4): "basin" -> "catchment" nestes dois rotulos.
        "leg_p_in_in": "Telemetric rain gauge inside the catchment (n={n})",
        "leg_p_out_in": "Telemetric rain gauge outside the catchment (n={n})",
        "leg_divisa_in": "RJ/MG state border",
        "leg_md_in": "Catchment boundary (3,117 km²)",
        # titulos de eixo: so no modo inset_rps (rodada 4); o canonico nao os usa
        "xlabel_in": "Easting (km), UTM 23S / SIRGAS 2000",
        "ylabel_in": "Northing (km)",
    },
}

C_BACIA = "#1a1a1a"
C_OTTO = "#ffffff"
C_POSTO_IN = "#08519c"
C_POSTO_OUT = "#3182bd"
C_EST = "#d7191c"
C_RIO = "#2166ac"
C_RP = "#2c7fb8"
C_RPS = "#dfe3e6"
C_MD_FILL = "#d7191c"
C_DIVISA = "#333333"


def carregar():
    bacia = gpd.read_file(BACIA_MD).to_crs(CRS_UTM).dissolve()
    ottos = gpd.read_file(OTTOS).to_crs(CRS_UTM)
    rp = gpd.read_file(GDB, layer="bh_rp").to_crs(CRS_UTM)
    rps = gpd.read_file(GDB, layer="bh_rps").to_crs(CRS_UTM)
    hidro = gpd.read_file(GDB, layer="hidro").to_crs(CRS_UTM)
    hidro["nuareamont"] = hidro["nuareamont"].astype(float)
    uf = gpd.read_file(UF).to_crs(CRS_GEO)
    est = gpd.GeoSeries(
        gpd.points_from_xy([EST_MD_LONLAT[0]], [EST_MD_LONLAT[1]]), crs=CRS_GEO
    ).to_crs(CRS_UTM)
    return bacia, ottos, rp, rps, hidro, uf, est


def carregar_postos_idw():
    """Postos PLUVIOMETRICOS efetivamente usados no IDW do produto TELEM."""
    meta = pd.read_parquet(POSTOS_META)
    meta["codigo"] = meta["codigo"].astype(str).str.zfill(8)
    ser = pd.read_parquet(POSTOS_SERIE)
    cods = set(ser["codigo"].astype(str).str.zfill(8).unique())
    m = meta[meta["codigo"].isin(cods)].copy()
    m["tem_dado"] = m["frac_cob"] >= COB_MIN
    g = gpd.GeoDataFrame(
        m, geometry=gpd.points_from_xy(m["lon"], m["lat"]), crs=CRS_GEO
    ).to_crs(CRS_UTM)
    return g


def lw_area(area, amin, amax, lo, hi):
    a = (np.sqrt(np.clip(area, amin, amax)) - np.sqrt(amin)) / (np.sqrt(amax) - np.sqrt(amin))
    return lo + a * (hi - lo)


def divisa_rj_mg(uf_geo):
    rj = uf_geo[uf_geo["SIGLA_UF"] == "RJ"].geometry.union_all()
    mg = uf_geo[uf_geo["SIGLA_UF"] == "MG"].geometry.union_all()
    return gpd.GeoSeries([rj.boundary.intersection(mg.boundary)], crs=CRS_GEO)


def hipsometria(ax, bacia):
    with rasterio.open(DEM) as src:
        out, transform = rio_mask(
            src, [bacia.geometry.iloc[0]], crop=True, filled=True, nodata=np.nan
        )
    elev = out[0].astype("float64")
    valid = np.isfinite(elev) & (elev > 0)
    elev[~valid] = np.nan
    # vmax = maximo real (nao p99): divisores ao norte passam de 2.000 m e o texto
    # do artigo (secao 2.1) cita "over 2,000 m" -- a rampa precisa cobrir isso
    vmin = float(np.nanpercentile(elev, 1))
    vmax = float(np.nanmax(elev))
    base = plt.get_cmap("terrain")
    cmap = mcolors.LinearSegmentedColormap.from_list("hypso", base(np.linspace(0.25, 1.0, 256)))
    norm = Normalize(vmin=vmin, vmax=vmax)
    ls = LightSource(azdeg=315, altdeg=45)
    filled = np.where(valid, elev, vmin)
    rgb = ls.shade(filled, cmap=cmap, norm=norm, blend_mode="soft", vert_exag=3, dx=30, dy=30)
    if rgb.shape[-1] == 3:
        rgb = np.dstack([rgb, np.ones(rgb.shape[:2])])
    rgb[..., 3] = np.where(valid, 1.0, 0.0)
    nrows, ncols = elev.shape
    left, top = transform.c, transform.f
    right = left + ncols * transform.a
    bottom = top + nrows * transform.e
    ax.imshow(rgb, extent=[left, right, bottom, top], origin="upper",
              interpolation="bilinear", zorder=1)
    return norm, cmap


def barra_escala(ax, x0, y0, comp_m=20_000, label="20 km"):
    ax.plot([x0, x0 + comp_m], [y0, y0], color="k", lw=3, solid_capstyle="butt", zorder=10)
    for x in (x0, x0 + comp_m):
        ax.plot([x, x], [y0, y0 + comp_m * 0.04], color="k", lw=1.2, zorder=10)
    ax.text(x0 + comp_m / 2, y0 + comp_m * 0.10, label, ha="center", va="bottom",
            fontsize=8, fontweight="bold", zorder=10)


def seta_norte(ax, x=0.965, y=0.82, tam=0.075):
    """Seta de norte classica (fracao dos eixos): haste + ponta cheia, 'N' acima."""
    ax.annotate("", xy=(x, y + tam), xytext=(x, y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2.4,
                                mutation_scale=26), zorder=10)
    ax.text(x, y + tam + 0.014, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=14, zorder=10)


def km_fmt(v, _):
    return f"{v/1000:.0f}"


def main(lang: str = "pt"):
    t = TXT[lang]
    bacia, ottos, rp, rps, hidro, uf, est = carregar()
    postos = carregar_postos_idw()
    uf_utm = uf.to_crs(CRS_UTM)
    div_geo = divisa_rj_mg(uf)
    div_utm = div_geo.to_crs(CRS_UTM)

    p_in = postos[postos["local"] == "dentro"]
    p_out = postos[postos["local"] == "buffer"]
    print(f"Produto TELEM: {len(postos)} postos pluviométricos IDW | dentro={len(p_in)} entorno={len(p_out)}")

    hidro_md = gpd.clip(hidro, bacia.geometry.iloc[0])
    hidro_md = hidro_md[hidro_md["nuareamont"] >= 60]
    lw_md = lw_area(hidro_md["nuareamont"].values, 60, 3117.0, 0.4, 2.0)

    fig = plt.figure(figsize=(12.5, 11.7))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.20, 0.28, 1.05], width_ratios=[1.9, 1.0],
                          wspace=0.10, hspace=0.24)
    ax = fig.add_subplot(gs[0, :])
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")
    ax_r = fig.add_subplot(gs[2, 0])
    ax_c = fig.add_subplot(gs[2, 1])

    # ----------------- (a) PAINEL PRINCIPAL ----------------- #
    norm, cmap = hipsometria(ax, bacia)
    # 245 ottobacias com halo (branco por baixo + linha escura fina por cima) p/ ficarem
    # visiveis tanto sobre o verde quanto sobre o marrom da hipsometria, sem poluir
    ottos.boundary.plot(ax=ax, color="white", linewidth=0.9, alpha=0.40, zorder=1.9)
    ottos.boundary.plot(ax=ax, color="0.15", linewidth=0.32, alpha=0.72, zorder=2.0)
    hidro_md.plot(ax=ax, color=C_RIO, linewidth=lw_md, alpha=0.85, zorder=3)
    div_utm.plot(ax=ax, color="white", linewidth=1.6, linestyle=(0, (6, 3)), zorder=3.4)
    div_utm.plot(ax=ax, color=C_DIVISA, linewidth=0.8, linestyle=(0, (6, 3)), zorder=3.5)
    bacia.boundary.plot(ax=ax, color=C_BACIA, linewidth=1.8, zorder=4)

    p_out.plot(ax=ax, marker="^", color=C_POSTO_OUT, edgecolor="white",
               markersize=40, linewidth=0.5, zorder=5.5)
    p_in.plot(ax=ax, marker="^", color=C_POSTO_IN, edgecolor="white",
              markersize=64, linewidth=0.6, zorder=6)
    est.plot(ax=ax, marker="*", color=C_EST, edgecolor="k", markersize=340,
             linewidth=0.8, zorder=7)

    minx = min(bacia.total_bounds[0], postos.total_bounds[0])
    miny = min(bacia.total_bounds[1], postos.total_bounds[1])
    maxx = max(bacia.total_bounds[2], postos.total_bounds[2])
    maxy = max(bacia.total_bounds[3], postos.total_bounds[3])
    padx = (maxx - minx) * 0.02
    pady = (maxy - miny) * 0.04
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)
    ax.set_aspect("equal", adjustable="datalim")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(km_fmt))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(km_fmt))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Easting (km), UTM 23S / SIRGAS 2000", fontsize=9)
    ax.set_ylabel("Northing (km)", fontsize=9)
    ax.set_title(t["title_a"], loc="left", fontsize=10, fontweight="bold")

    for sig in ("MG", "RJ"):
        porc = gpd.overlay(bacia, uf_utm[uf_utm["SIGLA_UF"] == sig], how="intersection")
        if len(porc):
            pt = porc.geometry.union_all().representative_point()
            ax.text(pt.x, pt.y, sig, fontsize=12, fontweight="bold", color="0.12",
                    ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.5", alpha=0.72))

    seta_norte(ax)

    # Quadro geral: itens dos paineis (a) e (b) juntos (sem legenda dentro do inset b).
    # Bacia em MD = entrada unica (preenchimento vermelho do inset b + contorno preto do a).
    md_handle = (
        Patch(facecolor=C_MD_FILL, edgecolor="none", alpha=0.55),
        Line2D([0], [0], color=C_BACIA, lw=1.8),
    )
    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=C_EST,
               markeredgecolor="k", markersize=13),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=C_POSTO_IN,
               markeredgecolor="white", markersize=9),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=C_POSTO_OUT,
               markeredgecolor="white", markersize=8),
        Line2D([0], [0], color=C_RIO, lw=1.8),
        Line2D([0], [0], color=C_DIVISA, lw=1.0, linestyle=(0, (4, 2))),
        Line2D([0], [0], color="0.15", lw=0.9),
        md_handle,
        Patch(facecolor=C_RPS, edgecolor="#5a6066"),
        Patch(facecolor="none", edgecolor="#0b3d91"),
    ]
    labels = [
        t["leg_est"],
        t["leg_p_in"].format(n=len(p_in)),
        t["leg_p_out"].format(n=len(p_out)),
        t["leg_rede"],
        t["leg_divisa"],
        t["leg_ottos"],
        t["leg_md"],
        t["leg_rps"],
        t["leg_rp"],
    ]
    ax_leg.legend(handles, labels, loc="center", ncol=3, fontsize=7.3, framealpha=1.0,
                  edgecolor="0.6", borderpad=0.7, columnspacing=1.2, handletextpad=0.4,
                  labelspacing=0.5, handler_map={tuple: HandlerTuple(ndivide=1)})

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cax = ax.inset_axes([0.785, 0.105, 0.19, 0.024])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(t["cbar"], fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ----------------- (b) REGIONAL ----------------- #
    rps_g = rps.to_crs(CRS_GEO)
    rp_g = rp.to_crs(CRS_GEO)
    md_g = bacia.to_crs(CRS_GEO)
    mask_reg = hidro["nuareamont"] >= 2500
    hidro_reg = hidro[mask_reg].to_crs(CRS_GEO)
    lw_reg = lw_area(hidro[mask_reg]["nuareamont"].values, 2500, float(hidro["nuareamont"].max()), 0.3, 1.6)

    cx = [c[1] for c in CIDADES]
    cy = [c[2] for c in CIDADES]
    rb = rps_g.total_bounds
    x0 = min(rb[0], min(cx)) - 0.3
    x1 = max(rb[2], max(cx)) + 0.3
    y0 = min(rb[1], min(cy)) - 0.3
    y1 = max(rb[3], max(cy)) + 0.3

    uf.boundary.plot(ax=ax_r, color="0.55", linewidth=0.5, linestyle=(0, (4, 2)), zorder=1)
    div_geo.plot(ax=ax_r, color=C_DIVISA, linewidth=1.1, zorder=1.5)
    rps_g.plot(ax=ax_r, color=C_RPS, edgecolor="#8a8f94", linewidth=0.7, alpha=0.55, zorder=2)
    hidro_reg.plot(ax=ax_r, color=C_RIO, linewidth=lw_reg, alpha=0.8, zorder=3)
    rps_g.boundary.plot(ax=ax_r, color="#5a6066", linewidth=0.8, zorder=3.5)
    rp_g.plot(ax=ax_r, color="none", edgecolor="#0b3d91", linewidth=1.0, zorder=4)
    md_g.plot(ax=ax_r, color=C_MD_FILL, edgecolor="k", linewidth=0.5, alpha=0.55, zorder=5)

    for nome, lon, lat, ha, va in CIDADES:
        ax_r.plot(lon, lat, marker="o", color="white", markersize=4.0,
                  markeredgecolor="k", markeredgewidth=0.6, zorder=6)
        dx = 0.10 if ha == "left" else (-0.10 if ha == "right" else 0.0)
        dy = 0.06 if va == "bottom" else (-0.06 if va == "top" else 0.0)
        ax_r.text(lon + dx, lat + dy, nome, fontsize=6.0, ha=ha, va=va, zorder=6,
                  bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.6))

    ax_r.set_xlim(x0, x1)
    ax_r.set_ylim(y0, y1)
    ax_r.set_aspect(1.0 / np.cos(np.radians(22)), adjustable="datalim")
    ax_r.tick_params(labelsize=6)
    ax_r.set_xlabel("Longitude", fontsize=7)
    ax_r.set_ylabel("Latitude", fontsize=7)
    ax_r.set_title(t["title_b"], loc="left", fontsize=9.5, fontweight="bold")

    # ----------------- (c) CONTINENTAL ----------------- #
    world = gpd.read_file(NE)
    sa = world[world["continent"] == "South America"].to_crs(CRS_GEO)
    br = world[world["name"] == "Brazil"].to_crs(CRS_GEO)
    sa.plot(ax=ax_c, color="#f2f2f2", edgecolor="#9a9a9a", linewidth=0.4)
    br.plot(ax=ax_c, color="#cfe0cf", edgecolor="#4d4d4d", linewidth=0.6)
    ax_c.plot(*EST_MD_LONLAT, marker="o", color=C_EST, markersize=7,
              markeredgecolor="k", markeredgewidth=0.6, zorder=5)
    ax_c.annotate(t["area_estudo"], xy=EST_MD_LONLAT,
                  xytext=(EST_MD_LONLAT[0] + 9, EST_MD_LONLAT[1] + 4), fontsize=7,
                  ha="left", va="center", arrowprops=dict(arrowstyle="-", color="k", lw=0.6))
    sb = sa.total_bounds
    ax_c.set_xlim(sb[0] - 1, sb[2] + 1)
    ax_c.set_ylim(sb[1] - 1, sb[3] + 1)
    ax_c.set_aspect(1.0 / np.cos(np.radians(20)), adjustable="datalim")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.set_title(t["title_c"], loc="left", fontsize=9.5, fontweight="bold")
    for s in ax_c.spines.values():
        s.set_edgecolor("0.5")

    suf = "" if lang == "pt" else f"_{lang}"
    out_png = OUTDIR / f"mapa_localizacao_md{suf}.png"
    out_pdf = OUTDIR / f"mapa_localizacao_md{suf}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Salvo: {out_png}")
    print(f"Salvo: {out_pdf}")


# =========================================================================== #
# LAYOUT "INSET" (item P07 do Paulo)
#
# Rodada 4 (04/09/2026): painel (a) ocupa a figura inteira; (b) Paraiba do Sul
# e (c) America do Sul viraram INSETS nos cantos vazios de (a).
#
# Rodada 5 (15/09/2026, decisao do Vinicius de atender o P07 integralmente):
# o inset do Paraiba do Sul SAI, a America do Sul passa a ser o unico inset
# (letra b), a legenda perde os itens que so existiam por causa dele e as
# fontes sobem. O inset do Paraiba do Sul continua disponivel via
# inset_rps=True (CLI: modo "inset_rps") para reproduzir a versao enviada.
# Segunda passada no mesmo dia (verificador): a bacia tinha ficado ~3,6%
# MENOR no papel do que na rodada 4; sairam os titulos de eixo, a margem
# esquerda passou a ser medida do rotulo de tick e a folga lateral caiu de
# 2% para 1%, para a bacia passar dos 9,2 cm da rodada 4 (alvo >= 9,4 cm).
#
# TAMANHO: a figura e desenhada com a LARGURA EXATA do \linewidth do elsarticle
# em modo review (345 pt = 4,79 pol, medido com \the\linewidth em 15/09/2026),
# de modo que "fontsize=7.5" no script vale 7,5 pt no papel. A versao de
# 12,5 pol era reduzida a 0,38x e a de 7,4 pol a 0,65x, e foi isso que o Paulo
# leu como "fonte pequena". Nada abaixo de 7 pt aqui.
# =========================================================================== #

FIG_W_IN = 345.0 / 72.0
FIG_DPI = 450  # 1:1 com o papel; 450 dpi preserva o detalhe das 245 ottobacias
# Retangulos (fracao dos eixos do painel a) das areas vazias aproveitadas.
# Conferidos contra o poligono da bacia e contra os 27 pluviometros (15/09).
RECT_SA = (0.008, 0.530, 0.178, 0.400)   # America do Sul, canto superior esquerdo
# (x=0.008: com 0.015 a moldura do inset encostava no pluviometro de (565, 7567) km)
RECT_CB = (0.730, 0.190, 0.215, 0.032)   # barra de cor horizontal, canto inferior direito
RECT_RPS = (0.735, 0.015, 0.258, 0.300)  # so com inset_rps=True (rodada 4)
RECT_CB_RPS = (0.895, 0.450, 0.016, 0.230)
CIDADES_INSET = ("Juiz de Fora", "Rio de Janeiro")


def _seta_norte(ax, x, y, tam, lw, fs):
    ax.annotate("", xy=(x, y + tam), xytext=(x, y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=lw,
                                mutation_scale=15), zorder=10)
    ax.text(x, y + tam + 0.012, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=fs, zorder=10)


def _escala_geo(ax, x0, y0, km=100.0, lat_ref=-22.0, fs=5.2):
    """Barra de escala em graus de longitude (inset Paraiba do Sul, sem eixos)."""
    dl = km / (111.32 * np.cos(np.radians(lat_ref)))
    ax.plot([x0, x0 + dl], [y0, y0], color="k", lw=1.6, solid_capstyle="butt", zorder=9)
    ax.text(x0 + dl / 2, y0 + 0.06, f"{km:.0f} km", ha="center", va="bottom",
            fontsize=fs, zorder=9,
            bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.7))


def _inset_rps(axb, dados, t):
    """Contexto na bacia do Paraiba do Sul, sem eixos, com barra de escala (rodada 4)."""
    bacia, rp, rps, hidro, uf, div_geo = dados
    rps_g = rps.to_crs(CRS_GEO)
    rp_g = rp.to_crs(CRS_GEO)
    md_g = bacia.to_crs(CRS_GEO)
    mask = hidro["nuareamont"] >= 2500
    hidro_reg = hidro[mask].to_crs(CRS_GEO)
    lw_reg = lw_area(hidro[mask]["nuareamont"].values, 2500,
                     float(hidro["nuareamont"].max()), 0.25, 1.0)

    uf.boundary.plot(ax=axb, color="0.6", linewidth=0.35, linestyle=(0, (3, 2)), zorder=1)
    div_geo.plot(ax=axb, color=C_DIVISA, linewidth=0.6, zorder=1.5)
    rps_g.plot(ax=axb, color=C_RPS, edgecolor="#8a8f94", linewidth=0.4, alpha=0.60, zorder=2)
    hidro_reg.plot(ax=axb, color=C_RIO, linewidth=lw_reg, alpha=0.85, zorder=3)
    rps_g.boundary.plot(ax=axb, color="#5a6066", linewidth=0.6, zorder=3.5)
    rp_g.plot(ax=axb, color="none", edgecolor="#0b3d91", linewidth=0.8, zorder=4)
    md_g.plot(ax=axb, color=C_MD_FILL, edgecolor="k", linewidth=0.4, alpha=0.60, zorder=5)

    for nome, lon, lat, ha, va in CIDADES:
        if nome not in CIDADES_INSET:
            continue
        axb.plot(lon, lat, marker="o", color="white", markersize=2.6,
                 markeredgecolor="k", markeredgewidth=0.45, zorder=6)
        dy = 0.10 if va == "bottom" else -0.10
        axb.text(lon, lat + dy, nome, fontsize=5.2, ha="center", va=va, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.65))

    rb = rps_g.total_bounds
    padx = (rb[2] - rb[0]) * 0.02
    pady = (rb[3] - rb[1]) * 0.05
    axb.set_xlim(rb[0] - padx, rb[2] + padx)
    axb.set_ylim(rb[1] - pady, rb[3] + pady)
    axb.set_aspect(1.0 / np.cos(np.radians(22)), adjustable="datalim")
    axb.set_xticks([])
    axb.set_yticks([])
    for s in axb.spines.values():
        s.set_edgecolor("0.35")
        s.set_linewidth(0.7)
    _escala_geo(axb, rb[0] + 0.05, rb[1] + 0.10)


def _inset_sa(axc, t, fs=7.0):
    """Localizacao na America do Sul (Brasil destacado, ponto na area de estudo)."""
    world = gpd.read_file(NE)
    sa = world[world["continent"] == "South America"].to_crs(CRS_GEO)
    br = world[world["name"] == "Brazil"].to_crs(CRS_GEO)
    sa.plot(ax=axc, color="#f2f2f2", edgecolor="#9a9a9a", linewidth=0.3)
    br.plot(ax=axc, color="#cfe0cf", edgecolor="#4d4d4d", linewidth=0.5)
    axc.plot(*EST_MD_LONLAT, marker="o", color=C_EST, markersize=5.0,
             markeredgecolor="k", markeredgewidth=0.5, zorder=5)
    # rotulo sobre o Atlantico, ao sul do ponto: cabe dentro da caixa do inset
    axc.annotate(t["area_estudo"], xy=EST_MD_LONLAT,
                 xytext=(EST_MD_LONLAT[0] - 4.0, EST_MD_LONLAT[1] - 19.0), fontsize=fs,
                 ha="center", va="center", linespacing=0.95,
                 arrowprops=dict(arrowstyle="-", color="k", lw=0.5,
                                 shrinkA=0, shrinkB=3))
    sb = sa.total_bounds
    axc.set_xlim(sb[0] - 1, sb[2] + 1)
    axc.set_ylim(sb[1] - 1, sb[3] + 1)
    axc.set_aspect(1.0 / np.cos(np.radians(20)), adjustable="datalim")
    axc.set_xticks([])
    axc.set_yticks([])
    for s in axc.spines.values():
        s.set_edgecolor("0.35")
        s.set_linewidth(0.7)


def main_inset(lang: str = "en", inset_rps: bool = False):
    t = TXT[lang]
    bacia, ottos, rp, rps, hidro, uf, est = carregar()
    postos = carregar_postos_idw()
    uf_utm = uf.to_crs(CRS_UTM)
    div_geo = divisa_rj_mg(uf)
    div_utm = div_geo.to_crs(CRS_UTM)

    p_in = postos[postos["local"] == "dentro"]
    p_out = postos[postos["local"] == "buffer"]
    print(f"[inset] TELEM: {len(postos)} postos IDW | dentro={len(p_in)} entorno={len(p_out)}")

    hidro_md = gpd.clip(hidro, bacia.geometry.iloc[0])
    hidro_md = hidro_md[hidro_md["nuareamont"] >= 60]
    lw_md = lw_area(hidro_md["nuareamont"].values, 60, 3117.0, 0.30, 1.10)

    # --- geometria da pagina --------------------------------------------- #
    minx = min(bacia.total_bounds[0], postos.total_bounds[0])
    miny = min(bacia.total_bounds[1], postos.total_bounds[1])
    maxx = max(bacia.total_bounds[2], postos.total_bounds[2])
    maxy = max(bacia.total_bounds[3], postos.total_bounds[3])
    # folga lateral em fracao da extensao (P07: 1,2% no canonico, 2% na rodada 4).
    # A 1% os triangulos extremos (oeste e leste) ficavam a ~0,4 mm da moldura;
    # 1,2% da ~0,7 mm e custa 0,4% de largura da bacia (conferido no PNG, 15/09).
    padx = (maxx - minx) * (0.02 if inset_rps else 0.012)
    pady = (maxy - miny) * 0.04
    X0, X1 = minx - padx, maxx + padx
    Y0, Y1 = miny - pady, maxy + pady
    razao = (X1 - X0) / (Y1 - Y0)

    # Larguras/alturas em polegadas (1:1 com o papel).
    tick_fs, tick_len, tick_pad = 7.5, 2.0, 1.0
    fig = plt.figure(figsize=(FIG_W_IN, FIG_W_IN))  # altura acertada logo abaixo
    if inset_rps:
        # rodada 4: titulos de eixo e margem fixa da versao enviada aos coautores
        esq, larg, h_rot = 0.100, 0.885, 0.40
    else:
        # P07 (15/09/2026, exigencia do verificador): sem titulos de eixo, porque
        # "UTM 23S / SIRGAS 2000" ja esta na legenda do .tex e cada ponto de
        # margem e largura que a bacia perde no papel. A margem esquerda e MEDIDA
        # do rotulo de tick mais largo ("7580" a 7,5 pt), nao chutada.
        ren = fig.canvas.get_renderer()
        tmp = [fig.text(0, 0, km_fmt(v, None), fontsize=tick_fs) for v in (Y0, Y1)]
        w_ytick = max(tx.get_window_extent(ren).width for tx in tmp) / fig.dpi
        for tx in tmp:
            tx.remove()
        esq = (w_ytick + (tick_len + tick_pad + 1.0) / 72.0) / FIG_W_IN  # 1 pt de folga
        larg = 1.0 - esq - 0.004      # ~0,02 pol de margem direita (so a moldura)
        h_rot = 0.20                  # so os rotulos dos ticks de x
    w_map = FIG_W_IN * larg
    h_map = w_map / razao          # aspecto igual sem expansao de limites
    h_tit, h_pad = 0.22, 0.03
    h_leg = 0.98 if inset_rps else 0.78
    FIG_H_IN = h_tit + h_map + h_rot + h_leg + h_pad
    fig.set_size_inches(FIG_W_IN, FIG_H_IN)

    ax = fig.add_axes([esq, (h_pad + h_leg + h_rot) / FIG_H_IN, larg, h_map / FIG_H_IN])
    ax_leg = fig.add_axes([0.02, h_pad / FIG_H_IN, 0.96, h_leg / FIG_H_IN])
    ax_leg.axis("off")

    # ----------------- (a) PAINEL PRINCIPAL ------------------------------ #
    norm, cmap = hipsometria(ax, bacia)
    ottos.boundary.plot(ax=ax, color="white", linewidth=0.70, alpha=0.40, zorder=1.9)
    ottos.boundary.plot(ax=ax, color="0.15", linewidth=0.28, alpha=0.72, zorder=2.0)
    hidro_md.plot(ax=ax, color=C_RIO, linewidth=lw_md, alpha=0.85, zorder=3)
    div_utm.plot(ax=ax, color="white", linewidth=1.3, linestyle=(0, (6, 3)), zorder=3.4)
    div_utm.plot(ax=ax, color=C_DIVISA, linewidth=0.65, linestyle=(0, (6, 3)), zorder=3.5)
    bacia.boundary.plot(ax=ax, color=C_BACIA, linewidth=1.2, zorder=4)

    p_out.plot(ax=ax, marker="^", color=C_POSTO_OUT, edgecolor="white",
               markersize=24, linewidth=0.45, zorder=5.5)
    # rodada 7 (layout, 11/10): o posto 02243303 fica junto da estacao e sumia sob a
    # estrela; os postos internos passam a ser desenhados por cima dela
    p_in.plot(ax=ax, marker="^", color=C_POSTO_IN, edgecolor="white",
              markersize=36, linewidth=0.5, zorder=8)
    est.plot(ax=ax, marker="*", color=C_EST, edgecolor="k", markersize=150,
             linewidth=0.6, zorder=7)

    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(km_fmt))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(km_fmt))
    ax.tick_params(labelsize=tick_fs, length=tick_len, pad=tick_pad)
    if inset_rps:
        ax.set_xlabel(t["xlabel_in"], fontsize=8, labelpad=2)
        ax.set_ylabel(t["ylabel_in"], fontsize=8, labelpad=2)
    ax.set_title(t["title_a_in"], loc="left", fontsize=8.5, fontweight="bold", pad=3)

    # siglas RJ/MG (item H01 do Hugo)
    for sig in ("MG", "RJ"):
        porc = gpd.overlay(bacia, uf_utm[uf_utm["SIGLA_UF"] == sig], how="intersection")
        if len(porc):
            pt = porc.geometry.union_all().representative_point()
            ax.text(pt.x, pt.y, sig, fontsize=9, fontweight="bold", color="0.12",
                    ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.5", alpha=0.72))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    if inset_rps:
        _seta_norte(ax, x=0.908, y=0.710, tam=0.055, lw=1.3, fs=9)
        cax = ax.inset_axes(list(RECT_CB_RPS))
        cb = fig.colorbar(sm, cax=cax, orientation="vertical")
        cb.set_label(t["cbar"], fontsize=7)
        cb.ax.tick_params(labelsize=6.2, length=2, pad=1.2)
    else:
        # canto inferior direito, vazio desde a saida do inset do Paraiba do Sul;
        # a seta fica acima da barra de cor para nao encostar no tick "2500"
        _seta_norte(ax, x=0.972, y=0.290, tam=0.110, lw=1.4, fs=9)
        cax = ax.inset_axes(list(RECT_CB))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.ax.xaxis.set_label_position("top")
        cb.set_label(t["cbar"], fontsize=7.5, labelpad=2)
        cb.set_ticks([500, 1500, 2500])  # rampa vai de ~420 m (p1) a 2.600 m (max)
        cb.ax.tick_params(labelsize=7.5, length=2, pad=1.2)
    cb.outline.set_linewidth(0.5)

    # ----------------- INSETS ------------------------------------------- #
    letra_sa = "c" if inset_rps else "b"
    if inset_rps:
        axb = ax.inset_axes(list(RECT_RPS))
        _inset_rps(axb, (bacia, rp, rps, hidro, uf, div_geo), t)
        ax.text(RECT_RPS[0], RECT_RPS[1] + RECT_RPS[3] + 0.012, t["title_rps_in"],
                transform=ax.transAxes, fontsize=7, fontweight="bold",
                ha="left", va="bottom", zorder=9)

    axc = ax.inset_axes(list(RECT_SA))
    _inset_sa(axc, t, fs=7.5)
    ax.text(RECT_SA[0], RECT_SA[1] + RECT_SA[3] + 0.012, t["title_sa_in"].format(k=letra_sa),
            transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="bottom", zorder=9)

    # ----------------- LEGENDA (duas colunas) ---------------------------- #
    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=C_EST,
               markeredgecolor="k", markersize=11),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=C_POSTO_IN,
               markeredgecolor="white", markersize=7),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=C_POSTO_OUT,
               markeredgecolor="white", markersize=6),
        Line2D([0], [0], color=C_RIO, lw=1.4),
        Line2D([0], [0], color=C_DIVISA, lw=0.9, linestyle=(0, (4, 2))),
        Line2D([0], [0], color="0.15", lw=0.8),
    ]
    labels = [
        t["leg_est_in"],
        t["leg_p_in_in"].format(n=len(p_in)),
        t["leg_p_out_in"].format(n=len(p_out)),
        t["leg_rede"],
        t["leg_divisa_in"],
        t["leg_ottos"],
    ]
    if inset_rps:
        # bacia em MD = preenchimento vermelho do inset + contorno preto do painel a
        handles += [
            (Patch(facecolor=C_MD_FILL, edgecolor="none", alpha=0.60),
             Line2D([0], [0], color=C_BACIA, lw=1.4)),
            Patch(facecolor=C_RPS, edgecolor="#5a6066"),
            Patch(facecolor="none", edgecolor="#0b3d91"),
        ]
        labels += [t["leg_md"], t["leg_rps"], t["leg_rp"]]
    else:
        handles.append(Line2D([0], [0], color=C_BACIA, lw=1.4))
        labels.append(t["leg_md_in"])
        # rodada 7 (adotada): limite da bacia na 1a coluna e drenagem na 2a (ncol=2 preenche por
        # coluna: 1a = estacao, pluviometros dentro/fora, limite; 2a = divisa, sub-bacias,
        # drenagem). Com "catchment" os rotulos longos passavam da largura da figura.
        handles[3], handles[6] = handles[6], handles[3]
        labels[3], labels[6] = labels[6], labels[3]
    # espacamentos apertados para a legenda caber nos 4,79 pol sem alargar o
    # bbox final (senao o LaTeX reduz a figura toda para caber no \linewidth).
    # columnspacing 1.5 em (~4 mm) no canonico: com 0.8 em o "(n=21)" ficava a
    # ~2 mm do tracejado da 2a coluna (verificador, 15/09); o rotulo mais curto
    # ("outside the basin") paga o espaco extra.
    ax_leg.legend(handles, labels, loc="center", ncol=2, fontsize=7.5, framealpha=1.0,
                  edgecolor="0.6", borderpad=0.5, handletextpad=0.4,
                  columnspacing=(0.8 if inset_rps else 1.5),
                  handlelength=2.0, labelspacing=0.45,
                  handler_map={tuple: HandlerTuple(ndivide=1)})

    suf = ("" if lang == "pt" else f"_{lang}") + ("_rps" if inset_rps else "")
    # rodada 7: o canonico (inset, en) sai com o nome usado no LaTeX e so em PNG
    # (overleaf_rev02/figuras so tem fig_area_estudo.png; um .pdf de mesmo nome
    # passaria na frente no pdflatex)
    canonico = lang == "en" and not inset_rps
    out_png = OUTDIR / ("fig_area_estudo.png" if canonico else f"mapa_localizacao_md_inset{suf}.png")
    out_pdf = None if canonico else OUTDIR / f"mapa_localizacao_md_inset{suf}.pdf"
    # bbox tight so por seguranca (nada deve ser cortado); pad minimo (0,01 pol)
    # para a largura final ficar igual ao \linewidth: com 0,03 pol o PNG saia
    # 4,82 pol e o LaTeX encolhia tudo 0,6% (P07, 15/09/2026)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    if out_pdf is not None:  # rodada 7
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    # largura da bacia no papel (escala 1:1 com o linewidth): alvo >= 9,4 cm (P07).
    # Usa os limites REAIS apos o draw: com aspect='equal' + adjustable='datalim'
    # o matplotlib pode alargar um pouco o eixo x em vez de respeitar X0/X1.
    xl = ax.get_xlim()
    bw_cm = w_map * (bacia.total_bounds[2] - bacia.total_bounds[0]) / (xl[1] - xl[0]) * 2.54
    plt.close(fig)
    print(f"[inset] figura {FIG_W_IN:.2f} x {FIG_H_IN:.2f} pol | mapa {w_map:.2f} x {h_map:.2f} pol"
          f" | eixo x {(xl[1]-xl[0])/1e3:.1f} km (pedido {(X1-X0)/1e3:.1f})"
          f" | bacia {bw_cm:.2f} cm de largura no papel")
    print(f"Salvo: {out_png}")
    if out_pdf is not None:  # rodada 7
        print(f"Salvo: {out_pdf}")


if __name__ == "__main__":
    import sys

    # modos: inset (canonico, P07 integral) | inset_rps (rodada 4) | paineis (3 paineis)
    modo = sys.argv[1] if len(sys.argv) > 1 else "inset"
    langs = sys.argv[2:] or ["en"]  # rodada 7: so o EN (a D4 e do texto em ingles)
    for lang in langs:
        if modo == "paineis":
            main(lang)
        elif modo == "inset_rps":
            main_inset(lang, inset_rps=True)
        else:
            main_inset(lang)
