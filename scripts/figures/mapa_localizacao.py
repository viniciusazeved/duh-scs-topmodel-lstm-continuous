"""Figure 1: location map of the Preto River catchment upstream of Manuel Duarte.

Panel (a), in SIRGAS 2000 / UTM zone 23S (EPSG:31983, tick labels in km): hypsometry of the
ANADEM v1 terrain model, the 245 sub-catchments, the drainage network, the 27 telemetric rain
gauges of the IDW rainfall forcing (6 inside and 21 outside the catchment), the gauging station
58585000 and the RJ/MG state border. Inset (b): location in South America.

Inputs shipped in the repository:
    data/gis/bacia_manuel_duarte.shp     catchment outline
    data/gis/ottobacias_tc.gpkg          the 245 sub-catchments
    data/gis/naturalearth_lowres/        countries for the inset (Natural Earth)
    data/gauges/                         gauge metadata, and the hourly series that define which
                                         gauges enter the IDW forcing
Three public layers are not redistributed. Pass them by argument or environment variable, or
place them in data/gis/external/ (another folder: --layers-dir or FIG1_LAYERS_DIR) under the
expected names; data/gis/README.md gives the source, licence and expected name of each:
    --dem       ANADEM v1 in EPSG:31983             dem.tif                     FIG1_DEM
    --drainage  drainage reaches with the upstream  BH.gdb (layer "hidro")      FIG1_DRAINAGE
                area field "nuareamont" (km2)
    --states    IBGE 2025 state boundaries          BR_UF_2025/BR_UF_2025.shp   FIG1_STATES

Usage, from the repository root and with the optional "gis" dependencies:
    uv run python scripts/figures/mapa_localizacao.py [--layers-dir DIR] [--out DIR]

Output: outputs/figures/fig_area_estudo.png. The figure is drawn at the 345 pt text width of the
journal template, so the font sizes are the printed sizes, and saved at 450 dpi.

Adapted from the generator of the submitted figure (SHA-256 7bea261e0865eeec...), keeping only
the layout of the manuscript; the drawing code, colours and sizes are unchanged. With the copies
of the three public layers used by the authors (identified in data/gis/README.md) it reproduces
the submitted PNG byte for byte.
"""

from __future__ import annotations

import os

# GDAL must not write .aux.xml files next to the DEM; set before any raster is opened.
os.environ.setdefault("GDAL_PAM_ENABLED", "NO")
# Non-interactive backend: the script only writes the PNG.
os.environ["MPLBACKEND"] = "Agg"

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, Normalize
from matplotlib.lines import Line2D
from rasterio.crs import CRS
from rasterio.mask import mask as rio_mask

ROOT = Path(__file__).resolve().parents[2]
GIS = ROOT / "data" / "gis"
GAUGES = ROOT / "data" / "gauges"
CATCHMENT = GIS / "bacia_manuel_duarte.shp"
SUBCATCHMENTS = GIS / "ottobacias_tc.gpkg"
NATURAL_EARTH = GIS / "naturalearth_lowres" / "naturalearth_lowres.shp"
GAUGE_META = GAUGES / "postos_meta.parquet"
GAUGE_SERIES = GAUGES / "postos_telem_series.parquet"
LAYERS_DIR = GIS / "external"
DEM_NAME = Path("dem.tif")
DRAINAGE_NAME = Path("BH.gdb")
DRAINAGE_LAYER = "hidro"
STATES_NAME = Path("BR_UF_2025") / "BR_UF_2025.shp"
OUT_DEFAULT = ROOT / "outputs" / "figures"
OUT_NAME = "fig_area_estudo.png"

CRS_UTM = 31983  # SIRGAS 2000 / UTM zone 23S
CRS_GEO = 4326
STATION_LONLAT = (-43.5589, -22.0844)  # ANA gauging station 58585000

# Labels. "gauging station" follows Section 2.1 of the manuscript.
TITLE_A = "(a) Preto River basin upstream of Manuel Duarte"
TITLE_B = "(b) South America"
CBAR_LABEL = "Elevation (m)"
STUDY_AREA = "study\narea"
LEG_STATION = "Manuel Duarte gauging station (58585000)"
LEG_GAUGE_IN = "Telemetric rain gauge inside the catchment (n={n})"
LEG_GAUGE_OUT = "Telemetric rain gauge outside the catchment (n={n})"
LEG_OUTLINE = "Catchment boundary (3,117 km²)"
LEG_BORDER = "RJ/MG state border"
LEG_SUBCATCHMENTS = "245 sub-catchments"
LEG_DRAINAGE = "Drainage network"

C_OUTLINE = "#1a1a1a"
C_GAUGE_IN = "#08519c"
C_GAUGE_OUT = "#3182bd"
C_STATION = "#d7191c"
C_RIVER = "#2166ac"
C_BORDER = "#333333"

# The figure is drawn at the exact \linewidth of elsarticle in review mode (345 pt), so that
# fontsize 7.5 here is 7.5 pt on paper; 450 dpi keeps the detail of the 245 sub-catchments.
FIG_W_IN = 345.0 / 72.0
FIG_DPI = 450
# Empty areas of panel (a) used by the inset and the colour bar (fractions of the axes).
RECT_SA = (0.008, 0.530, 0.178, 0.400)  # South America, upper left corner
RECT_CB = (0.730, 0.190, 0.215, 0.032)  # horizontal colour bar, lower right corner


def _resolve(arg: Path | None, env: str, default: Path) -> Path:
    """Command-line argument, else environment variable, else default."""
    if arg is not None:
        return arg
    value = os.environ.get(env)
    return Path(value) if value else default


def _require(path: Path, what: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"{what} not found: {path}\n"
            "See data/gis/README.md for the source and the expected file names of the public "
            "layers, and pass them with --layers-dir, --dem, --drainage and --states."
        )


def _show(path: Path) -> str:
    """Path relative to the repository root when possible (for messages)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_vectors(
    subcatchments: Path, drainage: Path, drainage_layer: str | None, states: Path
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoSeries]:
    """Catchment outline, sub-catchments, drainage reaches, states and the gauging station."""
    catchment = gpd.read_file(CATCHMENT).to_crs(CRS_UTM).dissolve()
    subs = gpd.read_file(subcatchments).to_crs(CRS_UTM)
    reaches = gpd.read_file(drainage, layer=drainage_layer).to_crs(CRS_UTM)
    reaches["nuareamont"] = reaches["nuareamont"].astype(float)
    states_geo = gpd.read_file(states).to_crs(CRS_GEO)
    station = gpd.GeoSeries(
        gpd.points_from_xy([STATION_LONLAT[0]], [STATION_LONLAT[1]]), crs=CRS_GEO
    ).to_crs(CRS_UTM)
    return catchment, subs, reaches, states_geo, station


def load_gauges() -> gpd.GeoDataFrame:
    """Telemetric rain gauges that enter the IDW forcing (those with an hourly series)."""
    meta = pd.read_parquet(GAUGE_META)
    meta["codigo"] = meta["codigo"].astype(str).str.zfill(8)
    series = pd.read_parquet(GAUGE_SERIES, columns=["codigo"])
    codes = set(series["codigo"].astype(str).str.zfill(8).unique())
    used = meta[meta["codigo"].isin(codes)].copy()
    return gpd.GeoDataFrame(
        used, geometry=gpd.points_from_xy(used["lon"], used["lat"]), crs=CRS_GEO
    ).to_crs(CRS_UTM)


def line_widths(area: np.ndarray, amin: float, amax: float, lo: float, hi: float) -> np.ndarray:
    """Line width growing with the square root of the upstream area."""
    a = (np.sqrt(np.clip(area, amin, amax)) - np.sqrt(amin)) / (np.sqrt(amax) - np.sqrt(amin))
    return lo + a * (hi - lo)


def state_border(states_geo: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Border between Rio de Janeiro and Minas Gerais, in geographic coordinates."""
    rj = states_geo[states_geo["SIGLA_UF"] == "RJ"].geometry.union_all()
    mg = states_geo[states_geo["SIGLA_UF"] == "MG"].geometry.union_all()
    return gpd.GeoSeries([rj.boundary.intersection(mg.boundary)], crs=CRS_GEO)


def check_dem_crs(crs: CRS | None) -> None:
    """The DEM must be in EPSG:31983.

    The authors' file stores its CRS as an unnamed LOCAL_CS ("SIRGAS 2000 / UTM zone 23S"),
    without an EPSG code; such a file is taken as EPSG:31983. A DEM with another EPSG code is
    rejected, since the catchment polygon is applied to it in EPSG:31983 coordinates.
    """
    epsg = crs.to_epsg() if crs is not None else None
    if epsg is not None and epsg != CRS_UTM:
        raise SystemExit(
            f"The DEM is in EPSG:{epsg}; reproject it to EPSG:{CRS_UTM} (data/gis/README.md)."
        )


def hypsometry(ax: plt.Axes, catchment: gpd.GeoDataFrame, dem: Path) -> tuple:
    """Shaded hypsometry of the DEM inside the catchment; returns the colour norm, map and stats."""
    with rasterio.open(dem) as src:
        check_dem_crs(src.crs)
        out, transform = rio_mask(
            src, [catchment.geometry.iloc[0]], crop=True, filled=True, nodata=np.nan
        )
    elev = out[0].astype("float64")
    valid = np.isfinite(elev) & (elev > 0)
    elev[~valid] = np.nan
    # vmax is the true maximum (not a percentile): the western headwater divide exceeds 2,000 m
    # (Section 2.1) and the colour ramp has to cover it.
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
    ax.imshow(
        rgb, extent=[left, right, bottom, top], origin="upper", interpolation="bilinear", zorder=1
    )
    stats = {"min": float(np.nanmin(elev)), "p1": vmin, "max": vmax}
    return norm, cmap, stats


def north_arrow(ax: plt.Axes, x: float, y: float, size: float, lw: float, fs: float) -> None:
    """North arrow in axes fractions: shaft with a filled head and an 'N' above it."""
    ax.annotate(
        "",
        xy=(x, y + size),
        xytext=(x, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="k", lw=lw, mutation_scale=15),
        zorder=10,
    )
    ax.text(
        x,
        y + size + 0.012,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fs,
        zorder=10,
    )


def km_label(v: float, _: object) -> str:
    return f"{v / 1000:.0f}"


def inset_south_america(axc: plt.Axes, fs: float = 7.0) -> None:
    """Location in South America (Brazil highlighted, dot on the study area)."""
    world = gpd.read_file(NATURAL_EARTH)
    sa = world[world["continent"] == "South America"].to_crs(CRS_GEO)
    br = world[world["name"] == "Brazil"].to_crs(CRS_GEO)
    sa.plot(ax=axc, color="#f2f2f2", edgecolor="#9a9a9a", linewidth=0.3)
    br.plot(ax=axc, color="#cfe0cf", edgecolor="#4d4d4d", linewidth=0.5)
    axc.plot(
        *STATION_LONLAT,
        marker="o",
        color=C_STATION,
        markersize=5.0,
        markeredgecolor="k",
        markeredgewidth=0.5,
        zorder=5,
    )
    # label over the Atlantic, south of the dot, inside the inset frame
    axc.annotate(
        STUDY_AREA,
        xy=STATION_LONLAT,
        xytext=(STATION_LONLAT[0] - 4.0, STATION_LONLAT[1] - 19.0),
        fontsize=fs,
        ha="center",
        va="center",
        linespacing=0.95,
        arrowprops=dict(arrowstyle="-", color="k", lw=0.5, shrinkA=0, shrinkB=3),
    )
    sb = sa.total_bounds
    axc.set_xlim(sb[0] - 1, sb[2] + 1)
    axc.set_ylim(sb[1] - 1, sb[3] + 1)
    axc.set_aspect(1.0 / np.cos(np.radians(20)), adjustable="datalim")
    axc.set_xticks([])
    axc.set_yticks([])
    for s in axc.spines.values():
        s.set_edgecolor("0.35")
        s.set_linewidth(0.7)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Figure 1: location map (public layers: data/gis/README.md)."
    )
    ap.add_argument(
        "--layers-dir",
        type=Path,
        default=None,
        help="folder with dem.tif, BH.gdb and BR_UF_2025/BR_UF_2025.shp "
        "(env FIG1_LAYERS_DIR; default data/gis/external)",
    )
    ap.add_argument("--dem", type=Path, default=None, help="ANADEM v1 in EPSG:31983 (env FIG1_DEM)")
    ap.add_argument(
        "--drainage",
        type=Path,
        default=None,
        help="vector file with the drainage reaches and the field nuareamont (env FIG1_DRAINAGE)",
    )
    ap.add_argument(
        "--drainage-layer",
        default=DRAINAGE_LAYER,
        help=f"layer of --drainage (default {DRAINAGE_LAYER}; empty for the first layer)",
    )
    ap.add_argument(
        "--states",
        type=Path,
        default=None,
        help="IBGE 2025 state boundaries, field SIGLA_UF (env FIG1_STATES)",
    )
    ap.add_argument(
        "--sub-catchments",
        type=Path,
        default=None,
        help="sub-catchment polygons (env FIG1_SUBCATCHMENTS; default data/gis/ottobacias_tc.gpkg)",
    )
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT, help="output folder")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    layers_dir = _resolve(args.layers_dir, "FIG1_LAYERS_DIR", LAYERS_DIR)
    dem = _resolve(args.dem, "FIG1_DEM", layers_dir / DEM_NAME)
    drainage = _resolve(args.drainage, "FIG1_DRAINAGE", layers_dir / DRAINAGE_NAME)
    states = _resolve(args.states, "FIG1_STATES", layers_dir / STATES_NAME)
    subcatchments = _resolve(args.sub_catchments, "FIG1_SUBCATCHMENTS", SUBCATCHMENTS)
    for path, what in (
        (dem, "DEM"),
        (drainage, "Drainage layer"),
        (states, "State boundaries"),
        (subcatchments, "Sub-catchments"),
        (CATCHMENT, "Catchment outline"),
        (NATURAL_EARTH, "Natural Earth countries"),
        (GAUGE_META, "Gauge metadata"),
        (GAUGE_SERIES, "Gauge series"),
    ):
        _require(path, what)
    print(f"DEM:            {_show(dem)}")
    print(f"Drainage:       {_show(drainage)} (layer {args.drainage_layer or 'first'})")
    print(f"States:         {_show(states)}")
    print(f"Sub-catchments: {_show(subcatchments)}")

    catchment, subs, reaches, states_geo, station = load_vectors(
        subcatchments, drainage, args.drainage_layer or None, states
    )
    gauges = load_gauges()
    states_utm = states_geo.to_crs(CRS_UTM)
    border_utm = state_border(states_geo).to_crs(CRS_UTM)

    g_in = gauges[gauges["local"] == "dentro"]  # inside the catchment
    g_out = gauges[gauges["local"] == "buffer"]  # outside, within 20 km of the boundary

    reaches_in = gpd.clip(reaches, catchment.geometry.iloc[0])
    reaches_in = reaches_in[reaches_in["nuareamont"] >= 60]
    lw_rivers = line_widths(reaches_in["nuareamont"].values, 60, 3117.0, 0.30, 1.10)

    # --- page geometry ------------------------------------------------------------------- #
    minx = min(catchment.total_bounds[0], gauges.total_bounds[0])
    miny = min(catchment.total_bounds[1], gauges.total_bounds[1])
    maxx = max(catchment.total_bounds[2], gauges.total_bounds[2])
    maxy = max(catchment.total_bounds[3], gauges.total_bounds[3])
    # side margin of 1.2% of the extent keeps the outermost gauges about 0.7 mm off the frame
    padx = (maxx - minx) * 0.012
    pady = (maxy - miny) * 0.04
    x0, x1 = minx - padx, maxx + padx
    y0, y1 = miny - pady, maxy + pady
    ratio = (x1 - x0) / (y1 - y0)

    # widths and heights in inches, 1:1 with the paper
    tick_fs, tick_len, tick_pad = 7.5, 2.0, 1.0
    fig = plt.figure(figsize=(FIG_W_IN, FIG_W_IN))  # height set below
    # No axis titles (the CRS is in the caption). The left margin is measured from the widest
    # y tick label at 7.5 pt, plus the tick length, the pad and 1 pt of slack.
    ren = fig.canvas.get_renderer()
    tmp = [fig.text(0, 0, km_label(v, None), fontsize=tick_fs) for v in (y0, y1)]
    w_ytick = max(tx.get_window_extent(ren).width for tx in tmp) / fig.dpi
    for tx in tmp:
        tx.remove()
    left = (w_ytick + (tick_len + tick_pad + 1.0) / 72.0) / FIG_W_IN
    width = 1.0 - left - 0.004  # about 0.02 in of right margin (the frame only)
    h_ticks = 0.20  # x tick labels only
    w_map = FIG_W_IN * width
    h_map = w_map / ratio  # equal aspect without expanding the limits
    h_title, h_pad = 0.22, 0.03
    h_legend = 0.78
    fig_h_in = h_title + h_map + h_ticks + h_legend + h_pad
    fig.set_size_inches(FIG_W_IN, fig_h_in)

    ax = fig.add_axes([left, (h_pad + h_legend + h_ticks) / fig_h_in, width, h_map / fig_h_in])
    ax_leg = fig.add_axes([0.02, h_pad / fig_h_in, 0.96, h_legend / fig_h_in])
    ax_leg.axis("off")

    # --- (a) main panel ------------------------------------------------------------------ #
    norm, cmap, elev = hypsometry(ax, catchment, dem)
    # sub-catchment outlines with a halo (white below, thin dark line above), visible over both
    # the green and the brown of the hypsometry
    subs.boundary.plot(ax=ax, color="white", linewidth=0.70, alpha=0.40, zorder=1.9)
    subs.boundary.plot(ax=ax, color="0.15", linewidth=0.28, alpha=0.72, zorder=2.0)
    reaches_in.plot(ax=ax, color=C_RIVER, linewidth=lw_rivers, alpha=0.85, zorder=3)
    border_utm.plot(ax=ax, color="white", linewidth=1.3, linestyle=(0, (6, 3)), zorder=3.4)
    border_utm.plot(ax=ax, color=C_BORDER, linewidth=0.65, linestyle=(0, (6, 3)), zorder=3.5)
    catchment.boundary.plot(ax=ax, color=C_OUTLINE, linewidth=1.2, zorder=4)

    g_out.plot(
        ax=ax,
        marker="^",
        color=C_GAUGE_OUT,
        edgecolor="white",
        markersize=24,
        linewidth=0.45,
        zorder=5.5,
    )
    # gauges inside the catchment are drawn over the station star: one of them lies next to it
    g_in.plot(
        ax=ax,
        marker="^",
        color=C_GAUGE_IN,
        edgecolor="white",
        markersize=36,
        linewidth=0.5,
        zorder=8,
    )
    station.plot(
        ax=ax, marker="*", color=C_STATION, edgecolor="k", markersize=150, linewidth=0.6, zorder=7
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(km_label))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(km_label))
    ax.tick_params(labelsize=tick_fs, length=tick_len, pad=tick_pad)
    ax.set_title(TITLE_A, loc="left", fontsize=8.5, fontweight="bold", pad=3)

    # state codes on each part of the catchment
    for code in ("MG", "RJ"):
        state = states_utm[states_utm["SIGLA_UF"] == code]
        part = gpd.overlay(catchment, state, how="intersection")
        if len(part):
            pt = part.geometry.union_all().representative_point()
            ax.text(
                pt.x,
                pt.y,
                code,
                fontsize=9,
                fontweight="bold",
                color="0.12",
                ha="center",
                va="center",
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.5", alpha=0.72),
            )

    # colour bar in the empty lower right corner; the north arrow above it, clear of "2500"
    sm = ScalarMappable(norm=norm, cmap=cmap)
    north_arrow(ax, x=0.972, y=0.290, size=0.110, lw=1.4, fs=9)
    cax = ax.inset_axes(list(RECT_CB))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(CBAR_LABEL, fontsize=7.5, labelpad=2)
    cb.set_ticks([500, 1500, 2500])  # the ramp spans the 1st percentile to the maximum
    cb.ax.tick_params(labelsize=7.5, length=2, pad=1.2)
    cb.outline.set_linewidth(0.5)

    # --- (b) inset ----------------------------------------------------------------------- #
    axc = ax.inset_axes(list(RECT_SA))
    inset_south_america(axc, fs=7.5)
    ax.text(
        RECT_SA[0],
        RECT_SA[1] + RECT_SA[3] + 0.012,
        TITLE_B,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=9,
    )

    # --- legend (two columns, filled column by column) ----------------------------------- #
    handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=C_STATION,
            markeredgecolor="k",
            markersize=11,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=C_GAUGE_IN,
            markeredgecolor="white",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=C_GAUGE_OUT,
            markeredgecolor="white",
            markersize=6,
        ),
        Line2D([0], [0], color=C_OUTLINE, lw=1.4),
        Line2D([0], [0], color=C_BORDER, lw=0.9, linestyle=(0, (4, 2))),
        Line2D([0], [0], color="0.15", lw=0.8),
        Line2D([0], [0], color=C_RIVER, lw=1.4),
    ]
    labels = [
        LEG_STATION,
        LEG_GAUGE_IN.format(n=len(g_in)),
        LEG_GAUGE_OUT.format(n=len(g_out)),
        LEG_OUTLINE,
        LEG_BORDER,
        LEG_SUBCATCHMENTS,
        LEG_DRAINAGE,
    ]
    # tight spacing so that the legend fits the 4.79 in width without widening the bounding
    # box (otherwise LaTeX would shrink the whole figure to fit \linewidth)
    ax_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        fontsize=7.5,
        framealpha=1.0,
        edgecolor="0.6",
        borderpad=0.5,
        handletextpad=0.4,
        columnspacing=1.5,
        handlelength=2.0,
        labelspacing=0.45,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out_png = args.out / OUT_NAME
    # tight bounding box with a minimal pad (0.01 in), so that the final width equals \linewidth
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)

    area_km2 = float(catchment.area.iloc[0]) / 1e6
    print(f"Sub-catchments: {len(subs)}")
    print(f"Catchment area (outline, EPSG:{CRS_UTM}): {area_km2!r} km2 ({area_km2:.2f})")
    print(
        f"Telemetric rain gauges of the IDW forcing: {len(gauges)} "
        f"({len(g_in)} inside the catchment, {len(g_out)} outside)"
    )
    print(f"Drainage reaches drawn (upstream area >= 60 km2): {len(reaches_in)}")
    print(
        "Elevation inside the catchment (DEM): "
        f"minimum {elev['min']!r} m ({elev['min']:.2f}); "
        f"1st percentile {elev['p1']!r} m ({elev['p1']:.2f}), start of the colour ramp; "
        f"maximum {elev['max']!r} m ({elev['max']:.2f}), end of the colour ramp"
    )
    print(f"Figure {FIG_W_IN:.2f} x {fig_h_in:.2f} in at {FIG_DPI} dpi")
    print(f"Saved: {_show(out_png)}")


if __name__ == "__main__":
    main()
