"""Figure S2: rainfall fields of the five products at the peak hour of the 1 November 2021 event.

Choropleth maps of the 245 sub-catchments for the telemetric (IDW), MERGE, ERA5-Land, IMERG and
MSWEP fields at 13:00 UTC on 1 November 2021, the rainfall that produced the largest flood of the
record, and the catchment-mean hyetograph of that day. The five maps share a linear colour scale
capped at the 96th percentile of the five peak-hour fields, so the telemetric field saturates
while the blocks of the gridded products stay visible. The value printed in each map is the
area-weighted catchment mean at the peak hour; the triangle marks the outlet.

The event is fixed (EVENT_DAY, PEAK_HOUR_UTC) as in the caption of Figure S2; the script checks
that 13:00 UTC is the hour of the largest catchment-mean telemetric rainfall of that day.

MSWEP V3 (GloH2O) is licensed CC BY-NC 4.0 with terms that restrict redistribution, so its
processed file is not part of this package (scripts/data_prep/products/ documents how it was
built from the GloH2O files). The MSWEP map and hyetograph curve are drawn only when that file
exists: pass --mswep PATH or set the environment variable MSWEP_H5; the default location is
data/dataset_58585000_mswep.h5. Without it the MSWEP map is replaced by a note, the MSWEP curve
is left out of the hyetograph, and the other panels are those of the submitted figure (the colour
scale keeps the cap computed with the five products, VMAX_FIVE_PRODUCTS).

Usage, from the repository root and with the optional "gis" dependencies:
    uv run python scripts/figures/fig_produtos_evento.py [--mswep PATH] [--out DIR]

Output: outputs/figures/fig_produtos_evento.png (300 dpi).

Adapted from the generator of the submitted figure (SHA-256 b36c489814148af6...): fixed event
instead of the day of largest catchment-mean rainfall, relative paths, optional MSWEP, PNG only
(the manuscript uses the PNG); data handling, colours and layout are unchanged.
"""

from __future__ import annotations

import os

# Non-interactive backend: the script only writes the PNG.
os.environ["MPLBACKEND"] = "Agg"

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
GPKG = ROOT / "data" / "gis" / "ottobacias_tc.gpkg"
MSWEP_DEFAULT = DATA / "dataset_58585000_mswep.h5"
MSWEP_ENV = "MSWEP_H5"
OUT_DEFAULT = ROOT / "outputs" / "figures"
OUT_NAME = "fig_produtos_evento.png"
SPLITS = ("train", "val", "test")

EVENT_DAY = pd.Timestamp("2021-11-01", tz="UTC")
PEAK_HOUR_UTC = 13
# Upper end of the colour scale in the submitted figure: 96th percentile of the five peak-hour
# fields, MSWEP included. With the five files the script recomputes it (and reports any
# difference); without MSWEP it keeps this value, so the other four maps match the submitted
# figure.
VMAX_FIVE_PRODUCTS = 18.28605079650879

# key, label, colour (Okabe-Ito, safe for colour blindness), hyetograph marker
PRODUCTS = [
    ("telem", "Telemetric (IDW)", "#0072B2", "o"),
    ("merge", "MERGE", "#E69F00", "s"),
    ("era5land", "ERA5-Land", "#009E73", "^"),
    ("imerg", "IMERG", "#D55E00", "D"),
    ("mswep", "MSWEP", "#CC79A7", "v"),
]

# Font sizes for print: the PNG (about 6.7 in wide with the pad) enters the SI at 16 cm, a scale
# of about 0.94, so 9 pt here prints at about 8.5 pt.
FS_TICK, FS_LABEL, FS_TITLE = 9.0, 9.5, 10.0


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precipitation (T, 245), timestamps (s, UTC) and sub-catchment areas of one product."""
    with h5py.File(path, "r") as f:
        precip = np.concatenate([f[f"{s}/precipitation"][:] for s in SPLITS])
        ts = np.concatenate([f[f"{s}/timestamps"][:] for s in SPLITS]).astype("int64")
        area = f["ottobacia/area_km2"][:].astype(np.float64)
    return precip, ts, area


def _show(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Figure S2: rainfall fields of the five products.")
    ap.add_argument("--data", type=Path, default=DATA, help="folder of the product h5 files")
    ap.add_argument("--gpkg", type=Path, default=GPKG, help="sub-catchment polygons")
    ap.add_argument(
        "--mswep",
        type=Path,
        default=None,
        help=f"MSWEP h5 (env {MSWEP_ENV}; default data/dataset_58585000_mswep.h5)",
    )
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT, help="output folder")
    args = ap.parse_args(argv)

    mpl.rcParams.update({"font.size": FS_TICK, "font.family": "DejaVu Sans"})
    paths = {key: args.data / f"dataset_58585000_{key}.h5" for key, *_ in PRODUCTS}
    env_mswep = os.environ.get(MSWEP_ENV)
    paths["mswep"] = args.mswep or (Path(env_mswep) if env_mswep else MSWEP_DEFAULT)
    products = [row for row in PRODUCTS if row[0] != "mswep" or paths["mswep"].exists()]
    for key, *_ in products:
        if not paths[key].exists():
            raise SystemExit(f"Missing input: {paths[key]}")
    if len(products) < len(PRODUCTS):
        print(
            f"MSWEP file not found ({_show(paths['mswep'])}): the MSWEP map and curve are "
            "omitted; the other panels keep the colour scale of the submitted figure."
        )

    # 1. the event day of each product; weights from the telemetric file
    data = {key: load(paths[key]) for key, *_ in products}
    _, _, area = data["telem"]
    w = area / area.sum()
    day_end = EVENT_DAY + pd.Timedelta(days=1)
    peak_time = EVENT_DAY + pd.Timedelta(hours=PEAK_HOUR_UTC)

    # 2. field at the peak hour (resolution test: the grid blocks stand out) and hyetograph
    fields, hyeto = {}, {}
    for key, *_ in products:
        precip, ts, _ = data[key]
        ti = pd.to_datetime(ts, unit="s", utc=True)
        in_day = (ti >= EVENT_DAY) & (ti < day_end)
        hours = ti[in_day]
        if len(hours) != 24 or hours[PEAK_HOUR_UTC] != peak_time:
            raise SystemExit(f"{key}: the event day is not a complete hourly day in the file")
        fields[key] = precip[in_day][PEAK_HOUR_UTC]  # (245,) mm/h at the peak hour
        hyeto[key] = precip[in_day] @ w  # (24,) catchment-mean mm/h
    hpeak = PEAK_HOUR_UTC
    # linear cap: the telemetric field saturates, but the blocks of the gridded products show
    vmax_shown = float(np.nanpercentile(np.concatenate([fields[key] for key, *_ in products]), 96))
    all_five = len(products) == len(PRODUCTS)
    vmax = vmax_shown if all_five else VMAX_FIVE_PRODUCTS

    # 3. geometry (lon/lat) and outlet marker: centroid of the sub-catchment with the shortest
    # concentration time, the one that drains to the gauge
    g = gpd.read_file(args.gpkg).to_crs(4326)
    assert len(g) == 245
    with h5py.File(paths["telem"], "r") as f:
        tc_base = f["ottobacia/tc_base_h"][:]
    with warnings.catch_warnings():
        # centroid computed in lon/lat; it only places the marker
        warnings.simplefilter("ignore", UserWarning)
        outlet = g.geometry.centroid.iloc[int(np.argmin(tc_base))]

    # 4. layout in inches: 3 rows x 2 columns; rows 1 and 2 hold four maps; row 3 holds the
    # MSWEP map with the colour bar below it (left) and the hyetograph at full height (right)
    b = g.total_bounds
    asp = 1.0 / np.cos(np.radians((b[1] + b[3]) / 2))  # the factor geopandas uses in lon/lat
    xlim = (b[0] - 0.02 * (b[2] - b[0]), b[2] + 0.02 * (b[2] - b[0]))
    ylim = (b[1] - 0.04 * (b[3] - b[1]), b[3] + 0.04 * (b[3] - b[1]))
    map_w = 2.7  # width of each map (in)
    map_h = map_w * (ylim[1] - ylim[0]) * asp / (xlim[1] - xlim[0])  # box matches the aspect
    x_l, gap = 0.55, 0.55  # margin for the latitude labels; gap that holds the hyetograph y axis
    x_r = x_l + map_w + gap
    top, t_row2, t_row3, cb_drop, cb_h, bottom = 0.30, 0.32, 0.55, 0.58, 0.13, 0.48
    fig_h = top + 3 * map_h + t_row2 + t_row3 + cb_drop + bottom
    fig_w = x_r + map_w + 0.10
    fig = plt.figure(figsize=(fig_w, fig_h))

    def ax_in(x0: float, y0: float, w_: float, h_: float) -> plt.Axes:
        """Axes placed in inches (origin at the lower left corner)."""
        return fig.add_axes([x0 / fig_w, y0 / fig_h, w_ / fig_w, h_ / fig_h])

    y1 = fig_h - top - map_h
    y2 = y1 - t_row2 - map_h
    y3 = y2 - t_row3 - map_h
    y_cb = y3 - cb_drop
    slots = [(x_l, y1), (x_r, y1), (x_l, y2), (x_r, y2), (x_l, y3)]

    cmap = "YlGnBu"
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)  # linear: shows the peak underestimation
    fmt_lon = FuncFormatter(lambda x, _: f"{abs(x):.1f}°W")
    fmt_lat = FuncFormatter(lambda y, _: f"{abs(y):.1f}°S")
    for k, (key, label, _, _) in enumerate(products):
        ax = ax_in(slots[k][0], slots[k][1], map_w, map_h)
        g2 = g.copy()
        g2["mm"] = fields[key]
        g2.plot(column="mm", cmap=cmap, norm=norm, ax=ax, edgecolor="0.55", linewidth=0.25)
        ax.set_aspect(asp)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(label, fontsize=FS_TITLE, weight="bold", pad=4)
        # catchment mean at the peak hour, in the empty upper left corner of the panel
        ax.text(
            0.02,
            0.95,
            f"mean {hyeto[key][hpeak]:.1f} mm/h",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FS_TICK,
        )
        # 0.4 degree steps keep the last longitude label clear of the hyetograph
        ax.xaxis.set_major_locator(MultipleLocator(0.4))
        ax.xaxis.set_major_formatter(fmt_lon)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(fmt_lat)
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
        # latitude labels on the left column only; longitude labels on the lowest maps only
        ax.tick_params(labelsize=FS_TICK, length=2.5, labelleft=(k % 2 == 0), labelbottom=(k >= 3))
        ax.plot(outlet.x, outlet.y, marker="v", ms=7, color="red", mec="black", mew=0.6, zorder=6)
    if not all_five:
        ax = ax_in(slots[4][0], slots[4][1], map_w, map_h)
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "MSWEP not shown:\nits file is not redistributed\n(see the README)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FS_TICK,
            color="0.4",
        )

    # horizontal colour bar below the MSWEP map (same scale for all maps)
    cax = ax_in(x_l + 0.15, y_cb, map_w - 0.30, cb_h)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("precipitation at peak hour (mm/h)", fontsize=FS_LABEL)
    cb.ax.xaxis.set_major_locator(MultipleLocator(3))
    cb.ax.tick_params(labelsize=FS_TICK, length=2.5)

    # hyetograph: full height of row 3 (map plus colour bar), right column
    axh = ax_in(x_r, y_cb, map_w, y3 + map_h - y_cb)
    for key, label, color, marker in products:
        axh.plot(range(24), hyeto[key], marker=marker, ms=3, lw=1.4, color=color, label=label)
    ymax = max(float(np.max(hyeto[key])) for key, *_ in products)
    axh.set_xlim(0, 23)
    axh.set_ylim(0, 1.45 * ymax)  # headroom so that the legend does not cover the curves
    axh.xaxis.set_major_locator(MultipleLocator(3))
    axh.set_xlabel("hour of day (UTC)", fontsize=FS_LABEL)
    axh.set_ylabel("precipitation (mm/h)", fontsize=FS_LABEL)
    axh.tick_params(labelsize=FS_TICK, length=2.5)
    axh.axvline(hpeak, color="0.35", ls="--", lw=1.0)
    axh.text(
        hpeak + 0.4,
        1.40 * ymax,
        f"{hpeak:02d}:00 UTC",
        ha="left",
        va="top",
        fontsize=FS_TICK,
        color="0.35",
    )
    # short title to fit the panel width; "catchment-mean" is in the caption
    axh.set_title(
        f"Mean hyetograph, {EVENT_DAY.day} {EVENT_DAY:%b %Y}",
        fontsize=FS_TITLE,
        weight="bold",
        pad=4,
    )
    axh.legend(
        fontsize=FS_TICK,
        frameon=False,
        loc="upper left",
        borderaxespad=0.3,
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.25,
    )
    axh.grid(alpha=0.3)
    # no suptitle: the description goes in the caption; tight bounding box so that nothing on
    # the right edge is cut
    args.out.mkdir(parents=True, exist_ok=True)
    out_png = args.out / OUT_NAME
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    hour_max = int(np.argmax(hyeto["telem"]))
    day_total = float(np.sum(hyeto["telem"]))
    print("Inputs: " + ", ".join(f"{key} {_show(paths[key])}" for key, *_ in products))
    print(
        f"Event {EVENT_DAY:%Y-%m-%d}, peak hour {hpeak:02d}:00 UTC (fixed); hour of the largest "
        f"catchment-mean telemetric rainfall that day: {hour_max:02d}:00 UTC"
        + ("" if hour_max == hpeak else "  WARNING: differs from the fixed peak hour")
    )
    print(f"Catchment-mean telemetric rainfall that day: {day_total!r} mm ({day_total:.2f})")
    for key, label, *_ in products:
        v = float(hyeto[key][hpeak])
        print(f"  {label:17s} catchment mean at the peak hour: {v!r} mm/h ({v:.2f})")
    if all_five:
        print(f"Colour scale 0 to {vmax!r} mm/h ({vmax:.2f}), 96th percentile of the five fields")
        if vmax != VMAX_FIVE_PRODUCTS:
            print(f"  NOTE: the submitted figure used {VMAX_FIVE_PRODUCTS!r} mm/h")
    else:
        print(
            f"Colour scale 0 to {vmax!r} mm/h ({vmax:.2f}), as in the submitted figure; the four "
            f"fields shown alone would give {vmax_shown!r} mm/h ({vmax_shown:.2f})"
        )
    print(f"Figure {fig_w:.2f} x {fig_h:.2f} in; saved: {_show(out_png)}")


if __name__ == "__main__":
    main()
