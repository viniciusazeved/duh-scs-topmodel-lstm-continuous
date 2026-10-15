"""Figure 2: rainfall and observed streamflow at Manuel Duarte (ANA 58585000), 2021 to 2025.

Top panel (inverted): daily rainfall, area-weighted mean of the 245 sub-catchments of the
telemetric forcing used by every model. Bottom panel: observed hourly streamflow, with the hours
without streamflow in red; the models drop the windows that contain them instead of imputing.
Both panels shade the train, validation and test splits, and the labels above the top panel give
the share of missing hourly streamflow in each split.

Usage, from the repository root:
    uv run python scripts/figures/fig_serie_vazao.py [--h5 PATH] [--out DIR]

Output: outputs/figures/fig_serie_vazao.png (300 dpi).

Adapted from the generator of the submitted figure (SHA-256 79e16ca862a304f6...): only the input
and output paths changed and the code was wrapped in functions; data handling, colours and
layout are unchanged.
"""

from __future__ import annotations

import os

# Non-interactive backend: the script only writes the PNG.
os.environ["MPLBACKEND"] = "Agg"

import argparse
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
H5_DEFAULT = ROOT / "data" / "dataset_58585000_telem.h5"
OUT_DEFAULT = ROOT / "outputs" / "figures"
OUT_NAME = "fig_serie_vazao.png"
SPLITS = ("train", "val", "test")

# Font sizes for print: the figure enters with width=\linewidth (about 16 cm), a scale of about
# 0.9, so the 10 pt base prints at about 9 pt and nothing falls below about 7.5 pt.
FS_BASE, FS_LABEL, FS_TICK, FS_SPLIT, FS_LEGEND, FS_NOTE = 10, 10.5, 9.5, 9.5, 9, 8.5
RC = {
    "font.size": FS_BASE,
    "font.family": "DejaVu Sans",
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}
SPLIT_C = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
SPLIT_LB = {"train": "Train", "val": "Validation", "test": "Test"}
NOTE = (
    "Area-weighted mean of the 245 sub-catchments\n(the model is forced by the distributed field)"
)


def load(h5_path: Path) -> dict:
    """Train, validation and test concatenated, with the index range of each split."""
    with h5py.File(h5_path, "r") as f:
        area = f["ottobacia/area_km2"][:].astype(np.float64)
        p_all, q_all, ts_all, bounds, n = [], [], [], [], 0
        for s in SPLITS:
            p = f[f"{s}/precipitation"][:]
            q = f[f"{s}/streamflow"][:]
            ts = f[f"{s}/timestamps"][:].astype("int64")
            p_all.append(p)
            q_all.append(q)
            ts_all.append(ts)
            bounds.append((s, n, n + len(q)))
            n += len(q)
    precip = np.concatenate(p_all)
    flow = np.concatenate(q_all)
    t = pd.to_datetime(np.concatenate(ts_all), unit="s", utc=True).tz_convert(None)  # naive UTC
    w = area / area.sum()
    rain_h = pd.Series(precip @ w, index=t)  # catchment-mean rainfall (mm/h)
    rain_d = rain_h.resample("D").sum(min_count=1)  # mm/day
    nan_pct = {s: 100 * np.isnan(flow[i:j]).mean() for s, i, j in bounds}
    return {"t": t, "Q": flow, "rain_d": rain_d, "bounds": bounds, "nan_pct": nan_pct}


def make_figure(d: dict) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Hyetograph (inverted, top) and hydrograph (bottom) with a shared time axis."""
    t, flow, rain_d, bounds, nan_pct = d["t"], d["Q"], d["rain_d"], d["bounds"], d["nan_pct"]
    # 7.0 x 4.8 in: close to \linewidth, tall enough for the font sizes above
    fig, (axr, axq) = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 2.6], "hspace": 0.07},
    )

    # split bands (both panels), label with the share of gaps, divider lines
    for s, i, j in bounds:
        for ax in (axr, axq):
            ax.axvspan(t[i], t[min(j, len(t) - 1)], color=SPLIT_C[s], alpha=0.07, lw=0, zorder=0)
        xc = t[i] + (t[min(j, len(t) - 1)] - t[i]) / 2
        axr.text(
            xc,
            1.05,
            f"{SPLIT_LB[s]}\n{nan_pct[s]:.1f}% gaps",
            transform=axr.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=FS_SPLIT,
            color=SPLIT_C[s],
            weight="bold",
            linespacing=1.15,
        )
        if s != "train":
            for ax in (axr, axq):
                ax.axvline(t[i], color="0.4", ls=":", lw=1.1, zorder=1)

    # hyetograph, inverted (rain falls from the top)
    axr.bar(rain_d.index, rain_d.values, width=1.0, color="#3A6EA5", lw=0, zorder=2)
    axr.set_ylabel("Rainfall\n(mm d$^{-1}$)")
    # axis limit about 15% above the highest bar, so that the note fits below the bars
    axr.set_ylim(np.nanmax(rain_d.values) * 1.15, 0)
    axr.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
    axr.tick_params(labelbottom=False)
    # the panel shows the areal mean; the models are forced by the field of 245 sub-catchments
    axr.text(
        0.993,
        0.06,
        NOTE,
        transform=axr.transAxes,
        ha="right",
        va="bottom",
        fontsize=FS_NOTE,
        color="0.3",
        linespacing=1.15,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9),
    )

    # observed hourly hydrograph
    axq.plot(t, flow, color="#23404F", lw=0.6, zorder=3)
    axq.set_ylabel("Observed discharge\n(m$^3$ s$^{-1}$)")
    axq.set_xlabel("Date")
    axq.grid(ls=":", lw=0.5, alpha=0.5)
    axq.set_ylim(bottom=0)

    # streamflow gaps
    nan_mask = np.isnan(flow)
    ymax = np.nanmax(flow) * 1.05
    axq.fill_between(
        t,
        0,
        ymax,
        where=nan_mask,
        color="#C44E52",
        alpha=0.22,
        lw=0,
        zorder=2,
        label="Missing discharge (window dropped, not imputed)",
    )
    axq.set_ylim(0, ymax)
    axq.legend(loc="upper right", fontsize=FS_LEGEND, frameon=True, framealpha=0.9)

    axq.xaxis.set_major_locator(mdates.YearLocator())
    axq.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))
    axq.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axq.set_xlim(t[0], t[-1])

    # no suptitle: the description goes in the caption
    fig.subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.10)
    return fig, axr, axq


def _show(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Figure 2: rainfall and observed streamflow.")
    ap.add_argument("--h5", type=Path, default=H5_DEFAULT, help="telemetric dataset (h5)")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT, help="output folder")
    args = ap.parse_args(argv)

    mpl.rcParams.update(RC)
    d = load(args.h5)
    fig, _, _ = make_figure(d)
    args.out.mkdir(parents=True, exist_ok=True)
    out_png = args.out / OUT_NAME
    fig.savefig(out_png)
    plt.close(fig)

    t, flow, rain_d = d["t"], d["Q"], d["rain_d"]
    print(f"Input: {_show(args.h5)}")
    print(f"Hourly timestamps: {len(t)}, {t[0]:%Y-%m-%d %H:%M} to {t[-1]:%Y-%m-%d %H:%M} UTC")
    for s, i, j in d["bounds"]:
        pct = float(d["nan_pct"][s])
        print(
            f"  {s:5s}: {j - i} h, {t[i]:%Y-%m-%d %H:%M} to {t[j - 1]:%Y-%m-%d %H:%M} UTC; "
            f"missing streamflow {pct!r}% ({pct:.2f}%)"
        )
    miss = 100 * float(np.isnan(flow).mean())
    print(f"Missing streamflow, whole record: {miss!r}% ({miss:.2f}%); available {100 - miss:.2f}%")
    qmax, qmean = float(np.nanmax(flow)), float(np.nanmean(flow))
    t_qmax = t[int(np.nanargmax(flow))]
    print(
        f"Streamflow maximum {qmax!r} m3/s ({qmax:.2f}) at {t_qmax:%Y-%m-%d %H:%M} UTC; "
        f"mean {qmean!r} m3/s ({qmean:.2f})"
    )
    rmax = float(np.nanmax(rain_d.values))
    print(
        f"Largest daily areal-mean rainfall {rmax!r} mm ({rmax:.2f}) on "
        f"{rain_d.idxmax():%Y-%m-%d}; rainfall axis limit {rmax * 1.15:.2f} mm/day"
    )
    print(f"Saved: {_show(out_png)}")


if __name__ == "__main__":
    main()
