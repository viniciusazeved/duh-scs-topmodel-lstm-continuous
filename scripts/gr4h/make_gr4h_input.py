#!/usr/bin/env python
"""Rebuild the hourly input series of the GR4H benchmark from the canonical dataset.

The GR4H benchmark of the manuscript (Section 4.6; Supplementary Information Section S6) is driven
by catchment-mean hourly series, versioned in ``data/gr4h/serie_horaria_gr4h.csv``: 43,801 rows,
the train, validation and test splits of ``data/dataset_58585000_telem.h5`` in sequence, from
1 January 2021 00:00 to 31 December 2025 00:00 UTC. Its columns are

- ``Dates``: hourly timestamp (UTC);
- ``P``: precipitation (mm/h), area-weighted mean over the 245 sub-catchments;
- ``E``: potential evapotranspiration (mm/h), area-weighted mean of the hourly PET that feeds the
  TOPMODEL layer (daily Hargreaves-Samani PET divided by 24);
- ``Qmm``: observed discharge at the outlet (mm/h), ``Q * 3.6 / 3117`` with ``Q`` in m3/s and the
  3,117 km2 catchment area of the GR4H inputs (the sub-catchment areas of the dataset add up to
  3,117.13 km2); empty where the discharge is missing.

The script rebuilds the four columns from the dataset and checks them against the versioned CSV:
same timestamps, same hours of missing discharge and a maximum absolute difference of at most
``--tol`` (default 2e-6 mm/h) in P, E and Qmm. The arithmetic follows the original export, in
float32 as the dataset stores it: weights ``area / sum(area)``, weighted sums over the
sub-catchments, discharge multiplied by 3.6 and divided by 3117. In the environment of the runs
(numpy 2.4.1, pandas 3.0.0) the rebuilt file is byte-identical to the versioned one, which the
script also reports; another summation order only moves the sixth decimal, within the tolerance.
The CSV keeps the format of the versioned file: six decimals and the line endings of the local
copy of the versioned CSV (the original export has Windows line endings, CRLF; git may check the
file out with LF).

Reads ``data/dataset_58585000_telem.h5`` and ``data/gr4h/serie_horaria_gr4h.csv`` and writes
nothing unless ``--out`` is given; the versioned CSV itself is never overwritten.

Usage, from the repository root:
    python scripts/gr4h/make_gr4h_input.py
    python scripts/gr4h/make_gr4h_input.py --out outputs/retrain/gr4h/serie_horaria_gr4h.csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H5 = ROOT / "data" / "dataset_58585000_telem.h5"
DEFAULT_REFERENCE = ROOT / "data" / "gr4h" / "serie_horaria_gr4h.csv"

SPLITS = ("train", "val", "test")  # chronological order of the dataset splits
AREA_KM2 = 3117.0  # catchment area of the GR4H inputs (km2); a Python float keeps float32 math
COLUMNS = ("P", "E", "Qmm")
FLOAT_FORMAT = "%.6f"  # six decimals, as in the versioned CSV


def display_path(path: Path) -> str:
    """Path relative to the repository root when possible."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_series(h5_path: Path) -> pd.DataFrame:
    """Catchment-mean hourly P, E and Qmm of the whole record, as float32."""
    with h5py.File(h5_path, "r") as f:
        area = f["ottobacia/area_km2"][:]
        precip = np.concatenate([f[f"{s}/precipitation"][:] for s in SPLITS])
        pet = np.concatenate([f[f"{s}/pet"][:] for s in SPLITS])
        flow = np.concatenate([f[f"{s}/streamflow"][:] for s in SPLITS]).reshape(-1)
        seconds = np.concatenate([f[f"{s}/timestamps"][:] for s in SPLITS])
    for name, array in (("precipitation", precip), ("pet", pet)):
        if array.dtype != np.float32:
            raise TypeError(f"{name} is {array.dtype}, expected float32")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} has non-finite values; GR4H needs gap-free forcing")
    time = pd.to_datetime(seconds, unit="s", utc=True)
    if not time.is_monotonic_increasing or time.has_duplicates:
        raise ValueError("timestamps of the dataset are not strictly increasing")
    if not (np.diff(seconds) == 3600).all():
        raise ValueError("timestamps of the dataset are not hourly and contiguous")
    weights = area / area.sum()  # float32
    return pd.DataFrame(
        {
            "Dates": time,
            "P": (precip * weights).sum(axis=1),
            "E": (pet * weights).sum(axis=1),
            "Qmm": flow * 3.6 / AREA_KM2,  # float32; NaN where the discharge is missing
        }
    )


def to_csv_bytes(series: pd.DataFrame, line_terminator: str) -> bytes:
    """The series in the format of the versioned CSV."""
    text = series.to_csv(index=False, float_format=FLOAT_FORMAT, lineterminator=line_terminator)
    return text.encode("ascii")


def sha256(data: bytes) -> str:
    """SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


def compare(series: pd.DataFrame, reference_path: Path, tol: float) -> dict[str, float]:
    """Check the rebuilt series against the versioned CSV and return max |difference| by column."""
    reference = pd.read_csv(reference_path)
    if list(reference.columns) != ["Dates", *COLUMNS]:
        raise AssertionError(f"unexpected columns in the reference: {list(reference.columns)}")
    if len(reference) != len(series):
        raise AssertionError(f"rows: rebuilt {len(series)}, reference {len(reference)}")
    reference_time = pd.to_datetime(reference["Dates"], utc=True)
    if not (reference_time.to_numpy() == series["Dates"].to_numpy()).all():
        raise AssertionError("timestamps differ from the reference")
    max_diff = {}
    for column in COLUMNS:
        rebuilt = series[column].to_numpy(dtype=np.float64)
        versioned = reference[column].to_numpy(dtype=np.float64)
        if not (np.isnan(rebuilt) == np.isnan(versioned)).all():
            raise AssertionError(f"{column}: missing values differ from the reference")
        finite = np.isfinite(versioned)
        max_diff[column] = float(np.max(np.abs(rebuilt[finite] - versioned[finite])))
        if max_diff[column] > tol:
            raise AssertionError(f"{column}: max |difference| {max_diff[column]:.3e} > {tol:.1e}")
    return max_diff


def count_text_differences(rebuilt: bytes, reference: bytes) -> int:
    """Number of CSV lines whose text differs, whatever the line endings."""
    new_lines, old_lines = rebuilt.splitlines(), reference.splitlines()
    unmatched = abs(len(new_lines) - len(old_lines))
    return unmatched + sum(a != b for a, b in zip(new_lines, old_lines, strict=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5, help="canonical dataset")
    parser.add_argument(
        "--reference", type=Path, default=DEFAULT_REFERENCE, help="versioned GR4H input CSV"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="write the rebuilt CSV to this path (optional)"
    )
    parser.add_argument(
        "--tol", type=float, default=2e-6, help="max absolute difference allowed (mm/h)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out is not None and args.out.resolve() == args.reference.resolve():
        raise SystemExit("--out must not be the versioned reference CSV; choose another path")

    series = build_series(args.h5)
    reference = args.reference.read_bytes()
    line_terminator = "\r\n" if b"\r\n" in reference else "\n"
    rebuilt = to_csv_bytes(series, line_terminator)
    time = series["Dates"]
    print(f"dataset:   {display_path(args.h5)}")
    print(f"reference: {display_path(args.reference)}")
    print(
        f"rows: {len(series):,} hourly, {time.iloc[0]:%Y-%m-%d %H:%M} to "
        f"{time.iloc[-1]:%Y-%m-%d %H:%M} UTC; missing discharge: "
        f"{int(series['Qmm'].isna().sum()):,} h"
    )

    max_diff = compare(series, args.reference, args.tol)
    for column in COLUMNS:
        print(f"max |rebuilt - versioned| {column:>3}: {max_diff[column]!r} mm/h")
    print(f"check passed: every column within {args.tol:.1e} mm/h of the versioned CSV")
    endings = "CRLF" if line_terminator == "\r\n" else "LF"
    print(
        f"byte-identical to the versioned CSV ({endings} line endings, as in the local copy): "
        f"{rebuilt == reference}; lines with different text: "
        f"{count_text_differences(rebuilt, reference)}"
    )
    print(f"SHA-256 rebuilt:   {sha256(rebuilt)}")
    print(f"SHA-256 versioned: {sha256(reference)}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(rebuilt)
        print(f"written: {display_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
