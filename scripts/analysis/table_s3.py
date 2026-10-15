#!/usr/bin/env python
"""Reproduce Table S3 of the Supplementary Information: KGE decomposition of the attribution chain.

Prints, for the 13 rungs of the attribution chain (order of Table 1 of the manuscript), the mean
across seeds of the Kling-Gupta efficiency and of its three components over the test period
(6,323 hourly time steps, telemetric forcing): correlation r, variability ratio
alpha = sd_sim / sd_obs and bias ratio beta = mean_sim / mean_obs. Rungs 1 to 12 have 10 seeds and
rung 13 (Phys_DUH_Topmodel) two, 42 and 43. 13 x 4 = 52 cells; the KGE column is the KGE column
of Table 3.

Sources, as in the computation behind the manuscript (``scripts/consolida_sim.py``):

- KGE: ``test_metrics.kge`` of each ``results.json``, written by ``scripts/train.py``.
- r, alpha and beta: recomputed from ``predictions.npz`` (simulated and observed discharge in
  m3/s, float32, used as stored) over the finite pairs: Pearson correlation, ratio of the
  population standard deviations and ratio of the means (both ratios with 1e-9 added to the
  denominator).

Rounded values are rounded half away from zero to two decimals directly from the unrounded value,
never from an already rounded number. For example, beta of Phys_DUH_SCS is 0.69484 and rounds to
0.69, whereas rounding a three-decimal 0.695 would give 0.70. Each rounded cell is compared with
Table S3 of the submitted Supplementary Information and any difference is reported.

Reads ``outputs/grade_sim`` (``results.json`` and ``predictions.npz``) and writes nothing unless
``--csv`` is given.

Usage, from the repository root:
    python scripts/analysis/table_s3.py
    python scripts/analysis/table_s3.py --cells    # every cell: unrounded, rounded, manuscript
    python scripts/analysis/table_s3.py --csv      # also write outputs/tables/table_s3.csv
    python scripts/analysis/table_s3.py --grade outputs/retrain --strict
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRADE = ROOT / "outputs" / "grade_sim"
DEFAULT_CSV = ROOT / "outputs" / "tables" / "table_s3.csv"

# The 13 rungs of the attribution chain, in the order of Table 1 of the manuscript.
CHAIN = (
    "LSTM_Lumped_CalOnly",
    "LSTM_Lumped_RainOnly",
    "LSTM_Lumped",
    "LSTM",
    "LSTM_DUH",
    "LSTM_DUH_Fixed",
    "LSTM_DUH_SCS",
    "LSTM_DUH_SCS_PeOnly",
    "LSTM_DUH_Topmodel",
    "LSTM_DUH_Topmodel_PeOnly",
    "LSTM_DUH_Topmodel_Baseflow",
    "Phys_DUH_SCS",
    "Phys_DUH_Topmodel",
)

# Columns of Table S3 (key and label), all printed with two decimals.
COLUMNS = (("kge", "KGE"), ("r", "r"), ("alpha", "alpha"), ("beta", "beta"))
DECIMALS = 2

# Table S3 as printed in the submitted Supplementary Information, columns in the order of
# COLUMNS. Used only to report whether the recomputed values match.
MANUSCRIPT_TABLE_S3 = """
LSTM_Lumped_CalOnly         0.37  0.50  0.65  1.11
LSTM_Lumped_RainOnly        0.34  0.42  0.77  1.20
LSTM_Lumped                 0.69  0.83  0.76  1.04
LSTM                        0.76  0.90  0.81  1.01
LSTM_DUH                    0.79  0.89  0.85  1.02
LSTM_DUH_Fixed              0.76  0.88  0.84  1.01
LSTM_DUH_SCS                0.79  0.89  0.84  1.03
LSTM_DUH_SCS_PeOnly         0.76  0.80  0.94  1.06
LSTM_DUH_Topmodel           0.77  0.91  0.81  1.00
LSTM_DUH_Topmodel_PeOnly    0.85  0.92  0.88  1.00
LSTM_DUH_Topmodel_Baseflow  0.82  0.91  0.85  0.98
Phys_DUH_SCS                0.24  0.31  0.97  0.69
Phys_DUH_Topmodel           0.38  0.40  1.15  1.00
"""


@dataclass
class Run:
    """One completed run (configuration and seed)."""

    seed: int
    kge: float  # test_metrics.kge of results.json
    sim: np.ndarray  # simulated discharge of the test period, as stored
    obs: np.ndarray  # observed discharge of the test period, as stored


def kge_components(run: Run) -> dict[str, float]:
    """r, alpha and beta of one run over the finite pairs (as in scripts/consolida_sim.py)."""
    ok = np.isfinite(run.sim) & np.isfinite(run.obs)
    sim, obs = run.sim[ok], run.obs[ok]
    return {
        "r": float(np.corrcoef(sim, obs)[0, 1]),
        "alpha": float(sim.std() / (obs.std() + 1e-9)),
        "beta": float(sim.mean() / (obs.mean() + 1e-9)),
    }


def kge_from(parts: dict[str, float]) -> float:
    """KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)."""
    return 1 - math.sqrt(sum((parts[key] - 1) ** 2 for key in ("r", "alpha", "beta")))


def table_row(runs: list[Run], parts: list[dict[str, float]]) -> dict[str, float]:
    """Unrounded cells of one row of Table S3: means across seeds."""
    row = {"kge": float(np.mean([run.kge for run in runs]))}
    for key in ("r", "alpha", "beta"):
        row[key] = float(np.mean([p[key] for p in parts]))
    return row


def seed_folders(grade: Path) -> list[tuple[int, Path]]:
    """Folders seedNN of the grid, by increasing seed."""
    found = []
    for path in grade.glob("seed*"):
        match = re.fullmatch(r"seed(\d+)", path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def load_runs(grade: Path, config: str) -> list[Run]:
    """Completed runs of one configuration, by increasing seed.

    A seed folder without results.json is skipped: seeds 44 and 48 of Phys_DUH_Topmodel are
    interrupted runs that keep only config.json.
    """
    runs = []
    for seed, folder in seed_folders(grade):
        found = sorted((folder / config).glob("*/results.json"))
        if not found:
            continue
        if len(found) > 1:
            raise RuntimeError(f"more than one results.json under {folder / config}")
        metrics = json.loads(found[0].read_text(encoding="utf-8"))["test_metrics"]
        with np.load(found[0].with_name("predictions.npz")) as npz:
            sim, obs = npz["pred"][:, 0], npz["target"][:, 0]
        runs.append(Run(seed, float(metrics["kge"]), sim, obs))
    return runs


def manuscript_cells() -> dict[str, dict[str, Decimal]]:
    """Parse MANUSCRIPT_TABLE_S3 into {configuration: {column key: printed value}}."""
    cells = {}
    for line in MANUSCRIPT_TABLE_S3.strip().splitlines():
        name, *values = line.split()
        cells[name] = {key: Decimal(value) for (key, _), value in zip(COLUMNS, values, strict=True)}
    if tuple(cells) != CHAIN:
        raise RuntimeError("MANUSCRIPT_TABLE_S3 does not list the rungs in the order of CHAIN")
    return cells


def round_half_up(value: float, decimals: int = DECIMALS) -> Decimal:
    """Round half away from zero, from the shortest repr of the unrounded value."""
    step = Decimal(1).scaleb(-decimals)
    rounded = Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_UP)
    return abs(rounded) if rounded == 0 else rounded  # no negative zero


def rounding_margin(value: float, decimals: int = DECIMALS) -> float:
    """Distance from the value to the nearest rounding boundary (half a unit of the last digit)."""
    scaled = abs(value) * 10**decimals
    return abs(scaled - math.floor(scaled) - 0.5) / 10**decimals


def rounded_text(value: float) -> str:
    """Cell as printed in the Supplementary Information."""
    return f"{round_half_up(value)}" if math.isfinite(value) else "NA"


def number_list(values: list[int]) -> str:
    """'42 to 51' for a run of consecutive integers, else a comma-separated list."""
    if len(values) > 2 and values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]} to {values[-1]}"
    return ", ".join(str(v) for v in values)


def display_path(path: Path) -> str:
    """Path relative to the repository root when possible."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--grade",
        type=Path,
        default=DEFAULT_GRADE,
        help="grid folder with seedNN/<configuration>/<class>/ (default: outputs/grade_sim)",
    )
    parser.add_argument(
        "--cells",
        action="store_true",
        help="list every cell: unrounded value, rounded value and manuscript value",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        type=Path,
        const=DEFAULT_CSV,
        default=None,
        metavar="PATH",
        help="also write the unrounded and rounded values to a CSV file "
        "(default path: outputs/tables/table_s3.csv)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 if any cell differs from the manuscript",
    )
    args = parser.parse_args()

    manuscript = manuscript_cells()
    runs = {config: load_runs(args.grade, config) for config in CHAIN}
    missing = [config for config, found in runs.items() if not found]
    if missing:
        parser.error(f"no completed runs in {args.grade} for: {', '.join(missing)}")
    parts = {config: [kge_components(run) for run in found] for config, found in runs.items()}
    rows = {config: table_row(runs[config], parts[config]) for config in CHAIN}

    # ---- Table S3, rounded as in the Supplementary Information
    print("Table S3. KGE decomposition of the attribution chain")
    print("Test period, telemetric product. Mean across seeds of the KGE and of its components:")
    print("correlation r, variability ratio alpha = sd_sim / sd_obs, bias ratio")
    print("beta = mean_sim / mean_obs.")
    print(f"Grid: {display_path(args.grade)}")
    print()
    print(f"{'#':>2}  {'Configuration':<26} {'n':>2}" + "".join(f" {lab:>6}" for _, lab in COLUMNS))
    for rung, config in enumerate(CHAIN, 1):
        cells = "".join(f" {rounded_text(rows[config][key]):>6}" for key, _ in COLUMNS)
        print(f"{rung:>2}  {config:<26} {len(runs[config]):>2}{cells}")

    # ---- unrounded values
    print()
    print("Unrounded values (KGE from results.json; r, alpha and beta from predictions.npz)")
    print()
    print(f"{'#':>2}  {'Configuration':<26}" + "".join(f" {lab:>11}" for _, lab in COLUMNS))
    for rung, config in enumerate(CHAIN, 1):
        values = "".join(f" {rows[config][key]:>+11.6f}" for key, _ in COLUMNS)
        print(f"{rung:>2}  {config:<26}{values}")

    # ---- checks
    all_runs = [run for found in runs.values() for run in found]
    lengths = sorted({len(run.obs) for run in all_runs})
    kge_gap = max(
        abs(run.kge - kge_from(p))
        for config in CHAIN
        for run, p in zip(runs[config], parts[config], strict=True)
    )
    by_seeds: dict[tuple[int, ...], list[int]] = {}
    for rung, config in enumerate(CHAIN, 1):
        by_seeds.setdefault(tuple(run.seed for run in runs[config]), []).append(rung)

    print()
    print("Checks")
    print(f"  Runs read: {len(all_runs)}")
    for seeds, rungs in by_seeds.items():
        noun = "rung" if len(rungs) == 1 else "rungs"
        print(f"    seeds {number_list(list(seeds))}: {noun} {number_list(rungs)}")
    print(f"  Test time steps per run: {', '.join(str(n) for n in lengths)}")
    print("  KGE of results.json against 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)")
    print(f"    from predictions.npz, per run: max |difference| = {kge_gap:.1e}")

    compared, mismatches, margins, cell_lines = 0, [], [], []
    for rung, config in enumerate(CHAIN, 1):
        for key, label in COLUMNS:
            printed = manuscript[config][key]
            value = rows[config][key]
            compared += 1
            ok = math.isfinite(value) and round_half_up(value) == printed
            if not ok:
                mismatches.append((rung, config, label, printed, value))
            if math.isfinite(value):
                margins.append((rounding_margin(value), rung, config, label, value))
            cell_lines.append(
                f"{rung:>2}  {config:<26} {label:<6} {value:>+12.6f} {rounded_text(value):>7}"
                f" {printed!s:>10}  {'ok' if ok else 'DIFFERS'}"
            )
    print(
        f"  Manuscript Table S3: {compared - len(mismatches)} of {compared} cells match"
        " the values recomputed here."
    )
    for rung, config, label, printed, value in mismatches:
        print(f"    DIFFERS: rung {rung} {config} {label}: manuscript {printed}, data {value:+.6f}")
    print("  Cells closest to a rounding boundary (distance from the unrounded value):")
    for margin, rung, config, label, value in sorted(margins)[:3]:
        print(
            f"    rung {rung} {config} {label}: {value:+.6f} -> {round_half_up(value)}"
            f" (distance {margin:.6f})"
        )

    if args.cells:
        print()
        print(
            f"{'#':>2}  {'Configuration':<26} {'Column':<6} {'Unrounded':>12} {'Rounded':>7}"
            f" {'Manuscript':>10}  Status"
        )
        print("\n".join(cell_lines))

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["rung", "configuration", "n_seeds", "seeds"]
                + [key for key, _ in COLUMNS]
                + [f"{key}_rounded" for key, _ in COLUMNS]
            )
            for rung, config in enumerate(CHAIN, 1):
                row = rows[config]
                writer.writerow(
                    [rung, config, len(runs[config]), " ".join(str(r.seed) for r in runs[config])]
                    + [repr(row[key]) for key, _ in COLUMNS]
                    + [rounded_text(row[key]) for key, _ in COLUMNS]
                )
        print(f"\nWrote {display_path(args.csv)}")

    return 1 if args.strict and mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
