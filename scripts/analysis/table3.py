#!/usr/bin/env python
"""Reproduce Table 3 of the manuscript: performance of the attribution chain.

Prints every column of Table 3 for the 13 rungs of the attribution chain, in the order of
Table 1, over the test period (6,323 hourly time steps, telemetric forcing): NSE (mean and sample
standard deviation across seeds), NSE of the seed ensemble, logNSE (mean and sample standard
deviation), KGE, PBIAS, FHV and FLV. That is 12 x 9 + 7 = 115 cells, because rung 13
(Phys_DUH_Topmodel, seeds 42 and 43 only) has no standard deviation in the manuscript.

Sources, as in the computation behind the manuscript (``scripts/consolida_sim.py`` for the
per-seed metrics and the ensemble helpers ``_pred_mean`` and ``_nse_ensemble`` of
``scripts/figures/gerar_figuras_sim.py``):

- NSE and KGE: ``test_metrics`` of each ``results.json``, written by ``scripts/train.py`` after
  training.
- logNSE, PBIAS, FHV and FLV: recomputed from ``predictions.npz`` (simulated and observed
  discharge in m3/s, float32, used as stored) over the finite pairs.
- NSE (ens.): NSE of the single series obtained by averaging the predictions of the seeds step by
  step, scored against the observed series.

Definitions (manuscript Section 3.6 and Supplementary Section S11; denominators regularized
with 1e-9):

- logNSE: NSE of ln(Q + 0.01 mean(Q_obs)), for the simulated and the observed series.
- PBIAS = 100 sum(Q_sim - Q_obs) / sum(Q_obs), negative for underestimation.
- FHV: 100 (S_sim - S_obs) / S_obs, with S the sum of the k = floor(0.02 N) highest values of
  each independently sorted series.
- FLV, modified low-flow log-tail bias (Eq. S1): 100 (A_sim - A_obs) / |A_obs|, with A the sum
  of ln(Q + 0.001) over the k = floor(0.30 N) lowest values of each independently sorted series.

Rounded values are rounded half away from zero directly from the unrounded value (two decimals
for NSE, logNSE and KGE, one for the percentages), never from an already rounded number. Each
rounded cell is compared with Table 3 of the submitted manuscript and any difference is reported.

Reads ``outputs/grade_sim`` (``results.json`` and ``predictions.npz``) and writes nothing unless
``--csv`` is given.

Usage, from the repository root:
    python scripts/analysis/table3.py
    python scripts/analysis/table3.py --cells      # every cell: unrounded, rounded, manuscript
    python scripts/analysis/table3.py --csv        # also write outputs/tables/table3.csv
    python scripts/analysis/table3.py --grade outputs/retrain --strict
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
DEFAULT_CSV = ROOT / "outputs" / "tables" / "table3.csv"

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

# Columns of Table 3: key, label, decimals printed in the manuscript.
COLUMNS = (
    ("nse", "NSE", 2),
    ("nse_sd", "NSE sd", 2),
    ("nse_ens", "NSE (ens.)", 2),
    ("lognse", "logNSE", 2),
    ("lognse_sd", "logNSE sd", 2),
    ("kge", "KGE", 2),
    ("pbias", "PBIAS", 1),
    ("fhv", "FHV", 1),
    ("flv", "FLV", 1),
)
SIGNED = {"pbias", "fhv"}  # printed with an explicit plus sign in the manuscript

# Table 3 as printed in the submitted manuscript, columns in the order of COLUMNS (NA: cell not
# printed). Used only to report whether the recomputed values match.
MANUSCRIPT_TABLE3 = """
LSTM_Lumped_CalOnly          0.17  0.05  0.20 -0.01  0.07  0.37 +10.5 -47.6   7.6
LSTM_Lumped_RainOnly        -0.09  0.10 -0.04 -0.33  0.11  0.34 +19.8  -3.2  15.5
LSTM_Lumped                  0.67  0.07  0.72  0.58  0.13  0.69  +3.7 -14.6   7.1
LSTM                         0.76  0.11  0.84  0.69  0.13  0.76  +0.6  -8.0   6.0
LSTM_DUH                     0.77  0.08  0.85  0.70  0.12  0.79  +2.0  -5.8   6.1
LSTM_DUH_Fixed               0.74  0.06  0.83  0.67  0.11  0.76  +0.8  -6.9   5.8
LSTM_DUH_SCS                 0.77  0.10  0.84  0.73  0.13  0.79  +3.1  -7.3   5.7
LSTM_DUH_SCS_PeOnly          0.59  0.12  0.66  0.61  0.15  0.76  +5.7  +8.7   6.9
LSTM_DUH_Topmodel            0.80  0.06  0.86  0.75  0.04  0.77  +0.0 -10.4   5.6
LSTM_DUH_Topmodel_PeOnly     0.83  0.06  0.88  0.79  0.06  0.85  -0.3  -5.9   4.2
LSTM_DUH_Topmodel_Baseflow   0.82  0.06  0.88  0.78  0.06  0.82  -1.7  -6.5   4.2
Phys_DUH_SCS                -0.67  0.16 -0.66 -0.92  0.61  0.24 -30.5 +20.6  -0.4
Phys_DUH_Topmodel           -0.40    NA -0.40 -0.52    NA  0.38  -0.4 +10.9   0.9
"""


@dataclass
class Run:
    """One completed run (configuration and seed)."""

    seed: int
    nse: float  # test_metrics.nse of results.json
    kge: float  # test_metrics.kge of results.json
    sim: np.ndarray  # simulated discharge of the test period, as stored
    obs: np.ndarray  # observed discharge of the test period, as stored


# ---------------------------------------------------------------------------- metrics


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency over the finite pairs (``_nse`` of the figure generator)."""
    ok = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[ok], obs[ok]
    den = np.sum((o - o.mean()) ** 2)
    return float(1 - np.sum((s - o) ** 2) / den) if den > 0 else float("nan")


def log_nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """NSE of ln(Q + 0.01 mean(Q_obs)) (``_lognse`` of scripts/consolida_sim.py)."""
    eps = 0.01 * np.nanmean(obs)
    log_sim, log_obs = np.log(sim + eps), np.log(obs + eps)
    den = np.sum((log_obs - log_obs.mean()) ** 2)
    return float(1 - np.sum((log_sim - log_obs) ** 2) / den) if den > 0 else float("nan")


def percent_bias(sim: np.ndarray, obs: np.ndarray) -> float:
    """PBIAS in percent, negative for underestimation (as in scripts/consolida_sim.py)."""
    return float(100 * (sim.sum() - obs.sum()) / (obs.sum() + 1e-9))


def fhv_flv(sim: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    """FHV (2% highest flows) and modified FLV (Eq. S1), as ``_fdc`` of consolida_sim.py."""
    n = len(obs)
    k_high = max(1, int(0.02 * n))
    sim_desc, obs_desc = np.sort(sim)[::-1], np.sort(obs)[::-1]
    high_sim, high_obs = sim_desc[:k_high].sum(), obs_desc[:k_high].sum()
    fhv = 100 * (high_sim - high_obs) / (high_obs + 1e-9)
    k_low = int(0.3 * n)
    a_sim = np.log(np.sort(sim)[:k_low] + 1e-3).sum()
    a_obs = np.log(np.sort(obs)[:k_low] + 1e-3).sum()
    flv = 100 * (a_sim - a_obs) / (abs(a_obs) + 1e-9)
    return float(fhv), float(flv)


def signatures(run: Run) -> dict[str, float]:
    """logNSE, PBIAS, FHV and FLV of one run, over the finite pairs."""
    ok = np.isfinite(run.sim) & np.isfinite(run.obs)
    sim, obs = run.sim[ok], run.obs[ok]
    fhv, flv = fhv_flv(sim, obs)
    return {"lognse": log_nse(sim, obs), "pbias": percent_bias(sim, obs), "fhv": fhv, "flv": flv}


def ensemble_nse(runs: list[Run]) -> float:
    """NSE of the seed ensemble (``_pred_mean`` and ``_nse_ensemble`` of the figure generator).

    The predictions of the seeds are averaged step by step, truncated to the shortest series
    (all have the same length here), and scored against the observed series of the first seed.
    """
    n = min(len(run.sim) for run in runs)
    mean_sim = np.mean([run.sim[:n] for run in runs], axis=0)
    return nse(mean_sim, runs[0].obs[:n])


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (divisor n - 1); NaN deviation for a single value."""
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    return float(arr.mean()), sd


def table_row(runs: list[Run]) -> dict[str, float]:
    """Unrounded cells of one row of Table 3."""
    sigs = [signatures(run) for run in runs]
    nse_mean, nse_sd = mean_sd([run.nse for run in runs])
    lognse_mean, lognse_sd = mean_sd([s["lognse"] for s in sigs])
    return {
        "nse": nse_mean,
        "nse_sd": nse_sd,
        "nse_ens": ensemble_nse(runs),
        "lognse": lognse_mean,
        "lognse_sd": lognse_sd,
        "kge": float(np.mean([run.kge for run in runs])),
        "pbias": float(np.mean([s["pbias"] for s in sigs])),
        "fhv": float(np.mean([s["fhv"] for s in sigs])),
        "flv": float(np.mean([s["flv"] for s in sigs])),
    }


# ---------------------------------------------------------------------------- input


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
        runs.append(Run(seed, float(metrics["nse"]), float(metrics["kge"]), sim, obs))
    return runs


def manuscript_cells() -> dict[str, dict[str, Decimal | None]]:
    """Parse MANUSCRIPT_TABLE3 into {configuration: {column key: printed value or None}}."""
    cells = {}
    for line in MANUSCRIPT_TABLE3.strip().splitlines():
        name, *values = line.split()
        cells[name] = {
            key: None if value == "NA" else Decimal(value)
            for (key, _, _), value in zip(COLUMNS, values, strict=True)
        }
    if tuple(cells) != CHAIN:
        raise RuntimeError("MANUSCRIPT_TABLE3 does not list the rungs in the order of CHAIN")
    return cells


# ---------------------------------------------------------------------------- formatting


def round_half_up(value: float, decimals: int) -> Decimal:
    """Round half away from zero, from the shortest repr of the unrounded value."""
    step = Decimal(1).scaleb(-decimals)
    rounded = Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_UP)
    return abs(rounded) if rounded == 0 else rounded  # no negative zero


def rounding_margin(value: float, decimals: int) -> float:
    """Distance from the value to the nearest rounding boundary (half a unit of the last digit)."""
    scaled = abs(value) * 10**decimals
    return abs(scaled - math.floor(scaled) - 0.5) / 10**decimals


def rounded_text(value: float, key: str, decimals: int) -> str:
    """Cell as printed in the manuscript (explicit plus sign for PBIAS and FHV)."""
    if not math.isfinite(value):
        return "NA"
    rounded = round_half_up(value, decimals)
    return f"{rounded:+}" if key in SIGNED else f"{rounded}"


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


# ---------------------------------------------------------------------------- main


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
        "(default path: outputs/tables/table3.csv)",
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
    rows = {config: table_row(found) for config, found in runs.items()}

    # ---- Table 3, rounded as in the manuscript
    print("Table 3. Performance of the attribution chain in continuous simulation")
    print("Test period, telemetric product. Mean +/- sample standard deviation across seeds;")
    print("NSE (ens.): NSE of the seed-mean series; PBIAS, FHV and FLV in percent.")
    print(f"Grid: {display_path(args.grade)}")
    print()
    print(
        f"{'#':>2}  {'Model':<26} {'n':>2}  {'NSE':<14} {'NSE (ens.)':>10}  {'logNSE':<14}"
        f" {'KGE':>5} {'PBIAS':>6} {'FHV':>6} {'FLV':>5}"
    )
    for rung, config in enumerate(CHAIN, 1):
        row = rows[config]
        cell = {key: rounded_text(row[key], key, nd) for key, _, nd in COLUMNS}
        nse_text, lognse_text = f"{cell['nse']:>5}", f"{cell['lognse']:>5}"
        if len(runs[config]) > 2:  # as in the manuscript, no deviation for the two-seed rung 13
            nse_text += f" +/- {cell['nse_sd']}"
            lognse_text += f" +/- {cell['lognse_sd']}"
        print(
            f"{rung:>2}  {config:<26} {len(runs[config]):>2}  {nse_text:<14} {cell['nse_ens']:>10}"
            f"  {lognse_text:<14} {cell['kge']:>5} {cell['pbias']:>6} {cell['fhv']:>6}"
            f" {cell['flv']:>5}"
        )

    # ---- unrounded values
    print()
    print("Unrounded values (NSE and KGE from results.json; NSE (ens.), logNSE, PBIAS, FHV and")
    print("FLV from predictions.npz)")
    blocks = (COLUMNS[:5], COLUMNS[5:])
    for block in blocks:
        print()
        print(f"{'#':>2}  {'Model':<26}" + "".join(f" {label:>11}" for _, label, _ in block))
        for rung, config in enumerate(CHAIN, 1):
            values = "".join(f" {rows[config][key]:>+11.6f}" for key, _, _ in block)
            print(f"{rung:>2}  {config:<26}{values}")

    # ---- checks
    all_runs = [run for found in runs.values() for run in found]
    lengths = sorted({len(run.obs) for run in all_runs})
    all_finite = all(
        bool(np.isfinite(run.sim).all() and np.isfinite(run.obs).all()) for run in all_runs
    )
    reference = all_runs[0].obs
    same_obs = all(np.array_equal(run.obs, reference, equal_nan=True) for run in all_runs)
    nse_gap = max(abs(run.nse - nse(run.sim, run.obs)) for run in all_runs)
    by_seeds: dict[tuple[int, ...], list[int]] = {}
    for rung, config in enumerate(CHAIN, 1):
        by_seeds.setdefault(tuple(run.seed for run in runs[config]), []).append(rung)

    print()
    print("Checks")
    print(f"  Runs read: {len(all_runs)}")
    for seeds, rungs in by_seeds.items():
        noun = "rung" if len(rungs) == 1 else "rungs"
        print(f"    seeds {number_list(list(seeds))}: {noun} {number_list(rungs)}")
    finite_text = "all finite" if all_finite else "NOT all finite"
    print(f"  Test time steps per run: {', '.join(str(n) for n in lengths)} ({finite_text})")
    print(f"  Observed series identical in all runs: {'yes' if same_obs else 'NO'}")
    print(
        "  NSE of results.json against NSE recomputed from predictions.npz:"
        f" max |difference| = {nse_gap:.1e}"
    )
    last = runs[CHAIN[-1]]
    per_seed = ", ".join(f"seed {run.seed}: {run.nse:+.6f}" for run in last)
    print(f"  Rung 13 ({CHAIN[-1]}) NSE by seed: {per_seed}")
    if len(last) == 2:
        print(f"    difference between the two seeds: {abs(last[0].nse - last[1].nse):.6f}")

    compared, mismatches, margins, cell_lines = 0, [], [], []
    for rung, config in enumerate(CHAIN, 1):
        for key, label, nd in COLUMNS:
            printed = manuscript[config][key]
            value = rows[config][key]
            if printed is None:
                continue
            compared += 1
            ok = math.isfinite(value) and round_half_up(value, nd) == printed
            if not ok:
                mismatches.append((rung, config, label, printed, value))
            if math.isfinite(value):
                margins.append((rounding_margin(value, nd), rung, config, label, value, nd))
            shown = f"{printed:+}" if key in SIGNED else f"{printed}"
            cell_lines.append(
                f"{rung:>2}  {config:<26} {label:<10} {value:>+12.6f}"
                f" {rounded_text(value, key, nd):>7} {shown:>10}  {'ok' if ok else 'DIFFERS'}"
            )
    print(
        f"  Manuscript Table 3: {compared - len(mismatches)} of {compared} cells match"
        " the values recomputed here."
    )
    for rung, config, label, printed, value in mismatches:
        print(f"    DIFFERS: rung {rung} {config} {label}: manuscript {printed}, data {value:+.6f}")
    print("  Cells closest to a rounding boundary (distance from the unrounded value):")
    for margin, rung, config, label, value, nd in sorted(margins)[:3]:
        print(
            f"    rung {rung} {config} {label}: {value:+.6f} -> {round_half_up(value, nd)}"
            f" (distance {margin:.6f})"
        )

    if args.cells:
        print()
        print(
            f"{'#':>2}  {'Model':<26} {'Column':<10} {'Unrounded':>12} {'Rounded':>7}"
            f" {'Manuscript':>10}  Status"
        )
        print("\n".join(cell_lines))

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["rung", "configuration", "n_seeds", "seeds"]
                + [key for key, _, _ in COLUMNS]
                + [f"{key}_rounded" for key, _, _ in COLUMNS]
            )
            for rung, config in enumerate(CHAIN, 1):
                row = rows[config]
                writer.writerow(
                    [rung, config, len(runs[config]), " ".join(str(r.seed) for r in runs[config])]
                    + [repr(row[key]) for key, _, _ in COLUMNS]
                    + [rounded_text(row[key], key, nd) for key, _, nd in COLUMNS]
                )
        print(f"\nWrote {display_path(args.csv)}")

    return 1 if args.strict and mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
