#!/usr/bin/env python
"""Reproduce the test-period diagnostics of Sections 2.5 and 5.1 of the manuscript.

Section 5.1 qualifies the test results by the test period itself: a single period of about nine
months, drier than training, whose NSE is weighted toward its first three weeks (the 501 scored
hours from 10 to 30 April 2025 carry about 59% of the NSE denominator), while the NSE gain of the
best hybrid (LSTM_DUH_Topmodel_PeOnly) over the pure LSTM is shared among April, October and
December; in the validation window the ordering of the two reverses (mean NSE 0.75 against 0.69,
the LSTM ahead in 9 of 10 seeds). The script prints:

1. the three chronological splits of the telemetric dataset (UTC), their share of the record, the
   number of scored windows and the length of the test period. With lookback 240 h and horizon
   1 h, the window that starts at hour i is scored when the discharge of its target hour i + 240
   and the rainfall of its input hours i to i + 239 are finite, the rule of ``AblationDataset`` in
   ``scripts/train.py``;
2. how dry the test period is: catchment-mean rainfall (area-weighted over the 245
   sub-catchments) and mean observed discharge of each split, and the same two quantities over the
   calendar window of the scored test hours (10 April 01:00 to 31 December 00:00 UTC) in each year;
3. the share of each month in the NSE denominator, the sum over the scored test hours of the
   squared deviation of the observed discharge from its test-period mean;
4. the monthly decomposition of the NSE gain of LSTM_DUH_Topmodel_PeOnly over LSTM. Both are
   scored against the same observed series, so NSE_hybrid - NSE_LSTM is the sum over the scored
   hours of [(Q_LSTM - Q_obs)^2 - (Q_hybrid - Q_obs)^2] divided by the denominator; the sum is
   split by month (UTC) and averaged over the ten seed pairs (same seed in both configurations);
5. the validation scores: ``best_val_nse`` of each results.json, the highest validation NSE of the
   run, which selects its checkpoint. Their mean and population standard deviation are compared
   with the NSE column of Supplementary Table S4, whose full panel is recomputed from the
   checkpoints by ``scripts/analysis/validation_panel.py``.

The numbers quoted in the manuscript are then compared with the data, and the observed series
rebuilt from the dataset is checked against ``target`` of every ``predictions.npz`` read.
Simulated and observed discharge are used as stored (float32) and summed in float64.

Values are printed unrounded and rounded half away from zero from the unrounded value: to two
decimals and, where the manuscript quotes another precision, also to that precision.

Reads ``data/dataset_58585000_telem.h5`` and ``outputs/grade_sim`` (results.json and
predictions.npz) and writes nothing unless ``--csv`` is given.

Usage, from the repository root:
    python scripts/analysis/test_period.py
    python scripts/analysis/test_period.py --per-seed  # monthly gain of every seed pair
    python scripts/analysis/test_period.py --csv       # outputs/tables/test_period_months.csv
    python scripts/analysis/test_period.py --grade outputs/retrain --strict
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRADE = ROOT / "outputs" / "grade_sim"
DEFAULT_H5 = ROOT / "data" / "dataset_58585000_telem.h5"
DEFAULT_CSV = ROOT / "outputs" / "tables" / "test_period_months.csv"

LOOKBACK = 240  # hours of forcing in each window (Table 2)
HORIZON = 1  # one-step window
SPLITS = ("train", "val", "test")
REFERENCE = "LSTM"  # pure distributed LSTM, rung 4
BEST = "LSTM_DUH_Topmodel_PeOnly"  # best hybrid, rung 10
VALIDATION = ("LSTM", "LSTM_DUH_Topmodel", "LSTM_DUH_Topmodel_PeOnly")  # rows of Table S4
DAYS_PER_MONTH = 365.25 / 12


@dataclass
class Split:
    """One chronological split of the dataset."""

    name: str
    time: pd.DatetimeIndex  # hourly, UTC
    flow: np.ndarray  # observed discharge (m3/s) as stored (float32), NaN in the gaps
    rain: np.ndarray  # catchment-mean rainfall (mm/h), area-weighted, float64
    scored: np.ndarray  # indices of the scored target hours


@dataclass
class Run:
    """One completed run of one configuration."""

    seed: int
    nse: float  # test_metrics.nse of results.json
    best_val: float  # best_val_nse of results.json
    sim: np.ndarray  # simulated test discharge, as stored
    obs: np.ndarray  # observed test discharge, as stored


# ---------------------------------------------------------------------------- input


def scored_targets(
    flow: np.ndarray, rain_gap: np.ndarray, lookback: int = LOOKBACK, horizon: int = HORIZON
) -> np.ndarray:
    """Indices of the first target hour of every valid window (``_create_valid_indices``)."""
    n_windows = max(len(flow) - lookback - horizon + 1, 0)
    start = np.arange(n_windows)
    flow_gaps = np.concatenate([[0], np.cumsum(np.isnan(flow))])
    rain_gaps = np.concatenate([[0], np.cumsum(rain_gap)])
    target_ok = flow_gaps[start + lookback + horizon] == flow_gaps[start + lookback]
    input_ok = rain_gaps[start + lookback] == rain_gaps[start]
    return start[target_ok & input_ok] + lookback


def load_splits(h5_path: Path) -> dict[str, Split]:
    """The three splits of the telemetric dataset."""
    splits = {}
    with h5py.File(h5_path, "r") as f:
        area = f["ottobacia/area_km2"][:].astype(np.float64)
        weights = area / area.sum()
        for name in SPLITS:
            precip = f[f"{name}/precipitation"][:]
            flow = f[f"{name}/streamflow"][:]
            splits[name] = Split(
                name=name,
                time=pd.to_datetime(f[f"{name}/timestamps"][:], unit="s", utc=True),
                flow=flow,
                rain=precip.astype(np.float64) @ weights,
                scored=scored_targets(flow, np.isnan(precip).any(axis=1)),
            )
    return splits


def seed_folders(grade: Path) -> list[tuple[int, Path]]:
    """Folders seedNN of the grid, by increasing seed."""
    found = []
    for path in grade.glob("seed*"):
        match = re.fullmatch(r"seed(\d+)", path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def load_runs(grade: Path, config: str) -> dict[int, Run]:
    """Completed runs of one configuration, by seed."""
    runs = {}
    for seed, folder in seed_folders(grade):
        found = sorted((folder / config).glob("*/results.json"))
        if not found:
            continue
        if len(found) > 1:
            raise RuntimeError(f"more than one results.json under {folder / config}")
        result = json.loads(found[0].read_text(encoding="utf-8"))
        with np.load(found[0].with_name("predictions.npz")) as npz:
            sim, obs = npz["pred"][:, 0], npz["target"][:, 0]
        runs[seed] = Run(
            seed,
            float(result["test_metrics"]["nse"]),
            float(result["best_val_nse"]),
            sim,
            obs,
        )
    return runs


# ---------------------------------------------------------------------------- analysis


def same_window_by_year(
    splits: dict[str, Split], first: pd.Timestamp, last: pd.Timestamp
) -> list[dict[str, int | float | str]]:
    """Rainfall total and mean discharge over the calendar window first..last of every year."""
    time = splits["train"].time.append([splits["val"].time, splits["test"].time])
    rain = np.concatenate([splits[s].rain for s in SPLITS])
    flow = np.concatenate([splits[s].flow for s in SPLITS]).astype(np.float64)
    label = np.concatenate([np.full(len(splits[s].time), s) for s in SPLITS])
    rows = []
    for year in sorted(set(time.year)):
        start = first.replace(year=year)
        end = last.replace(year=year)
        sel = np.asarray((time >= start) & (time <= end))
        if not sel.any():
            continue
        valid = sel & np.isfinite(flow)
        names = [s for s in SPLITS if (label[sel] == s).any()]
        rows.append(
            {
                "year": year,
                "splits": "+".join(names),
                "hours": int(sel.sum()),
                "rain_mm": float(rain[sel].sum()),
                "flow_mean": float(flow[valid].mean()),
                "flow_cover": float(valid.sum() / sel.sum()),
            }
        )
    return rows


def month_decomposition(
    obs: np.ndarray, months: np.ndarray, reference: dict[int, Run], hybrid: dict[int, Run]
) -> tuple[list[int], np.ndarray, float, np.ndarray]:
    """Seeds, per-hour denominator terms, denominator and gain by seed pair and month (1 to 12)."""
    o = obs.astype(np.float64)
    terms = (o - o.mean()) ** 2
    denominator = float(terms.sum())
    seeds = sorted(set(reference) & set(hybrid))
    gains = np.zeros((len(seeds), 12))
    for row, seed in enumerate(seeds):
        err_ref = (reference[seed].sim.astype(np.float64) - o) ** 2
        err_hyb = (hybrid[seed].sim.astype(np.float64) - o) ** 2
        contribution = (err_ref - err_hyb) / denominator
        for month in range(1, 13):
            gains[row, month - 1] = contribution[months == month].sum()
    return seeds, terms, denominator, gains


# ---------------------------------------------------------------------------- formatting


def round_half_up(value: float, decimals: int) -> Decimal:
    """Round half away from zero, from the shortest repr of the unrounded value."""
    step = Decimal(1).scaleb(-decimals)
    rounded = Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_UP)
    return abs(rounded) if rounded == 0 else rounded  # no negative zero


def stamp(t: pd.Timestamp) -> str:
    """Timestamp as 'YYYY-MM-DD HH:MM UTC'."""
    return t.strftime("%Y-%m-%d %H:%M UTC")


def display_path(path: Path) -> str:
    """Path relative to the repository root when possible."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


class Report:
    """Checks of the statements of the manuscript, printed as they are made."""

    def __init__(self) -> None:
        self.compared = 0
        self.differs: list[str] = []

    def _status(self, ok: bool, label: str) -> str:
        self.compared += 1
        if not ok:
            self.differs.append(label)
        return "ok" if ok else "DIFFERS"

    def value(self, label: str, raw: float, decimals: int, text: str) -> None:
        """One number: unrounded, two decimals, precision of the text, text value."""
        rounded = round_half_up(raw, decimals)
        status = self._status(rounded == Decimal(text), label)
        extra = f"  {decimals} dp {rounded}" if decimals != 2 else ""
        print(
            f"  {status:<7} {label:<60} {raw:>10.6g}  2 dp {round_half_up(raw, 2)}{extra}"
            f"  text {text}"
        )

    def count(self, label: str, found: int, text: int) -> None:
        """An integer quoted in the text."""
        status = self._status(found == text, label)
        print(f"  {status:<7} {label:<60} {found:>10,}  text {text:,}")

    def flag(self, label: str, ok: bool, detail: str = "") -> None:
        """A qualitative statement checked by a condition on the data."""
        status = self._status(ok, label)
        print(f"  {status:<7} {label}")
        for line in detail.splitlines():
            print(f"            {line}")


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
        "--h5",
        type=Path,
        default=DEFAULT_H5,
        help="telemetric dataset (default: data/dataset_58585000_telem.h5)",
    )
    parser.add_argument(
        "--per-seed", action="store_true", help="also print the monthly gain of every seed pair"
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        type=Path,
        const=DEFAULT_CSV,
        default=None,
        metavar="PATH",
        help="also write the monthly table to a CSV file "
        "(default path: outputs/tables/test_period_months.csv)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 if any statement differs from the data",
    )
    args = parser.parse_args()

    splits = load_splits(args.h5)
    runs = {config: load_runs(args.grade, config) for config in VALIDATION}
    missing = [config for config, found in runs.items() if not found]
    if missing:
        parser.error(f"no completed runs in {args.grade} for: {', '.join(missing)}")

    test = splits["test"]
    scored_time = test.time[test.scored]
    obs = test.flow[test.scored]
    months = np.asarray(scored_time.month)
    mismatched = [
        f"{config} seed {run.seed}"
        for config, found in runs.items()
        for run in found.values()
        if not np.array_equal(run.obs, obs)
    ]
    if mismatched:
        parser.error(
            "the observed test series rebuilt from the dataset differs from target of"
            f" predictions.npz for: {', '.join(mismatched)}"
        )

    print("Test-period diagnostics")
    print(f"Dataset: {display_path(args.h5)}; grid: {display_path(args.grade)}")
    print(f"Windows: lookback {LOOKBACK} h, horizon {HORIZON} h; timestamps in UTC")

    # ---- 1. splits and windows
    total_hours = sum(len(s.time) for s in splits.values())
    print()
    print("1. Splits")
    print(
        f"  {'split':<6} {'first hour':<21} {'last hour':<21} {'hours':>6} {'share':>8}"
        f" {'Q valid':>8} {'scored':>7}  first scored hour"
    )
    for s in splits.values():
        valid = np.isfinite(s.flow)
        print(
            f"  {s.name:<6} {stamp(s.time[0]):<21} {stamp(s.time[-1]):<21} {len(s.time):>6}"
            f" {100 * len(s.time) / total_hours:>7.2f}% {100 * valid.mean():>7.2f}%"
            f" {len(s.scored):>7}  {stamp(s.time[s.scored[0]])}"
        )
    split_days = len(test.time) / 24
    span_hours = int((scored_time[-1] - scored_time[0]) / pd.Timedelta(hours=1)) + 1
    print(
        f"  test split: {len(test.time)} h = {split_days:.2f} days = "
        f"{split_days / DAYS_PER_MONTH:.4f} months of {DAYS_PER_MONTH:.4f} days"
    )
    print(
        f"  scored test hours: {len(test.scored)}, from {stamp(scored_time[0])} to"
        f" {stamp(scored_time[-1])};\n  span {span_hours} h = {span_hours / 24:.2f} days ="
        f" {span_hours / 24 / DAYS_PER_MONTH:.4f} months"
    )
    present = sorted(set(months))
    print(
        f"  calendar months with scored hours: {len(present)}"
        f" ({calendar.month_name[present[0]]} to {calendar.month_name[present[-1]]})"
    )

    # ---- 2. how dry
    print()
    print("2. Rainfall and discharge (catchment-mean rainfall over all hours; discharge over the")
    print("   hours with data)")
    print(f"  {'split':<6} {'rain mm/day':>12} {'mm/yr rate':>11} {'Q mean m3/s':>12}")
    rates = {}
    for s in splits.values():
        rain_day = float(s.rain.mean() * 24)
        flow_mean = float(np.nanmean(s.flow.astype(np.float64)))
        rates[s.name] = (rain_day, flow_mean)
        print(f"  {s.name:<6} {rain_day:>12.4f} {rain_day * 365.25:>11.1f} {flow_mean:>12.4f}")
    window = same_window_by_year(splits, scored_time[0], scored_time[-1])
    print(
        f"  same calendar window in every year ({scored_time[0].strftime('%d %B %H:%M')} to"
        f" {scored_time[-1].strftime('%d %B %H:%M')} UTC):"
    )
    print(
        f"  {'year':<6} {'splits':<10} {'hours':>6} {'rain mm':>9} {'Q mean m3/s':>12}"
        f" {'Q cover':>8}"
    )
    for row in window:
        print(
            f"  {row['year']:<6} {row['splits']:<10} {row['hours']:>6} {row['rain_mm']:>9.1f}"
            f" {row['flow_mean']:>12.4f} {100 * row['flow_cover']:>7.1f}%"
        )
    last_year = window[-1]
    earlier = window[:-1]
    rain_rank = 1 + sum(r["rain_mm"] < last_year["rain_mm"] for r in earlier)
    flow_rank = 1 + sum(r["flow_mean"] < last_year["flow_mean"] for r in earlier)
    print(
        f"  {last_year['year']} against the mean of {earlier[0]['year']} to"
        f" {earlier[-1]['year']}: rainfall {last_year['rain_mm']:.1f} against"
        f" {np.mean([r['rain_mm'] for r in earlier]):.1f} mm, mean discharge"
        f" {last_year['flow_mean']:.2f} against {np.mean([r['flow_mean'] for r in earlier]):.2f}"
        f" m3/s;\n  rank of {last_year['year']} among the {len(window)} years from the lowest:"
        f" rainfall {rain_rank}, discharge {flow_rank}"
    )

    # ---- 3 and 4. months
    seeds, terms, denominator, gains = month_decomposition(obs, months, runs[REFERENCE], runs[BEST])
    mean_gain = gains.mean(axis=0)
    total_gain = float(mean_gain.sum())
    json_gain = float(np.mean([runs[BEST][s].nse - runs[REFERENCE][s].nse for s in seeds]))
    print()
    print(f"3 and 4. Months of the scored test period (denominator {denominator:.6g} (m3/s)^2)")
    print(f"  gain: NSE of {BEST} minus NSE of {REFERENCE}, mean over {len(seeds)} seed pairs")
    print(
        f"  {'month':<10} {'hours':>5} {'Q mean':>8} {'share of denominator':>21}"
        f" {'gain':>9} {'share of gain':>14} {'pairs > 0':>10}"
    )
    rows = []
    for month in present:
        sel = months == month
        hours = int(sel.sum())
        q_mean = float(obs[sel].astype(np.float64).mean())
        share_den = float(terms[sel].sum() / denominator)
        gain = float(mean_gain[month - 1])
        positive = int((gains[:, month - 1] > 0).sum())
        rows.append((month, hours, q_mean, share_den, gain, positive))
        print(
            f"  {calendar.month_name[month]:<10} {hours:>5} {q_mean:>8.2f}"
            f" {100 * share_den:>20.2f}% {gain:>+9.5f} {100 * gain / total_gain:>13.2f}%"
            f" {positive:>6}/{len(seeds)}"
        )
    print(
        f"  {'total':<10} {len(obs):>5} {float(obs.astype(np.float64).mean()):>8.2f}"
        f" {100.0:>20.2f}% {total_gain:>+9.5f} {100.0:>13.2f}%"
    )
    if args.per_seed:
        print("  gain by seed pair and month:")
        print(
            f"  {'seed':<6}"
            + "".join(f" {calendar.month_abbr[m]:>8}" for m in present)
            + f" {'total':>9}"
        )
        for row, seed in enumerate(seeds):
            values = "".join(f" {gains[row, m - 1]:>+8.4f}" for m in present)
            print(f"  {seed:<6}{values} {gains[row].sum():>+9.4f}")

    # ---- 5. validation
    print()
    print("5. Validation (best_val_nse: highest validation NSE of each run, which selects the")
    print("   checkpoint) and test NSE (results.json), mean over seeds; sd: sample (n - 1) and")
    print("   population (n, as in Table S4)")
    print(
        f"  validation split from {stamp(splits['val'].time[0])} to {stamp(splits['val'].time[-1])}"
    )
    print(
        f"  {'configuration':<26} {'validation':>10} {'(2 dp)':>6} {'sd n-1':>8} {'sd n':>8}"
        f" {'(2 dp)':>6} {'test':>10} {'(2 dp)':>6}"
    )
    val_stats = {}
    for config in VALIDATION:
        found = runs[config]
        val = np.array([found[s].best_val for s in sorted(found)])
        tst = np.array([found[s].nse for s in sorted(found)])
        val_stats[config] = (float(val.mean()), float(val.std(ddof=0)))
        print(
            f"  {config:<26} {val.mean():>10.6f} {round_half_up(val.mean(), 2)!s:>6}"
            f" {val.std(ddof=1):>8.5f} {val.std(ddof=0):>8.5f}"
            f" {round_half_up(val.std(ddof=0), 2)!s:>6} {tst.mean():>10.6f}"
            f" {round_half_up(tst.mean(), 2)!s:>6}"
        )
    val_months = set(splits["val"].time.month)
    paired = sorted(set(runs[REFERENCE]) & set(runs[BEST]))
    ahead = [s for s in paired if runs[REFERENCE][s].best_val > runs[BEST][s].best_val]
    print(
        f"  {REFERENCE} ahead of {BEST} in validation in {len(ahead)} of {len(paired)} seeds"
        f" (behind in seed {', '.join(str(s) for s in paired if s not in ahead) or 'none'})"
    )

    # ---- statements
    rep = Report()
    print()
    print("Statements of the manuscript")
    print("(status, statement, unrounded value, two decimals, precision of the text, text)")
    print()
    print("Section 2.5")
    windows = {name: len(s.scored) for name, s in splits.items()}
    rep.count("training windows", windows["train"], 27292)
    rep.count("validation windows", windows["val"], 5998)
    rep.count("test windows", windows["test"], 6323)
    for name, text in (("train", "70"), ("val", "15"), ("test", "15")):
        rep.value(
            f"{name} split, % of the hourly record",
            100 * len(splits[name].time) / total_hours,
            0,
            text,
        )
    rep.value(
        "test period spans nine months (test split, months)", split_days / DAYS_PER_MONTH, 0, "9"
    )
    drier = rates["test"][0] < rates["train"][0] and rates["test"][1] < rates["train"][1]
    rep.flag(
        "test period drier than the training period (also Sections 4.6, 5.1, S6, 6)",
        drier,
        f"mean rainfall {rates['test'][0]:.4f} against {rates['train'][0]:.4f} mm/day; mean"
        f" discharge {rates['test'][1]:.4f} against {rates['train'][1]:.4f} m3/s",
    )
    print()
    print("Section 5.1")
    rep.value(
        "about nine months (scored test hours, months)", span_hours / 24 / DAYS_PER_MONTH, 0, "9"
    )
    april = months == 4
    april_time = scored_time[april]
    rep.count("scored hours in April 2025", int(april.sum()), 501)
    rep.flag(
        "April hours run from 10 to 30 April, the first three weeks",
        april_time[0].day == 10 and april_time[-1].day == 30 and april_time[0] == scored_time[0],
        f"from {stamp(april_time[0])} to {stamp(april_time[-1])}",
    )
    rep.value(
        "share of April in the NSE denominator (%)", 100 * terms[april].sum() / denominator, 0, "59"
    )
    shares = {calendar.month_name[m]: float(mean_gain[m - 1] / total_gain) for m in present}
    top3 = sorted(shares, key=shares.get, reverse=True)[:3]
    rep.flag(
        "gain of the best hybrid shared among April, October and December, not concentrated in"
        " April",
        set(top3) == {"April", "October", "December"} and shares["April"] < 0.5,
        "share of the mean gain: "
        + ", ".join(
            f"{m} {100 * shares[m]:.4g}% ({round_half_up(100 * shares[m], 0)}%)" for m in top3
        )
        + f"; the three together {100 * sum(shares[m] for m in top3):.4g}%",
    )
    rep.flag(
        "validation window spans a full wet season (October to March)",
        {10, 11, 12, 1, 2, 3} <= val_months,
        "months with validation hours: "
        + ", ".join(calendar.month_abbr[m] for m in sorted(val_months)),
    )
    val_ref = np.mean([runs[REFERENCE][s].best_val for s in paired])
    val_best = np.mean([runs[BEST][s].best_val for s in paired])
    rep.value(f"validation NSE, {REFERENCE} (also Section S4)", float(val_ref), 2, "0.75")
    rep.value(f"validation NSE, {BEST} (also Section S4)", float(val_best), 2, "0.69")
    rep.count(f"seeds with {REFERENCE} ahead in validation (of {len(paired)})", len(ahead), 9)
    for config, (mean_text, sd_text) in zip(
        VALIDATION, (("0.75", "0.03"), ("0.73", "0.04"), ("0.69", "0.03")), strict=True
    ):
        mean_val, sd_val = val_stats[config]
        rep.value(f"Table S4 NSE, {config}, mean", mean_val, 2, mean_text)
        rep.value(f"Table S4 NSE, {config}, population sd", sd_val, 2, sd_text)
    test_ref = np.mean([runs[REFERENCE][s].nse for s in paired])
    test_best = np.mean([runs[BEST][s].nse for s in paired])
    rep.flag(
        "ordering reversed between the test and the validation windows",
        test_best > test_ref and val_ref > val_best,
        f"test {test_best:.6f} against {test_ref:.6f}; validation {val_best:.6f} against"
        f" {val_ref:.6f} ({BEST} against {REFERENCE})",
    )

    # ---- checks of the input
    print()
    print("Checks")
    print(
        f"  Runs read: {sum(len(f) for f in runs.values())}"
        f" ({', '.join(f'{c} {len(f)}' for c, f in runs.items())})"
    )
    print(
        "  Observed series rebuilt from the dataset equal to target of every predictions.npz: yes"
    )
    print(
        f"  Sum of the monthly gains {total_gain:+.6f} against the mean difference of the NSE of"
        f" results.json {json_gain:+.6f} (|difference| {abs(total_gain - json_gain):.1e})"
    )
    print(
        f"  Statements of the manuscript: {rep.compared - len(rep.differs)} of {rep.compared}"
        " match the data."
    )
    for label in rep.differs:
        print(f"    DIFFERS: {label}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "month",
                    "scored_hours",
                    "q_obs_mean_m3s",
                    "share_of_nse_denominator",
                    "mean_gain_nse",
                    "share_of_mean_gain",
                    "seed_pairs_with_gain",
                ]
            )
            for month, hours, q_mean, share_den, gain, positive in rows:
                writer.writerow(
                    [
                        month,
                        hours,
                        repr(q_mean),
                        repr(share_den),
                        repr(gain),
                        repr(gain / total_gain),
                        positive,
                    ]
                )
        print(f"\nWrote {display_path(args.csv)}")

    return 1 if args.strict and rep.differs else 0


if __name__ == "__main__":
    raise SystemExit(main())
