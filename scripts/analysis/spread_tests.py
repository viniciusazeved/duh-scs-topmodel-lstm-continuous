#!/usr/bin/env python
"""Reproduce the seed-to-seed spread analysis of Section 4.3 and Supplementary Figure S3.

Section 4.3 compares the spread of the test scores across the ten training seeds of the pure
distributed LSTM with that of the two TOPMODEL hybrids (LSTM_DUH_Topmodel and
LSTM_DUH_Topmodel_PeOnly) and of the fixed-routing control (LSTM_DUH_Fixed). The script prints:

1. the sample standard deviation across seeds (divisor n - 1, as in Table 3 and Figure S3) of
   the NSE and of the logNSE of these configurations, and the spread ratio sd(LSTM) / sd(other);
2. the configuration with the largest NSE spread among those with mean NSE above 0.70, over the
   13 rungs of Table 3 and over all configurations of the grid;
3. the Brown-Forsythe test of equal spread of the NSE (one-way ANOVA on the absolute deviations
   from the group medians, that is, Levene's test centred on the median) of the LSTM against
   each TOPMODEL hybrid;
4. the 95% bootstrap percentile interval of the spread ratio sd(LSTM) / sd(LSTM_DUH_Topmodel_PeOnly)
   of the NSE, with a fixed configuration: 20,000 resamples, the ten seeds of each configuration
   resampled independently (not paired), random generator numpy.random.default_rng(0);
5. the bar values of Supplementary Figure S3 and the spread of LSTM_PET (Section 4.4).

On item 4: the interval printed in the manuscript, [0.68, 3.69], is reproduced only up to Monte
Carlo error, because percentile limits depend on the random stream. The script repeats the
bootstrap for the bootstrap seeds 0 to N - 1 (``--mc-seeds``, default 200) and prints the range
of the two limits, which contains the printed interval; ``--mc-seeds 0`` skips this.

NSE is ``test_metrics.nse`` of each ``results.json``; logNSE, the NSE of ln(Q + 0.01 mean(Q_obs)),
is recomputed from ``predictions.npz`` on the stored float32 series over the finite pairs, as in
Table 3 (``scripts/analysis/table3.py``). The spread ratio does not depend on the divisor of the
standard deviation, because both groups have ten seeds. With ``--extended`` the script also prints
tests that the manuscript does not report: Brown-Forsythe on the logNSE and for LSTM_DUH_Fixed and
LSTM_PET, and the bootstrap interval for LSTM_DUH_Topmodel.

Values are printed unrounded and, in parentheses, rounded half away from zero from the unrounded
value (two decimals); p-values as the text prints them (two decimals at or above 0.1, three
below).

Reads ``outputs/grade_sim`` and writes nothing.

Usage, from the repository root:
    python scripts/analysis/spread_tests.py
    python scripts/analysis/spread_tests.py --per-seed --extended
    python scripts/analysis/spread_tests.py --grade outputs/retrain --strict
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRADE = ROOT / "outputs" / "grade_sim"

REFERENCE = "LSTM"
# Section 4.3: sample standard deviation of the NSE and of the logNSE, as printed in the text.
SPREAD_TEXT = {
    "LSTM": ("0.11", "0.13"),
    "LSTM_DUH_Topmodel": ("0.06", "0.04"),
    "LSTM_DUH_Topmodel_PeOnly": ("0.06", "0.06"),
    "LSTM_DUH_Fixed": ("0.06", "0.11"),
}
# Section 4.3: Brown-Forsythe p-values of the NSE (LSTM against each configuration).
BROWN_FORSYTHE_TEXT = {"LSTM_DUH_Topmodel": "0.33", "LSTM_DUH_Topmodel_PeOnly": "0.28"}
# Section 4.3: bootstrap 95% interval of sd(LSTM) / sd(LSTM_DUH_Topmodel_PeOnly), NSE.
BOOTSTRAP_CONFIG = "LSTM_DUH_Topmodel_PeOnly"
BOOTSTRAP_TEXT = ("0.68", "3.69")
NSE_THRESHOLD = 0.70  # "the highest among configurations with mean NSE exceeding 0.70"

# Supplementary Figure S3: configuration, bar label and printed value.
FIGURE_S3 = (
    ("LSTM", "LSTM (neural)", "0.11"),
    ("LSTM_DUH", "+ DUH", "0.08"),
    ("LSTM_DUH_Fixed", "+ DUH (fixed)", "0.06"),
    ("LSTM_DUH_SCS", "+ SCS-CN", "0.10"),
    ("LSTM_DUH_Topmodel", "+ TOPMODEL", "0.06"),
    ("LSTM_DUH_Topmodel_Baseflow", "+ TOPMODEL + baseflow", "0.06"),
    ("LSTM_DUH_Topmodel_PeOnly", "+ TOPMODEL (Pe-only)", "0.06"),
)

# The 13 rungs of the attribution chain (Table 3) and the auxiliary configurations of the grid.
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
AUXILIARY = (
    "LSTM_Topmodel",
    "LSTM_SCS",
    "LSTM_PET",
    "LSTM_DUH_Manning",
    "LSTM_DUH_Manning_Fixed",
)
# --extended: Brown-Forsythe tests not reported in the manuscript.
EXTENDED_TESTS = (
    ("nse", ("LSTM_DUH_Fixed", "LSTM_PET")),
    ("lognse", ("LSTM_DUH_Topmodel", "LSTM_DUH_Topmodel_PeOnly", "LSTM_DUH_Fixed", "LSTM_PET")),
)


@dataclass
class Scores:
    """Per-seed test scores of one configuration."""

    nse: dict[int, float]  # test_metrics.nse of results.json
    lognse: dict[int, float]  # recomputed from predictions.npz

    def array(self, metric: str, seeds: list[int] | None = None) -> np.ndarray:
        """Scores of ``metric`` ("nse" or "lognse"), by increasing seed or for ``seeds``."""
        values = getattr(self, metric)
        return np.array([values[seed] for seed in (seeds or sorted(values))], dtype=float)


@dataclass
class Tally:
    """Comparison of recomputed values with the values printed in the manuscript."""

    compared: int = 0
    mismatches: list[str] = field(default_factory=list)

    def check(self, label: str, printed: str, data: str) -> str:
        """Record one comparison and return 'ok' or 'DIFFERS'."""
        self.compared += 1
        if printed == data:
            return "ok"
        self.mismatches.append(f"{label}: manuscript {printed}, data {data}")
        return "DIFFERS"


# ---------------------------------------------------------------------------- statistics


def log_nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """NSE of ln(Q + 0.01 mean(Q_obs)) (``_lognse`` of scripts/consolida_sim.py)."""
    eps = 0.01 * np.nanmean(obs)
    log_sim, log_obs = np.log(sim + eps), np.log(obs + eps)
    den = np.sum((log_obs - log_obs.mean()) ** 2)
    return float(1 - np.sum((log_sim - log_obs) ** 2) / den) if den > 0 else float("nan")


def sample_sd(values: np.ndarray) -> float:
    """Sample standard deviation (divisor n - 1), as in Table 3 and Figure S3."""
    return float(np.std(values, ddof=1))


def brown_forsythe(*groups: np.ndarray) -> tuple[float, int, int, float]:
    """Brown-Forsythe test of equal spread: F, its two degrees of freedom and the p-value.

    One-way ANOVA F test on the absolute deviations of each group from its own median.
    """
    deviations = [np.abs(g - np.median(g)) for g in groups]
    k, n = len(deviations), sum(d.size for d in deviations)
    grand = np.concatenate(deviations).mean()
    between = sum(d.size * (d.mean() - grand) ** 2 for d in deviations)
    within = sum(((d - d.mean()) ** 2).sum() for d in deviations)
    f_stat = float((between / (k - 1)) / (within / (n - k)))
    return f_stat, k - 1, n - k, float(stats.f.sf(f_stat, k - 1, n - k))


def bootstrap_ratio_interval(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, int]:
    """95% percentile bootstrap interval of sd(a) / sd(b), resampling a and b independently.

    Each replicate draws len(a) values of a and len(b) values of b with replacement (first all
    the indices of a, then all those of b, from numpy.random.default_rng(seed)). Replicates with
    sd(b) = 0 are dropped; their number is returned.
    """
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, a.size, size=(n_boot, a.size))
    idx_b = rng.integers(0, b.size, size=(n_boot, b.size))
    sd_a = a[idx_a].std(axis=1, ddof=1)
    sd_b = b[idx_b].std(axis=1, ddof=1)
    ok = sd_b > 0
    low, high = np.percentile(sd_a[ok] / sd_b[ok], [2.5, 97.5])
    return float(low), float(high), int(np.sum(~ok))


# ---------------------------------------------------------------------------- input


def seed_folders(grade: Path) -> list[tuple[int, Path]]:
    """Folders seedNN of the grid, by increasing seed."""
    found = []
    for path in grade.glob("seed*"):
        match = re.fullmatch(r"seed(\d+)", path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def load_scores(grade: Path, config: str) -> tuple[Scores, list[int]]:
    """NSE and logNSE of the completed runs of one configuration, and the test-period lengths.

    A seed folder without results.json is skipped: seeds 44 and 48 of Phys_DUH_Topmodel are
    interrupted runs that keep only config.json.
    """
    nse, lognse, lengths = {}, {}, []
    for seed, folder in seed_folders(grade):
        found = sorted((folder / config).glob("*/results.json"))
        if not found:
            continue
        if len(found) > 1:
            raise RuntimeError(f"more than one results.json under {folder / config}")
        nse[seed] = float(json.loads(found[0].read_text(encoding="utf-8"))["test_metrics"]["nse"])
        with np.load(found[0].with_name("predictions.npz")) as npz:
            sim, obs = npz["pred"][:, 0], npz["target"][:, 0]
        ok = np.isfinite(sim) & np.isfinite(obs)
        lognse[seed] = log_nse(sim[ok], obs[ok])
        lengths.append(int(ok.sum()))
    return Scores(nse, lognse), lengths


# ---------------------------------------------------------------------------- formatting


def round_half_up(value: float, decimals: int = 2) -> Decimal:
    """Round half away from zero, from the shortest repr of the unrounded value."""
    step = Decimal(1).scaleb(-decimals)
    rounded = Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_UP)
    return abs(rounded) if rounded == 0 else rounded  # no negative zero


def fmt(value: float, decimals: int = 2) -> str:
    """Value rounded half away from zero, as text."""
    return f"{round_half_up(value, decimals)}"


def both(value: float) -> str:
    """Unrounded value followed by the rounded value in parentheses."""
    return f"{value:.6f} ({fmt(value)})"


def p_text(p: float) -> str:
    """p-value rounded as in the manuscript: two decimals at or above 0.1, three below."""
    return fmt(p, 2 if p >= 0.1 else 3)


def plural(count: int, noun: str) -> str:
    """'1 configuration', '2 configurations'."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


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


# ---------------------------------------------------------------------------- blocks


def print_spread(scores: dict[str, Scores], tally: Tally) -> None:
    """Item 1: sample standard deviations and spread ratios of Section 4.3."""
    print()
    print("1. Sample standard deviation across seeds (Section 4.3)")
    print(
        f"   {'Configuration':<25} {'n':>2}  {'mean NSE':<15}  {'sd NSE':<15} {'text':<5}"
        f"  {'sd logNSE':<15} text"
    )
    for name, (text_nse, text_lognse) in SPREAD_TEXT.items():
        nse, lognse = scores[name].array("nse"), scores[name].array("lognse")
        sd_nse, sd_lognse = sample_sd(nse), sample_sd(lognse)
        status_nse = tally.check(f"sd of the NSE of {name}", text_nse, fmt(sd_nse))
        status_log = tally.check(f"sd of the logNSE of {name}", text_lognse, fmt(sd_lognse))
        print(
            f"   {name:<25} {nse.size:>2}  {both(float(nse.mean())):<15}  {both(sd_nse):<15}"
            f" {text_nse:<5}  {both(sd_lognse):<15} {text_lognse:<5}  {status_nse}, {status_log}"
        )
    print("   Spread ratio sd(LSTM) / sd(configuration) ('approximately halved' for the NSE):")
    reference = scores[REFERENCE]
    for name in list(SPREAD_TEXT)[1:]:
        ratio_nse = sample_sd(reference.array("nse")) / sample_sd(scores[name].array("nse"))
        ratio_log = sample_sd(reference.array("lognse")) / sample_sd(scores[name].array("lognse"))
        print(f"     {name:<25} NSE {both(ratio_nse)}   logNSE {both(ratio_log)}")


def print_highest(scores: dict[str, Scores], tally: Tally) -> None:
    """Item 2: the LSTM has the largest NSE spread among configurations with mean NSE > 0.70."""
    print()
    print(f"2. Largest NSE spread among configurations with mean NSE above {NSE_THRESHOLD:.2f}")
    for label, names in (
        ("13 rungs of Table 3", CHAIN),
        ("all configurations of the grid", CHAIN + AUXILIARY),
    ):
        rows = []
        for name in names:
            nse = scores[name].array("nse")
            if nse.mean() > NSE_THRESHOLD:
                rows.append((sample_sd(nse), float(nse.mean()), name))
        rows.sort(reverse=True)
        top = rows[0][2]
        status = tally.check(f"largest NSE spread ({label})", REFERENCE, top)
        print(f"   {label}: {plural(len(rows), 'configuration')}; largest sd: {top} -> {status}")
        for sd, mean, name in rows:
            print(f"     {name:<28} sd {both(sd)}   mean NSE {both(mean)}")


def print_brown_forsythe(
    scores: dict[str, Scores], names: tuple[str, ...], metric: str, tally: Tally | None
) -> list[float]:
    """Item 3: Brown-Forsythe tests of LSTM against each configuration; returns |own - scipy|."""
    gaps = []
    reference = scores[REFERENCE]
    for name in names:
        seeds = sorted(set(reference.nse) & set(scores[name].nse))
        a, b = reference.array(metric, seeds), scores[name].array(metric, seeds)
        f_stat, df1, df2, p = brown_forsythe(a, b)
        gaps.append(abs(p - float(stats.levene(a, b, center="median").pvalue)))
        line = (
            f"   {metric:<6} LSTM against {name:<25} F({df1}, {df2}) = {f_stat:.6f};"
            f" p = {p:.6f} ({p_text(p)})"
        )
        printed = BROWN_FORSYTHE_TEXT.get(name) if metric == "nse" else None
        if tally is not None and printed is not None:
            status = tally.check(f"Brown-Forsythe p of the NSE, {name}", printed, p_text(p))
            line += f"; manuscript {printed} -> {status}"
        print(line)
    return gaps


def print_bootstrap(
    scores: dict[str, Scores], name: str, args: argparse.Namespace, tally: Tally | None
) -> None:
    """Item 4: bootstrap percentile interval of sd(LSTM) / sd(name) of the NSE.

    With a tally, the interval is compared with the manuscript and its Monte Carlo spread over
    the bootstrap seeds 0 to ``args.mc_seeds`` - 1 is printed.
    """
    reference = scores[REFERENCE]
    seeds = sorted(set(reference.nse) & set(scores[name].nse))
    a, b = reference.array("nse", seeds), scores[name].array("nse", seeds)
    low, high, dropped = bootstrap_ratio_interval(a, b, args.n_boot, args.boot_seed)
    print(f"   sd(LSTM) / sd({name}) of the NSE: {both(sample_sd(a) / sample_sd(b))}")
    print(
        f"   {args.n_boot:,} resamples, seeds resampled independently within each configuration,"
        f" numpy.random.default_rng({args.boot_seed})"
    )
    contains = "contains 1" if low < 1 < high else "does NOT contain 1"
    print(
        f"   95% percentile interval [{low:.6f}, {high:.6f}] ([{fmt(low)}, {fmt(high)}]),"
        f" {contains}; {plural(dropped, 'replicate')} with sd 0 dropped"
    )
    if tally is None:
        return
    printed = f"[{BOOTSTRAP_TEXT[0]}, {BOOTSTRAP_TEXT[1]}]"
    if args.mc_seeds <= 0:
        print(f"   manuscript {printed}; Monte Carlo range not computed (--mc-seeds 0)")
        return
    limits = np.array(
        [bootstrap_ratio_interval(a, b, args.n_boot, seed)[:2] for seed in range(args.mc_seeds)]
    )
    lows, highs = limits[:, 0], limits[:, 1]
    rounded_match = sum((fmt(lo), fmt(hi)) == BOOTSTRAP_TEXT for lo, hi in limits)
    low_range = (round_half_up(lows.min()), round_half_up(lows.max()))
    high_range = (round_half_up(highs.min()), round_half_up(highs.max()))
    printed_low, printed_high = (Decimal(text) for text in BOOTSTRAP_TEXT)
    inside = (
        low_range[0] <= printed_low <= low_range[1]
        and high_range[0] <= printed_high <= high_range[1]
    )
    status = tally.check(
        "bootstrap interval: manuscript limits within the Monte Carlo range",
        "within",
        "within" if inside else "outside",
    )
    print(
        f"   Monte Carlo spread over the bootstrap seeds 0 to {args.mc_seeds - 1}"
        f" ({args.n_boot:,} resamples each):"
    )
    print(
        f"     lower limit {lows.min():.4f} to {lows.max():.4f} ({low_range[0]} to {low_range[1]}),"
        f" median {np.median(lows):.4f}"
    )
    print(
        f"     upper limit {highs.min():.4f} to {highs.max():.4f}"
        f" ({high_range[0]} to {high_range[1]}), median {np.median(highs):.4f}"
    )
    print(
        f"     every interval contains 1: {'yes' if np.all((lows < 1) & (highs > 1)) else 'NO'};"
        f" {rounded_match} of {args.mc_seeds} round to {printed}"
    )
    print(
        f"   NOTE: the manuscript prints {printed}. The fixed configuration gives"
        f" [{fmt(low)}, {fmt(high)}], so the printed"
    )
    print(
        "   interval is reproduced only up to Monte Carlo error; its limits lie within the range"
        f" above -> {status}"
    )


def print_figure_s3(scores: dict[str, Scores], tally: Tally) -> None:
    """Item 5: bar values of Supplementary Figure S3 and the spread of LSTM_PET (Section 4.4)."""
    print()
    print("5. Supplementary Figure S3 (sample sd of the NSE; bar labels rounded to two decimals)")
    for name, label, printed in FIGURE_S3:
        sd = sample_sd(scores[name].array("nse"))
        status = tally.check(f"Figure S3 bar '{label}'", printed, fmt(sd))
        print(f"   {label:<22} {name:<27} {both(sd)}; figure {printed} -> {status}")
    sd_pet = sample_sd(scores["LSTM_PET"].array("nse"))
    sd_ref = sample_sd(scores[REFERENCE].array("nse"))
    status = tally.check(
        "LSTM_PET 'reduces the seed-to-seed spread'",
        "smaller",
        "smaller" if sd_pet < sd_ref else "not smaller",
    )
    print("   Section 4.4, LSTM_PET 'reduces the seed-to-seed spread' (no value in the text):")
    print(f"     sd of the NSE {both(sd_pet)} against {both(sd_ref)} for LSTM -> {status}")


def print_checks(
    scores: dict[str, Scores], lengths: list[int], gaps: list[float], tally: Tally
) -> None:
    """Consistency checks and the summary against the manuscript."""
    seed_sets: dict[tuple[int, ...], list[str]] = {}
    for name, found in scores.items():
        seed_sets.setdefault(tuple(sorted(found.nse)), []).append(name)
    population_differs = []
    for name, texts in SPREAD_TEXT.items():
        for metric, text in zip(("nse", "lognse"), texts, strict=True):
            population = fmt(float(np.std(scores[name].array(metric))))
            if population != text:
                population_differs.append(f"{name} {metric} {population} (text {text})")

    print()
    print("Checks")
    print(f"  Configurations read: {len(scores)}")
    for seeds, names in seed_sets.items():
        print(f"    seeds {number_list(list(seeds))}: {plural(len(names), 'configuration')}")
    print(f"  Finite test time steps per run: {', '.join(map(str, lengths))}")
    print(
        "  Brown-Forsythe, own ANOVA against scipy.stats.levene(center='median'):"
        f" max |difference in p| = {max(gaps):.1e}"
    )
    if population_differs:
        print("  Divisor of the standard deviation: the population divisor (n) would round to")
        for line in population_differs:
            print(f"    {line}")
        print("    so the text uses the sample divisor (n - 1), as stated for Table 3 and Fig. S3.")
    else:
        print("  Divisor of the standard deviation: sample and population divisors round alike.")
    print(
        f"  Manuscript: {tally.compared - len(tally.mismatches)} of {tally.compared} printed"
        " values match the values recomputed here."
    )
    for line in tally.mismatches:
        print(f"    DIFFERS: {line}")


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
        "--n-boot", type=int, default=20000, help="bootstrap resamples (default: 20000)"
    )
    parser.add_argument(
        "--boot-seed",
        type=int,
        default=0,
        help="seed of numpy.random.default_rng for the bootstrap (default: 0)",
    )
    parser.add_argument(
        "--mc-seeds",
        type=int,
        default=200,
        help="repeat the bootstrap for the seeds 0 to N - 1 to show its Monte Carlo spread "
        "(default: 200; 0 skips it)",
    )
    parser.add_argument(
        "--per-seed", action="store_true", help="also print the NSE and logNSE of every seed"
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="also print tests that the manuscript does not report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 if any value differs from the manuscript",
    )
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")

    loaded = {name: load_scores(args.grade, name) for name in CHAIN + AUXILIARY}
    missing = [name for name, (found, _) in loaded.items() if not found.nse]
    if missing:
        parser.error(f"no completed runs in {args.grade} for: {', '.join(missing)}")
    scores = {name: found for name, (found, _) in loaded.items()}
    lengths = sorted({n for _, found_lengths in loaded.values() for n in found_lengths})

    print("Seed-to-seed spread of the test scores (Section 4.3; Supplementary Figure S3)")
    print("Test period, telemetric product. Values unrounded and (rounded); 'text' is the value")
    print("printed in the manuscript.")
    print(f"Grid: {display_path(args.grade)}")

    tally = Tally()
    print_spread(scores, tally)
    print_highest(scores, tally)

    print()
    print("3. Brown-Forsythe test of equal NSE spread (ANOVA on |x - median|)")
    gaps = print_brown_forsythe(scores, tuple(BROWN_FORSYTHE_TEXT), "nse", tally)

    print()
    print("4. Bootstrap interval of the spread ratio")
    print_bootstrap(scores, BOOTSTRAP_CONFIG, args, tally)

    print_figure_s3(scores, tally)

    if args.extended:
        print()
        print("Extended output: tests NOT reported in the manuscript (unadjusted p-values)")
        for metric, names in EXTENDED_TESTS:
            gaps += print_brown_forsythe(scores, names, metric, None)
        print_bootstrap(scores, "LSTM_DUH_Topmodel", args, None)

    if args.per_seed:
        print()
        print(f"Per-seed scores (seeds {number_list(sorted(scores[REFERENCE].nse))})")
        for name in (*SPREAD_TEXT, "LSTM_PET"):
            for metric in ("nse", "lognse"):
                listed = " ".join(f"{v:+.4f}" for v in scores[name].array(metric))
                print(f"   {name:<25} {metric:<6} {listed}")

    print_checks(scores, lengths, gaps, tally)
    return 1 if args.strict and tally.mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
