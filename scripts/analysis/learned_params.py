#!/usr/bin/env python
"""Reproduce the learned physical parameters reported in Sections 4.5 and 5.3 of the manuscript.

For every configuration with a differentiable physical layer, the script prints the mean, sample
standard deviation (divisor n - 1), minimum and maximum across seeds of each learned parameter,
with its initial value, its admissible range and the closest approach of any seed to a bound.
It then checks the statements of the manuscript about these parameters (Section 3.4 for the
constant baseflow of the pure-physics baselines, Section 4.5 and Figure 7, Section 5.3) and
reports any difference.

Source. ``scripts/train.py`` reloads the best checkpoint of each run (``best_model.pt``, selected
on the validation NSE) before the test evaluation and stores ``model.get_learned_params()`` under
``learned_params`` in ``results.json``. The values read here are therefore those of the
checkpoints that produced the test predictions, and neither torch nor the checkpoints are needed.

Parameters. Each learnable parameter except the baseflow is a sigmoid of an unconstrained
variable theta, value = lower + (upper - lower) * sigmoid(theta), and all have a deterministic
initial value (``src/ttd_scs_lstm/models/models.py`` and ``topmodel_diff.py``):

    key in results.json     name    unit   range           initial value
    lambda_scs              lambda  -      0.01 to 0.40    0.205 (theta = 0)
    tc_scale                s_tc    -      0.1 to 5.0      1.0
    sigma                   sigma   h      0.5 to 15.0     3.0
    topmodel_m              m       mm     5 to 200        30
    topmodel_T_0            T0      mm/h   0.001 to 0.5    0.02
    topmodel_sigmoid_temp   tau     mm     0.5 to 50       5.0
    baseflow_m3s            Qb      m3/s   softplus, > 0   0.693 (softplus(0) = ln 2)

The constant baseflow exists only in the two pure-physics baselines. The initial saturation
deficit ``topmodel_S_bar_init_fixed`` (70 mm) is a fixed buffer, not a parameter. In the
routing-free variants LSTM_SCS and LSTM_Topmodel the unit hydrograph is replaced by an impulse,
so tc_scale and sigma receive no gradient, and in LSTM_DUH_Fixed they are fixed by design: the
script lists them as not learned and checks that they keep their initial values.
LSTM_DUH_Manning_Fixed, run alongside the grid but not reported in the manuscript, is left out.

Closeness to a bound is the distance of a seed to the nearest bound in percent of the admissible
range, also given as |theta|. A seed counts as at a bound when that distance is below
``--bound-tol`` percent (default 1, the smallest round value under which every seed of the
parameters that Section 4.5 calls saturated in the pure-physics baselines counts as saturated).

Values are printed unrounded and rounded half away from zero from the unrounded value: to two
decimals and, where the manuscript quotes another precision, also to that precision.

Reads ``outputs/grade_sim`` (``results.json`` only) and writes nothing unless ``--csv`` is given.

Usage, from the repository root:
    python scripts/analysis/learned_params.py
    python scripts/analysis/learned_params.py --per-seed  # value of every seed
    python scripts/analysis/learned_params.py --csv       # outputs/tables/learned_params.csv
    python scripts/analysis/learned_params.py --grade outputs/retrain --strict
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
DEFAULT_CSV = ROOT / "outputs" / "tables" / "learned_params.csv"


@dataclass(frozen=True)
class Param:
    """A physical parameter as stored under ``learned_params`` in results.json."""

    key: str
    name: str
    unit: str
    lower: float | None  # None: no upper or lower sigmoid bound (softplus baseflow)
    upper: float | None
    init: float


PARAMS = {
    p.key: p
    for p in (
        Param("lambda_scs", "lambda", "-", 0.01, 0.40, 0.205),
        Param("topmodel_m", "m", "mm", 5.0, 200.0, 30.0),
        Param("topmodel_T_0", "T0", "mm/h", 0.001, 0.5, 0.02),
        Param("topmodel_sigmoid_temp", "tau", "mm", 0.5, 50.0, 5.0),
        Param("tc_scale", "s_tc", "-", 0.1, 5.0, 1.0),
        Param("sigma", "sigma", "h", 0.5, 15.0, 3.0),
        Param("baseflow_m3s", "Qb", "m3/s", None, None, math.log(2.0)),
    )
}

SCS = ("lambda_scs",)
TOPMODEL = ("topmodel_m", "topmodel_T_0", "topmodel_sigmoid_temp")
ROUTING = ("tc_scale", "sigma")
BASEFLOW = ("baseflow_m3s",)


@dataclass(frozen=True)
class Config:
    """A configuration of the grid with physical parameters."""

    name: str
    role: str
    learned: tuple[str, ...]
    inert: tuple[str, ...] = ()  # reported in results.json but fixed or without gradient
    hybrid: bool = True  # has an LSTM


CONFIGS = (
    Config("LSTM_DUH", "rung 5", ROUTING),
    Config("LSTM_DUH_Fixed", "rung 6, routing fixed", (), ROUTING),
    Config("LSTM_DUH_SCS", "rung 7", SCS + ROUTING),
    Config("LSTM_DUH_SCS_PeOnly", "rung 8", SCS + ROUTING),
    Config("LSTM_DUH_Topmodel", "rung 9", TOPMODEL + ROUTING),
    Config("LSTM_DUH_Topmodel_PeOnly", "rung 10, best hybrid", TOPMODEL + ROUTING),
    Config("LSTM_DUH_Topmodel_Baseflow", "rung 11", TOPMODEL + ROUTING),
    Config("Phys_DUH_SCS", "rung 12, physics only", SCS + ROUTING + BASEFLOW, hybrid=False),
    Config(
        "Phys_DUH_Topmodel", "rung 13, physics only", TOPMODEL + ROUTING + BASEFLOW, hybrid=False
    ),
    Config("LSTM_SCS", "auxiliary, no routing", SCS, ROUTING),
    Config("LSTM_Topmodel", "auxiliary, no routing", TOPMODEL, ROUTING),
    Config("LSTM_DUH_Manning", "auxiliary, Section S2", ROUTING),
)
CONFIG_BY_NAME = {c.name: c for c in CONFIGS}
FIXED_DEFICIT_MM = 70.0  # topmodel_S_bar_init_fixed
INERT_TOL = 1e-6  # float32 round-off of the initial values


# ---------------------------------------------------------------------------- input


def seed_folders(grade: Path) -> list[tuple[int, Path]]:
    """Folders seedNN of the grid, by increasing seed."""
    found = []
    for path in grade.glob("seed*"):
        match = re.fullmatch(r"seed(\d+)", path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def load_params(grade: Path, config: str) -> dict[int, dict[str, float]]:
    """``learned_params`` of every completed run of one configuration, by seed.

    A seed folder without results.json is skipped: seeds 44 and 48 of Phys_DUH_Topmodel are
    interrupted runs that keep only config.json.
    """
    runs = {}
    for seed, folder in seed_folders(grade):
        found = sorted((folder / config).glob("*/results.json"))
        if not found:
            continue
        if len(found) > 1:
            raise RuntimeError(f"more than one results.json under {folder / config}")
        stored = json.loads(found[0].read_text(encoding="utf-8")).get("learned_params") or {}
        runs[seed] = {key: float(value) for key, value in stored.items()}
    return runs


# ---------------------------------------------------------------------------- statistics


@dataclass
class Stats:
    """Distribution of one parameter of one configuration across seeds."""

    seeds: list[int]
    values: np.ndarray

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def sd(self) -> float:
        return float(self.values.std(ddof=1)) if self.values.size > 1 else float("nan")

    @property
    def lo(self) -> float:
        return float(self.values.min())

    @property
    def hi(self) -> float:
        return float(self.values.max())


@dataclass
class Approach:
    """Distance of the value of one seed to the nearest bound of the admissible range."""

    config: str
    param: str
    seed: int
    value: float
    side: str  # "lower" or "upper"
    percent: float  # distance to that bound, in percent of the range
    theta: float  # unconstrained variable, value = lower + (upper - lower) * sigmoid(theta)


def stats_of(runs: dict[int, dict[str, float]], key: str) -> Stats:
    """Values of one parameter over the seeds that report it."""
    seeds = [seed for seed in sorted(runs) if key in runs[seed]]
    return Stats(seeds, np.array([runs[seed][key] for seed in seeds], dtype=float))


def approach(config: str, key: str, seed: int, value: float) -> Approach | None:
    """Distance of one value to the nearest bound; None for an unbounded parameter."""
    param = PARAMS[key]
    if param.lower is None or param.upper is None:
        return None
    width = param.upper - param.lower
    frac = (value - param.lower) / width
    side, dist = ("lower", frac) if frac <= 0.5 else ("upper", 1.0 - frac)
    clipped = min(max(frac, 1e-12), 1.0 - 1e-12)
    theta = math.log(clipped / (1.0 - clipped))
    return Approach(config, param.name, seed, value, side, 100.0 * dist, theta)


def closest(config: str, key: str, stats: Stats) -> Approach | None:
    """Seed that comes closest to a bound."""
    found = [approach(config, key, s, v) for s, v in zip(stats.seeds, stats.values, strict=True)]
    found = [a for a in found if a is not None]
    return min(found, key=lambda a: a.percent) if found else None


# ---------------------------------------------------------------------------- formatting


def round_half_up(value: float, decimals: int) -> Decimal:
    """Round half away from zero, from the shortest repr of the unrounded value."""
    step = Decimal(1).scaleb(-decimals)
    rounded = Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_UP)
    return abs(rounded) if rounded == 0 else rounded  # no negative zero


def number_list(values: list[int]) -> str:
    """'42 to 51' for a run of consecutive integers, else a comma-separated list."""
    if len(values) > 2 and values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]} to {values[-1]}"
    return ", ".join(str(v) for v in values)


def bound_text(param: Param) -> str:
    """Admissible range as printed in the table."""
    if param.lower is None or param.upper is None:
        return "softplus > 0"
    return f"{param.lower:g} to {param.upper:g}"


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

    def span(
        self, label: str, values: dict[str, float], decimals: int, low: str, high: str
    ) -> None:
        """'between low and high': rounded minimum and maximum equal the two text values."""
        rounded = {name: round_half_up(v, decimals) for name, v in values.items()}
        ok = min(rounded.values()) == Decimal(low) and max(rounded.values()) == Decimal(high)
        status = self._status(ok, label)
        print(f"  {status:<7} {label} (text: {low} to {high})")
        for name, v in values.items():
            print(
                f"            {name:<52} {v:>10.6g}  2 dp {round_half_up(v, 2)}"
                f"  {decimals} dp {rounded[name]}"
            )

    def flag(self, label: str, ok: bool, detail: str) -> None:
        """A qualitative statement checked by a condition on the data."""
        status = self._status(ok, label)
        print(f"  {status:<7} {label}")
        for line in detail.splitlines():
            print(f"            {line}")

    @staticmethod
    def info(text: str) -> None:
        for line in text.splitlines():
            print(f"            {line}")


# ---------------------------------------------------------------------------- report sections


def print_tables(
    runs: dict[str, dict[int, dict[str, float]]], tol: float, per_seed: bool
) -> list[Approach]:
    """Summary table per configuration; returns the seeds found at a bound."""
    at_bound = []
    for config in CONFIGS:
        found = runs[config.name]
        seeds = sorted(found)
        print()
        print(f"{config.name} ({config.role}; seeds {number_list(seeds)})")
        print(
            f"  {'param':<6} {'unit':<5} {'mean':>10} {'(2 dp)':>7} {'sd':>11} {'min':>10}"
            f" {'max':>10} {'initial':>7}  {'range':<13} nearest bound (seed)"
        )
        for key in config.learned:
            param, stats = PARAMS[key], stats_of(found, key)
            near = closest(config.name, key, stats)
            near_text = "none"
            if near is not None:
                near_text = f"{near.side}, {near.percent:.2f}% ({near.seed})"
                at_bound.extend(
                    a
                    for s, v in zip(stats.seeds, stats.values, strict=True)
                    if (a := approach(config.name, key, s, v)) is not None and a.percent < tol
                )
            print(
                f"  {param.name:<6} {param.unit:<5} {stats.mean:>10.6g}"
                f" {round_half_up(stats.mean, 2)!s:>7} {stats.sd:>11.6g} {stats.lo:>10.6g}"
                f" {stats.hi:>10.6g} {param.init:>7.4g}  {bound_text(param):<13} {near_text}"
            )
        for key in config.inert:
            param, stats = PARAMS[key], stats_of(found, key)
            kept = bool(np.all(np.abs(stats.values - param.init) <= INERT_TOL))
            why = "fixed" if not config.learned else "no gradient"
            print(
                f"  {param.name:<6} {param.unit:<5} not learned ({why}); every seed at the initial"
                f" value {param.init:g}: {'yes' if kept else 'NO'} (seeds from {stats.lo:.6g} to"
                f" {stats.hi:.6g})"
            )
        if per_seed:
            keys = config.learned + config.inert
            print(f"  {'seed':<6}" + "".join(f" {PARAMS[k].name:>11}" for k in keys))
            for seed in seeds:
                values = "".join(f" {found[seed][k]:>11.6g}" for k in keys)
                print(f"  {seed:<6}{values}")
    return at_bound


def check_statements(
    runs: dict[str, dict[int, dict[str, float]]], tol: float, at_bound: list[Approach]
) -> Report:
    """Statements of the manuscript about the learned parameters."""

    def st(config: str, key: str) -> Stats:
        return stats_of(runs[config], key)

    def pooled(configs: tuple[str, ...], key: str) -> tuple[float, int]:
        values = np.concatenate([st(c, key).values for c in configs])
        return float(values.mean()), int(values.size)

    rep = Report()
    scs, top, peo, bfl = (
        "LSTM_DUH_SCS",
        "LSTM_DUH_Topmodel",
        "LSTM_DUH_Topmodel_PeOnly",
        "LSTM_DUH_Topmodel_Baseflow",
    )

    print()
    print("Statements of the manuscript")
    print("(status, statement, unrounded value, two decimals, precision of the text, text)")

    print()
    print("Section 3.4: constant baseflow of the pure-physics baselines")
    rep.span(
        "baseflow converges to about 24 to 25 m3/s (seed mean)",
        {c: st(c, "baseflow_m3s").mean for c in ("Phys_DUH_SCS", "Phys_DUH_Topmodel")},
        0,
        "24",
        "25",
    )

    print()
    print("Section 4.5 (values of LSTM_DUH_SCS for lambda, LSTM_DUH_Topmodel for m, T0 and tau,")
    print("both for s_tc and sigma; Figure 7 shows the same runs)")
    rep.value(
        "lambda, LSTM_DUH_SCS, seed mean (also Section 5.3)", st(scs, "lambda_scs").mean, 2, "0.07"
    )
    lam_noroute = st("LSTM_SCS", "lambda_scs")
    rep.value("lambda without routing, LSTM_SCS, seed mean", lam_noroute.mean, 2, "0.21")
    init = PARAMS["lambda_scs"].init
    rep.flag(
        "lambda of LSTM_SCS indistinguishable from its initialization",
        lam_noroute.lo <= init <= lam_noroute.hi,
        f"seeds range from {lam_noroute.lo:.6g} to {lam_noroute.hi:.6g}; initial value {init:g}",
    )
    lam_phys = st("Phys_DUH_SCS", "lambda_scs")
    rep.value("lambda, Phys_DUH_SCS, seed mean (upper bound 0.40)", lam_phys.mean, 2, "0.40")
    tc_both, n_both = pooled((scs, top), "tc_scale")
    rep.value(f"s_tc, LSTM_DUH_SCS and LSTM_DUH_Topmodel, mean of {n_both} runs", tc_both, 1, "1.7")
    rep.info(
        f"by configuration: LSTM_DUH_SCS {st(scs, 'tc_scale').mean:.6g},"
        f" LSTM_DUH_Topmodel {st(top, 'tc_scale').mean:.6g}; initial value 1.0"
    )
    sigma_both, n_both = pooled((scs, top), "sigma")
    rep.value(
        f"sigma (h), LSTM_DUH_SCS and LSTM_DUH_Topmodel, mean of {n_both} runs",
        sigma_both,
        1,
        "5.2",
    )
    rep.info(
        f"by configuration: LSTM_DUH_SCS {st(scs, 'sigma').mean:.6g},"
        f" LSTM_DUH_Topmodel {st(top, 'sigma').mean:.6g}; initial value 3.0 h"
    )
    interior = {PARAMS[k].name: closest(top, k, st(top, k)) for k in TOPMODEL}
    rep.flag(
        f"TOPMODEL parameters of LSTM_DUH_Topmodel in the interior of their ranges (no seed"
        f" within {tol:g}%)",
        all(a is not None and a.percent >= tol for a in interior.values()),
        "nearest bound: "
        + ", ".join(
            f"{name} {a.side} at {a.percent:.2f}% (seed {a.seed})"
            for name, a in interior.items()
            if a is not None
        ),
    )
    rep.value("m (mm), LSTM_DUH_Topmodel, seed mean", st(top, "topmodel_m").mean, 0, "27")
    t0_top = st(top, "topmodel_T_0")
    rep.value("T0 (mm/h), LSTM_DUH_Topmodel, seed mean", t0_top.mean, 2, "0.07")
    rep.value("T0 (mm/h), LSTM_DUH_Topmodel, lowest seed", t0_top.lo, 3, "0.015")
    rep.value("T0 (mm/h), LSTM_DUH_Topmodel, highest seed", t0_top.hi, 2, "0.17")
    spread = {
        PARAMS[k].name: (st(top, k).sd / st(top, k).mean, st(top, k).hi / st(top, k).lo)
        for k in TOPMODEL + ROUTING
    }
    widest = {max(spread, key=lambda n: spread[n][i]) for i in (0, 1)}
    rep.flag(
        "T0 has the largest spread across seeds of the LSTM_DUH_Topmodel parameters",
        widest == {"T0"},
        "coefficient of variation (sd / mean) and max / min by parameter:\n"
        + ", ".join(f"{n} {cv:.3f} and {r:.2f}" for n, (cv, r) in spread.items()),
    )
    tau_top = st(top, "topmodel_sigmoid_temp")
    rep.value("tau (mm), LSTM_DUH_Topmodel, seed mean", tau_top.mean, 1, "7.9")
    tau_near = closest(top, "topmodel_sigmoid_temp", tau_top)
    assert tau_near is not None  # tau is bounded
    rep.flag(
        "tau of LSTM_DUH_Topmodel well inside the 0.5 to 50 mm range",
        tau_near.percent >= tol,
        f"seeds range from {tau_top.lo:.6g} to {tau_top.hi:.6g} mm; nearest bound:"
        f" {tau_near.side} at {tau_near.percent:.2f}% of the range (seed {tau_near.seed})",
    )
    rep.value("m (mm), LSTM_DUH_Topmodel_PeOnly, seed mean", st(peo, "topmodel_m").mean, 0, "33")
    t0_peo = st(peo, "topmodel_T_0")
    rep.value("T0 (mm/h), LSTM_DUH_Topmodel_PeOnly, seed mean", t0_peo.mean, 3, "0.021")
    rep.value(
        "tau (mm), LSTM_DUH_Topmodel_PeOnly, mean (also Section 5.3)",
        st(peo, "topmodel_sigmoid_temp").mean,
        1,
        "2.4",
    )
    rep.value("T0 (mm/h), LSTM_DUH_Topmodel_PeOnly, lowest seed", t0_peo.lo, 3, "0.010")
    rep.value("T0 (mm/h), LSTM_DUH_Topmodel_PeOnly, highest seed", t0_peo.hi, 3, "0.033")
    m_peo = st(peo, "topmodel_m")
    rep.info(
        "m and T0 of LSTM_DUH_Topmodel_PeOnly against their initial values (30 mm, 0.02 mm/h):"
        f"\n{100 * (m_peo.mean / 30.0 - 1):+.1f}% and {100 * (t0_peo.mean / 0.02 - 1):+.1f}%"
        f" (LSTM_DUH_Topmodel: {100 * (st(top, 'topmodel_m').mean / 30.0 - 1):+.1f}% and"
        f" {100 * (t0_top.mean / 0.02 - 1):+.1f}%)"
    )
    sd_ratio = t0_top.sd / t0_peo.sd
    range_ratio = (t0_top.hi - t0_top.lo) / (t0_peo.hi - t0_peo.lo)
    rep.value("T0 spread ratio, LSTM_DUH_Topmodel / PeOnly, sd (also 5.3)", sd_ratio, 0, "7")
    rep.value("T0 spread ratio, LSTM_DUH_Topmodel / PeOnly, max - min", range_ratio, 0, "7")
    t0_bfl = st(bfl, "topmodel_T_0")
    rep.info(
        f"for reference, LSTM_DUH_Topmodel_Baseflow (shortcut open, Qb fed to the LSTM): T0 mean"
        f" {t0_bfl.mean:.6g},\nfrom {t0_bfl.lo:.6g} to {t0_bfl.hi:.6g}; sd ratio"
        f" LSTM_DUH_Topmodel / Baseflow {t0_top.sd / t0_bfl.sd:.3g}"
    )
    lines, all_saturated = [], True
    for config, key in (
        ("Phys_DUH_SCS", "sigma"),
        ("Phys_DUH_SCS", "lambda_scs"),
        ("Phys_DUH_Topmodel", "sigma"),
        ("Phys_DUH_Topmodel", "topmodel_T_0"),
    ):
        stats = st(config, key)
        found = [
            approach(config, key, s, v) for s, v in zip(stats.seeds, stats.values, strict=True)
        ]
        worst = max(found, key=lambda a: a.percent)
        ok = all(a.side == "upper" and a.percent < tol for a in found)
        all_saturated &= ok
        lines.append(
            f"{config} {PARAMS[key].name}: seed mean {stats.mean:.6g} (upper bound"
            f" {PARAMS[key].upper:g}); farthest seed {worst.seed} at {worst.percent:.2f}% of the"
            f" range, |theta| {abs(worst.theta):.2f}"
        )
    rep.flag(
        f"pure-physics baselines: sigma, lambda and T0 at their upper bounds (every seed within"
        f" {tol:g}% of the range)",
        all_saturated,
        "\n".join(lines),
    )

    print()
    print("Section 5.3 (LSTM_DUH_SCS, LSTM_DUH_Topmodel and the best hybrid)")
    trio = (scs, top, peo)
    rep.span(
        "s_tc rises from 1.0 to between 1.7 and 1.9 (seed means)",
        {c: st(c, "tc_scale").mean for c in trio},
        1,
        "1.7",
        "1.9",
    )
    rep.span(
        "sigma rises from 3 to between 4.7 and 5.3 h (seed means)",
        {c: st(c, "sigma").mean for c in trio},
        1,
        "4.7",
        "5.3",
    )
    hybrid_hits = [a for a in at_bound if CONFIG_BY_NAME[a.config].hybrid]
    chain_hits = [a for a in hybrid_hits if CONFIG_BY_NAME[a.config].role.startswith("rung")]
    nearest = sorted(
        (
            a
            for c in CONFIGS
            if c.hybrid
            for k in c.learned
            if (a := closest(c.name, k, stats_of(runs[c.name], k))) is not None
        ),
        key=lambda a: a.percent,
    )[:4]
    rep.flag(
        f"no parameter of any hybrid pinned at the edge of its range (no seed within {tol:g}%)",
        not hybrid_hits,
        "closest approaches among the hybrids (configuration, parameter, seed, value):\n"
        + "\n".join(
            f"{a.config} {a.param} seed {a.seed}: {a.value:.6g}, {a.percent:.2f}% of the range"
            f" from the {a.side} bound, |theta| {abs(a.theta):.2f}"
            for a in nearest
        )
        + f"\nhybrids of the attribution chain (rungs 5 to 11) at a bound: {len(chain_hits)}",
    )
    return rep


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
        "--bound-tol",
        type=float,
        default=1.0,
        metavar="PERCENT",
        help="distance to a bound, in percent of the range, below which a seed counts as at "
        "the bound (default: 1)",
    )
    parser.add_argument("--per-seed", action="store_true", help="also list the value of every seed")
    parser.add_argument(
        "--csv",
        nargs="?",
        type=Path,
        const=DEFAULT_CSV,
        default=None,
        metavar="PATH",
        help="also write every seed's values to a CSV file "
        "(default path: outputs/tables/learned_params.csv)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 if any statement differs from the data",
    )
    args = parser.parse_args()

    runs = {config.name: load_params(args.grade, config.name) for config in CONFIGS}
    missing = [name for name, found in runs.items() if not found]
    if missing:
        parser.error(f"no completed runs in {args.grade} for: {', '.join(missing)}")
    absent = [
        f"{config.name} seed {seed}: {key}"
        for config in CONFIGS
        for seed, values in runs[config.name].items()
        for key in config.learned + config.inert
        if key not in values
    ]
    if absent:
        parser.error("learned_params without " + "; ".join(absent))

    print("Learned physical parameters (results.json, best checkpoint of each run)")
    print(f"Grid: {display_path(args.grade)}")
    print("mean and sample sd (divisor n - 1) across seeds; |theta|: unconstrained variable of the")
    print("sigmoid bound, value = lower + (upper - lower) * sigmoid(theta)")
    at_bound = print_tables(runs, args.bound_tol, args.per_seed)

    rep = check_statements(runs, args.bound_tol, at_bound)

    # ---- checks of the input
    deficits = [
        values.get("topmodel_S_bar_init_fixed", float("nan"))
        for name, found in runs.items()
        if "Topmodel" in name and not name.startswith("Phys")
        for values in found.values()
    ]
    print()
    print("Checks")
    print(f"  Runs read: {sum(len(found) for found in runs.values())}")
    for config in CONFIGS:
        print(f"    {config.name:<28} seeds {number_list(sorted(runs[config.name]))}")
    same = all(abs(d - FIXED_DEFICIT_MM) <= INERT_TOL for d in deficits)
    print(
        f"  Initial deficit of the TOPMODEL hybrids: {FIXED_DEFICIT_MM:g} mm in all"
        f" {len(deficits)} runs: {'yes' if same else 'NO'}"
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
                ["configuration", "role", "seed", "key", "name", "unit", "value", "learned"]
            )
            for config in CONFIGS:
                for seed in sorted(runs[config.name]):
                    for key in config.learned + config.inert:
                        param = PARAMS[key]
                        writer.writerow(
                            [
                                config.name,
                                config.role,
                                seed,
                                key,
                                param.name,
                                param.unit,
                                repr(runs[config.name][seed][key]),
                                key in config.learned,
                            ]
                        )
        print(f"\nWrote {display_path(args.csv)}")

    return 1 if args.strict and rep.differs else 0


if __name__ == "__main__":
    raise SystemExit(main())
