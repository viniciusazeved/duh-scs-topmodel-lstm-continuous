#!/usr/bin/env python
"""Reproduce the low-flow signature diagnostics of Section 4.3 and Supplementary Section S11.

Over the test period (6,323 hourly time steps, telemetric forcing), the script prints:

- Supplementary Table S7: the modified low-flow log-tail bias (FLV, Supplementary Eq. S1) and
  the original low-flow signature of Yilmaz et al. (2008) (FLV_Y, Eq. S2) of the pure LSTM and
  of the two TOPMODEL hybrids, as mean and sample standard deviation across the seeds;
- the exact one-sided seed-paired Wilcoxon tests of the two hybrids against the pure LSTM: the
  reduction of the absolute signature error (Eqs. S1 and S2) and the gain in logNSE;
- the seed means of every signature for the configurations quoted in Sections 4.3 and 4.4 (the
  pure LSTM, the TOPMODEL hybrids, the baseflow variant and the two physics-only baselines),
  including the geometric mean of the simulated low-flow tail relative to the observed one;
- a comparison of each recomputed value with the value printed in the manuscript, and checks.

Definitions (Supplementary Section S11). The non-finite simulated-observed pairs are excluded,
each series is sorted independently and its k = floor(0.30 N) lowest values are kept: s_i
(simulated) and o_i (observed), discharge in m3/s.

- Eq. S1, modified FLV: A_x = sum of ln[(x_i + 0.001) / 1] and
  FLV = 100 (A_s - A_o) / (|A_o| + 1e-9). This is ``_fdc`` of ``scripts/consolida_sim.py``,
  the FLV of Table 3.
- Eq. S2, original FLV (Yilmaz et al., 2008, Eq. A4): B_x = sum of [ln x_i - ln min(x)] and
  FLV_Y = -100 (B_s - B_o) / B_o. All retained values are positive, so no regularization is used.
- Tail geometric-mean ratio: exp((A_s - A_o) / k), the geometric mean of the simulated tail over
  that of the observed tail, both regularized as in Eq. S1. Its seed mean, minus one and in
  percent, gives the "about 21.5% to about 14.6% above the observed one" of Section 4.3.
- logNSE: NSE of ln(Q + 0.01 mean(Q_obs)) over the whole series (manuscript Section 3.6;
  ``_lognse`` of ``scripts/consolida_sim.py``).

All signatures are computed on the stored float32 series, like the metrics of Table 3
(``scripts/analysis/table3.py``) and the contrasts (``scripts/analysis/contrasts.py``), so that
these scripts print the same unrounded values. A float64 recomputation runs as a check and must
leave every rounded value, seed count and p-value unchanged.

Tests. For each seed, d = |S(LSTM)| - |S(hybrid)| for the two FLV signatures (positive when the
hybrid has the smaller absolute bias, as in Supplementary Section S11) and
d = logNSE(hybrid) - logNSE(LSTM). The exact Wilcoxon signed-rank test is one-sided (d > 0): the
tail stated in Section S11 for the signatures and given by the rule of manuscript Section 3.6
(each hybrid against the pure neural model) for logNSE. As in the manuscript, p-values are not
adjusted for multiple comparisons. Zero differences are dropped, tied absolute differences get
average ranks, and p = P(W+ >= observed W+) is counted over the 2**n equally likely sign
patterns; the p-values are cross-checked against ``scipy.stats.wilcoxon(method="exact")``.

Unrounded values are printed next to the values rounded half away from zero from them, to two
decimals and, in the comparison with the manuscript, to the precision of the printed value (one
decimal for the FLV in the main text); p-values are rounded as the text prints them.

Reads ``outputs/grade_sim`` (``predictions.npz``) and writes nothing unless ``--csv`` is given.

Usage, from the repository root:
    python scripts/analysis/flv_signatures.py
    python scripts/analysis/flv_signatures.py --per-seed   # values and differences by seed
    python scripts/analysis/flv_signatures.py --csv        # also write CSV files to outputs/tables
    python scripts/analysis/flv_signatures.py --grade outputs/retrain --strict
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import textwrap
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRADE = ROOT / "outputs" / "grade_sim"
DEFAULT_CSV = ROOT / "outputs" / "tables" / "flv_signatures.csv"

TAIL_FRACTION = 0.3  # each tail keeps the k = floor(0.30 N) lowest values of its sorted series
EPSILON = 1e-3  # m3/s, regularization of Eq. S1 (reference discharge Q_ref = 1 m3/s)
ALPHA = 0.05  # nominal level of the statements on significance

REFERENCE = "LSTM"
TOP = "LSTM_DUH_Topmodel"
PEONLY = "LSTM_DUH_Topmodel_PeOnly"
HYBRIDS = (TOP, PEONLY)
CONFIGS = (
    REFERENCE,
    TOP,
    PEONLY,
    "LSTM_DUH_Topmodel_Baseflow",  # Section 4.4: the same FLV as PeOnly
    "Phys_DUH_SCS",  # Section 4.3: the two physics-only baselines
    "Phys_DUH_Topmodel",
)
# Rows of Supplementary Table S7, with the labels of the table.
TABLE_S7 = {
    REFERENCE: "LSTM",
    TOP: "TOPMODEL with raw rainfall",
    PEONLY: "TOPMODEL PeOnly",
}
# Per-seed quantities: attribute of Signatures and column label.
METRICS = {
    "flv": "FLV (Eq. S1), %",
    "flv_y": "FLV_Y (Eq. S2), %",
    "gm_pct": "Tail GM above obs., %",
    "lognse": "logNSE",
}
TEST_LABEL = {"flv": "FLV (Eq. S1)", "flv_y": "FLV_Y (Eq. S2)", "lognse": "logNSE"}
GAP_LABEL = {"flv": "FLV", "flv_y": "FLV_Y", "gm_pct": "tail GM", "lognse": "logNSE"}
# Seed-paired tests against the pure LSTM, as (metric, hybrid).
TESTS = tuple((metric, hybrid) for metric in ("flv", "flv_y", "lognse") for hybrid in HYBRIDS)


@dataclass(frozen=True)
class Printed:
    """A value printed in the manuscript and the key of the recomputed quantity it reports.

    Keys: ``mean:<metric>:<configuration>`` and ``sd:<metric>:<configuration>`` (seed mean and
    sample standard deviation), ``delta:<metric>:<hybrid>`` (mean paired difference),
    ``wins:<metric>:<hybrid>`` ("k/n", seeds that favor the hybrid), ``p:<metric>:<hybrid>``
    (exact one-sided p-value), ``significant:<metric>:<hybrid>`` ("yes" when p < 0.05), ``n``
    and ``k`` (pairs per run and values per tail) and ``range:flv`` (the Table S7 FLV means
    rounded to integers).
    """

    where: str
    what: str
    key: str
    text: str


# Every value of the manuscript that this script recomputes, in the order of the text. Seed
# counts that the main text writes as "9 of 10 seeds" are written "9/10"; the "significant" rows
# encode statements made without a number. TOPMODEL is LSTM_DUH_Topmodel (raw rainfall kept) and
# PeOnly is LSTM_DUH_Topmodel_PeOnly.
# fmt: off
MANUSCRIPT = (
    Printed("Section 4.3", "FLV of LSTM, %", "mean:flv:LSTM", "6.0"),
    Printed("Section 4.3", f"FLV of {PEONLY}, %", f"mean:flv:{PEONLY}", "4.2"),
    Printed("Section 4.3", "PeOnly vs LSTM, FLV: seeds with smaller |FLV|", f"wins:flv:{PEONLY}",
            "9/10"),
    Printed("Section 4.3", "PeOnly vs LSTM, FLV: p", f"p:flv:{PEONLY}", "0.007"),
    Printed("Section 4.3", "PeOnly vs LSTM, FLV: significant reduction",
            f"significant:flv:{PEONLY}", "yes"),
    Printed("Section 4.3", "LSTM: tail geometric mean above observed, %", "mean:gm_pct:LSTM",
            "21.5"),
    Printed("Section 4.3", "PeOnly: tail geometric mean above observed, %", f"mean:gm_pct:{PEONLY}",
            "14.6"),
    Printed("Section 4.3", "FLV of Phys_DUH_SCS, %", "mean:flv:Phys_DUH_SCS", "-0.4"),
    Printed("Section 4.3", "FLV of Phys_DUH_Topmodel, %", "mean:flv:Phys_DUH_Topmodel", "+0.9"),
    Printed("Section 4.3", f"FLV of {TOP}, %", f"mean:flv:{TOP}", "5.6"),
    Printed("Section 4.3", "TOPMODEL vs LSTM, FLV: seeds with smaller |FLV|", f"wins:flv:{TOP}",
            "5/10"),
    Printed("Section 4.3", "TOPMODEL vs LSTM, FLV: p", f"p:flv:{TOP}", "0.35"),
    Printed("Section 4.3", "TOPMODEL vs LSTM, FLV: significant reduction", f"significant:flv:{TOP}",
            "no"),
    Printed("Section 4.3", "TOPMODEL vs LSTM, FLV_Y: significant", f"significant:flv_y:{TOP}",
            "yes"),
    Printed("Section 4.3", "PeOnly vs LSTM, FLV_Y: significant", f"significant:flv_y:{PEONLY}",
            "yes"),
    Printed("Section 4.3", "PeOnly vs LSTM: logNSE gain", f"delta:lognse:{PEONLY}", "+0.09"),
    Printed("Section 4.3", "PeOnly vs LSTM, logNSE: p", f"p:lognse:{PEONLY}", "0.065"),
    Printed("Section 4.3", "PeOnly vs LSTM, logNSE: significant", f"significant:lognse:{PEONLY}",
            "no"),
    Printed("Section 4.3", "TOPMODEL vs LSTM: logNSE gain", f"delta:lognse:{TOP}", "+0.06"),
    Printed("Section 4.3", "TOPMODEL vs LSTM, logNSE: p", f"p:lognse:{TOP}", "0.14"),
    Printed("Section 4.3", "TOPMODEL vs LSTM, logNSE: significant", f"significant:lognse:{TOP}",
            "no"),
    Printed("Section 4.4", "FLV of LSTM_DUH_Topmodel_Baseflow, %",
            "mean:flv:LSTM_DUH_Topmodel_Baseflow", "4.2"),
    Printed("Section 4.4", f"FLV of {PEONLY}, %", f"mean:flv:{PEONLY}", "4.2"),
    Printed("Table 3", "logNSE of LSTM", "mean:lognse:LSTM", "0.69"),
    Printed("Table 3", f"logNSE of {TOP}", f"mean:lognse:{TOP}", "0.75"),
    Printed("Table 3", f"logNSE of {PEONLY}", f"mean:lognse:{PEONLY}", "0.79"),
    Printed("Table 3", "logNSE of LSTM_DUH_Topmodel_Baseflow",
            "mean:lognse:LSTM_DUH_Topmodel_Baseflow", "0.78"),
    Printed("Table 3", "logNSE of Phys_DUH_SCS", "mean:lognse:Phys_DUH_SCS", "-0.92"),
    Printed("Table 3", "logNSE of Phys_DUH_Topmodel", "mean:lognse:Phys_DUH_Topmodel", "-0.52"),
    Printed("Section S4", "test FLV of the Table S7 rows, rounded, %", "range:flv", "+4 to +6"),
    Printed("Table S7", "caption: valid hours per run (finite pairs)", "n", "6,323"),
    Printed("Table S7", "caption: values per tail", "k", "1,896"),
    Printed("Table S7", "LSTM, modified FLV: mean", "mean:flv:LSTM", "+6.04"),
    Printed("Table S7", "LSTM, modified FLV: sd", "sd:flv:LSTM", "1.97"),
    Printed("Table S7", "LSTM, original FLV (Yilmaz): mean", "mean:flv_y:LSTM", "+73.65"),
    Printed("Table S7", "LSTM, original FLV (Yilmaz): sd", "sd:flv_y:LSTM", "9.97"),
    Printed("Table S7", "TOPMODEL, modified FLV: mean", f"mean:flv:{TOP}", "+5.63"),
    Printed("Table S7", "TOPMODEL, modified FLV: sd", f"sd:flv:{TOP}", "1.62"),
    Printed("Table S7", "TOPMODEL, original FLV (Yilmaz): mean", f"mean:flv_y:{TOP}", "+56.13"),
    Printed("Table S7", "TOPMODEL, original FLV (Yilmaz): sd", f"sd:flv_y:{TOP}", "18.37"),
    Printed("Table S7", "PeOnly, modified FLV: mean", f"mean:flv:{PEONLY}", "+4.22"),
    Printed("Table S7", "PeOnly, modified FLV: sd", f"sd:flv:{PEONLY}", "1.86"),
    Printed("Table S7", "PeOnly, original FLV (Yilmaz): mean", f"mean:flv_y:{PEONLY}", "+49.88"),
    Printed("Table S7", "PeOnly, original FLV (Yilmaz): sd", f"sd:flv_y:{PEONLY}", "12.47"),
    Printed("Section S11", "Eq. S1, TOPMODEL vs LSTM: seeds improved", f"wins:flv:{TOP}", "5/10"),
    Printed("Section S11", "Eq. S1, TOPMODEL vs LSTM: p", f"p:flv:{TOP}", "0.348"),
    Printed("Section S11", "Eq. S1, PeOnly vs LSTM: seeds improved", f"wins:flv:{PEONLY}", "9/10"),
    Printed("Section S11", "Eq. S1, PeOnly vs LSTM: p", f"p:flv:{PEONLY}", "0.0068"),
    Printed("Section S11", "Eq. S2, TOPMODEL vs LSTM: seeds improved", f"wins:flv_y:{TOP}",
            "8/10"),
    Printed("Section S11", "Eq. S2, TOPMODEL vs LSTM: p", f"p:flv_y:{TOP}", "0.0322"),
    Printed("Section S11", "Eq. S2, PeOnly vs LSTM: seeds improved", f"wins:flv_y:{PEONLY}",
            "10/10"),
    Printed("Section S11", "Eq. S2, PeOnly vs LSTM: p", f"p:flv_y:{PEONLY}", "0.0010"),
)
# fmt: on


@dataclass(frozen=True)
class Run:
    """One completed run: the stored test-period series of one configuration and seed."""

    seed: int
    sim: np.ndarray  # simulated discharge, m3/s, as stored (float32)
    obs: np.ndarray  # observed discharge, m3/s, as stored (float32)


@dataclass(frozen=True)
class Signatures:
    """Low-flow signatures of one run, over its finite pairs."""

    n: int  # finite simulated-observed pairs
    k: int  # values per tail, floor(0.30 n)
    flv: float  # Eq. S1, percent
    flv_y: float  # Eq. S2, percent
    gm_pct: float  # tail geometric mean of the simulation above the observed, percent
    gm_plain_pct: float  # the same without the 0.001 regularization (a check)
    lognse: float
    sim_min: float  # smallest retained simulated value, m3/s
    obs_min: float  # smallest retained observed value, m3/s


@dataclass(frozen=True)
class PairedTest:
    """Exact one-sided Wilcoxon signed-rank test of a hybrid against the pure LSTM."""

    metric: str
    hybrid: str
    seeds: tuple[int, ...]
    diff: tuple[float, ...]  # d by seed, positive when the hybrid is better
    w_plus: float  # sum of the ranks of the positive differences
    p: float  # exact one-sided p-value, P(W+ >= observed W+)
    zeros: int  # zero differences, dropped
    ties: bool  # tied absolute differences (average ranks were used)
    scipy_p: float | None  # scipy.stats.wilcoxon(method="exact"); None with zeros or ties

    @property
    def wins(self) -> int:
        """Seeds in which the hybrid is better (d > 0)."""
        return sum(d > 0 for d in self.diff)

    @property
    def mean_diff(self) -> float:
        """Mean paired difference."""
        return float(np.mean(self.diff))


# ---------------------------------------------------------------------------- signatures


def low_tails(sim: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The k = floor(0.30 N) lowest values of each independently sorted series."""
    k = int(TAIL_FRACTION * len(obs))
    return np.sort(sim)[:k], np.sort(obs)[:k]


def flv_modified(tail_sim: np.ndarray, tail_obs: np.ndarray) -> float:
    """Supplementary Eq. S1, computed as the FLV of ``_fdc`` in scripts/consolida_sim.py."""
    a_sim = np.log(tail_sim + EPSILON).sum()
    a_obs = np.log(tail_obs + EPSILON).sum()
    return float(100 * (a_sim - a_obs) / (abs(a_obs) + 1e-9))


def flv_yilmaz(tail_sim: np.ndarray, tail_obs: np.ndarray) -> float:
    """Supplementary Eq. S2, the minimum-referenced signature of Yilmaz et al. (2008, Eq. A4)."""
    b_sim = np.sum(np.log(tail_sim) - np.log(tail_sim.min()))
    b_obs = np.sum(np.log(tail_obs) - np.log(tail_obs.min()))
    return float(-100 * (b_sim - b_obs) / b_obs)


def tail_gm_ratio(tail_sim: np.ndarray, tail_obs: np.ndarray, eps: float = EPSILON) -> float:
    """Geometric mean of the simulated tail over that of the observed tail, exp((A_s - A_o) / k).

    With the default ``eps`` both geometric means are regularized as in Eq. S1; ``eps`` = 0
    gives the plain geometric means.
    """
    log_sim, log_obs = np.log(tail_sim + eps), np.log(tail_obs + eps)
    return float(np.exp((log_sim.sum() - log_obs.sum()) / len(tail_obs)))


def log_nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """NSE of ln(Q + 0.01 mean(Q_obs)) (``_lognse`` of scripts/consolida_sim.py)."""
    eps = 0.01 * np.nanmean(obs)
    log_sim, log_obs = np.log(sim + eps), np.log(obs + eps)
    den = np.sum((log_obs - log_obs.mean()) ** 2)
    return float(1 - np.sum((log_sim - log_obs) ** 2) / den) if den > 0 else float("nan")


def signatures(run: Run, dtype: type[np.floating] | None = None) -> Signatures:
    """Signatures of one run over its finite pairs; ``dtype`` casts the series first."""
    ok = np.isfinite(run.sim) & np.isfinite(run.obs)
    sim, obs = run.sim[ok], run.obs[ok]
    if dtype is not None:
        sim, obs = sim.astype(dtype), obs.astype(dtype)
    tail_sim, tail_obs = low_tails(sim, obs)
    return Signatures(
        n=int(obs.size),
        k=int(tail_obs.size),
        flv=flv_modified(tail_sim, tail_obs),
        flv_y=flv_yilmaz(tail_sim, tail_obs),
        gm_pct=100 * (tail_gm_ratio(tail_sim, tail_obs) - 1),
        gm_plain_pct=100 * (tail_gm_ratio(tail_sim, tail_obs, eps=0.0) - 1),
        lognse=log_nse(sim, obs),
        sim_min=float(tail_sim.min()),
        obs_min=float(tail_obs.min()),
    )


# ---------------------------------------------------------------------------- tests


def exact_signed_rank_greater(diff: np.ndarray) -> tuple[float, float, int, bool]:
    """W+, exact one-sided p-value P(W+ >= observed W+), zeros dropped and presence of ties.

    The null distribution of W+ is built by counting the 2**n equally likely sign patterns of
    the ranks of the non-zero differences (ranks are doubled so that average ranks of tied
    absolute differences stay integers).
    """
    d = np.asarray(diff, dtype=float)
    zeros = int(np.sum(d == 0))
    d = d[d != 0]
    if d.size == 0 or d.size > 60:
        raise ValueError(f"the exact test needs 1 to 60 non-zero differences, got {d.size}")
    doubled = np.rint(2 * stats.rankdata(np.abs(d))).astype(np.int64)
    w_obs = int(doubled[d > 0].sum())
    counts = np.zeros(int(doubled.sum()) + 1, dtype=np.int64)  # sign patterns by 2 W+
    counts[0] = 1
    for rank in doubled:
        shifted = np.zeros_like(counts)
        shifted[rank:] = counts[: counts.size - rank]
        counts = counts + shifted
    p = int(counts[w_obs:].sum()) / 2**d.size
    ties = np.unique(np.abs(d)).size < d.size
    return w_obs / 2, p, zeros, bool(ties)


def paired_test(values: dict[str, dict[int, Signatures]], metric: str, hybrid: str) -> PairedTest:
    """Test of ``hybrid`` against the pure LSTM over the seeds they share."""
    reference, other = values[REFERENCE], values[hybrid]
    seeds = tuple(sorted(set(reference) & set(other)))
    if len(seeds) < 2:
        raise RuntimeError(f"{hybrid} and {REFERENCE} share fewer than two seeds")
    x = np.array([getattr(reference[seed], metric) for seed in seeds])
    y = np.array([getattr(other[seed], metric) for seed in seeds])
    diff = y - x if metric == "lognse" else np.abs(x) - np.abs(y)
    w_plus, p, zeros, ties = exact_signed_rank_greater(diff)
    scipy_p = None
    if not zeros and not ties:
        scipy_p = float(stats.wilcoxon(diff, alternative="greater", method="exact").pvalue)
    return PairedTest(
        metric, hybrid, seeds, tuple(float(v) for v in diff), w_plus, p, zeros, ties, scipy_p
    )


# ---------------------------------------------------------------------------- summaries


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (divisor n - 1); NaN deviation for a single value."""
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    return float(arr.mean()), sd


def summarize(
    values: dict[str, dict[int, Signatures]], tests: list[PairedTest]
) -> dict[str, float | str]:
    """Recomputed quantities under the keys of ``Printed``."""
    found: dict[str, float | str] = {}
    for config, by_seed in values.items():
        for metric in METRICS:
            mean, sd = mean_sd([getattr(sig, metric) for sig in by_seed.values()])
            found[f"mean:{metric}:{config}"] = mean
            found[f"sd:{metric}:{config}"] = sd
    for test in tests:
        key = f"{test.metric}:{test.hybrid}"
        found[f"delta:{key}"] = test.mean_diff
        found[f"wins:{key}"] = f"{test.wins}/{len(test.seeds)}"
        found[f"p:{key}"] = test.p
        found[f"significant:{key}"] = "yes" if test.p < ALPHA else "no"
    sigs = [sig for by_seed in values.values() for sig in by_seed.values()]
    found["n"] = ", ".join(f"{n:,}" for n in sorted({sig.n for sig in sigs}))
    found["k"] = ", ".join(f"{k:,}" for k in sorted({sig.k for sig in sigs}))
    means = [float(found[f"mean:flv:{config}"]) for config in TABLE_S7]
    low, high = round_half_up(min(means), 0), round_half_up(max(means), 0)
    found["range:flv"] = f"{low:+} to {high:+}"
    return found


def recompute(item: Printed, found: dict[str, float | str]) -> tuple[str, str, bool, float | None]:
    """Unrounded value, recomputed value as printed, match, and distance to a rounding boundary.

    Numbers are rounded half away from zero from the unrounded value to the decimals of the
    printed value. The distance (numbers other than p-values only) is that of the unrounded
    value to the nearest rounding boundary, in units of the last printed digit.
    """
    value = found[item.key]
    kind, _, rest = item.key.partition(":")
    if kind == "significant":
        return f"p={float(found['p:' + rest]):.4g}", str(value), value == item.text, None
    if kind == "range":
        means = [float(found[f"mean:flv:{config}"]) for config in TABLE_S7]
        return f"{min(means):.2f} to {max(means):.2f}", str(value), value == item.text, None
    if isinstance(value, str):
        return value, value, value == item.text, None
    nd = printed_decimals(item.text)
    rounded = round_half_up(value, nd)
    ok = rounded == Decimal(item.text)
    if kind == "p":
        return f"{value:.10g}", f"{rounded}", ok, None
    signed = item.text.startswith("+")
    unrounded = f"{value:+.6f}" if signed else f"{value:.6f}"
    shown = f"{rounded:+}" if signed else f"{rounded}"
    return unrounded, shown, ok, rounding_distance(value, nd)


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

    A seed folder without predictions.npz is skipped: seeds 44 and 48 of Phys_DUH_Topmodel are
    interrupted runs that keep only config.json.
    """
    runs = []
    for seed, folder in seed_folders(grade):
        found = sorted((folder / config).glob("*/predictions.npz"))
        if not found:
            continue
        if len(found) > 1:
            raise RuntimeError(f"more than one predictions.npz under {folder / config}")
        with np.load(found[0]) as npz:
            sim, obs = npz["pred"].reshape(-1), npz["target"].reshape(-1)
        runs.append(Run(seed, sim, obs))
    return runs


# ---------------------------------------------------------------------------- formatting


def round_half_up(value: float, decimals: int) -> Decimal:
    """Round half away from zero, from the shortest repr of the unrounded value."""
    step = Decimal(1).scaleb(-decimals)
    rounded = Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_UP)
    return abs(rounded) if rounded == 0 else rounded  # no negative zero


def printed_decimals(text: str) -> int:
    """Number of decimals of a printed number."""
    return max(0, -int(Decimal(text.replace(",", "")).as_tuple().exponent))


def rounding_distance(value: float, decimals: int) -> float:
    """Distance to the nearest rounding boundary, in units of the last printed digit (0 to 0.5)."""
    scaled = abs(value) * 10**decimals
    return abs(scaled - math.floor(scaled) - 0.5)


def with_two_decimals(value: float) -> str:
    """Unrounded value and, in parentheses, the value rounded to two decimals."""
    return f"{value:+.6f} ({round_half_up(value, 2):+})"


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


def print_wrapped(text: str, indent: str = "  ", width: int = 100) -> None:
    """Print ``text`` wrapped to ``width`` columns, continuation lines indented two more spaces."""
    print(
        textwrap.fill(
            text,
            width=width,
            initial_indent=indent,
            subsequent_indent=indent + "  ",
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


# ---------------------------------------------------------------------------- report


def print_table_s7(found: dict[str, float | str], values: dict[str, dict[int, Signatures]]) -> None:
    """Supplementary Table S7, rounded as printed and unrounded."""
    print()
    print("Supplementary Table S7 (percent; mean +/- sample standard deviation across seeds)")
    head = f"  {'Configuration':<27} {'Seeds':>5}  {'Modified FLV (Eq. S1)':<23}"
    print(f"{head}  Original FLV, Yilmaz (Eq. S2)")
    rounded, unrounded = [], []
    for config, label in TABLE_S7.items():
        cells_r, cells_u = [], []
        for metric in ("flv", "flv_y"):
            mean = float(found[f"mean:{metric}:{config}"])
            sd = float(found[f"sd:{metric}:{config}"])
            cells_r.append(f"{round_half_up(mean, 2):+} +/- {round_half_up(sd, 2)}")
            cells_u.append(f"{mean:+.6f} +/- {sd:.6f}")
        seeds = len(values[config])
        rounded.append(f"  {label:<27} {seeds:>5}  {cells_r[0]:<23}  {cells_r[1]}")
        unrounded.append(f"  {label:<27} {'':>5}  {cells_u[0]:<23}  {cells_u[1]}")
    print("\n".join(rounded))
    print("  Unrounded")
    print("\n".join(unrounded))


def print_tests(tests: list[PairedTest]) -> None:
    """The seed-paired tests against the pure LSTM."""
    print()
    print("Seed-paired tests against the pure LSTM (exact Wilcoxon signed-rank, one-sided,")
    print("unadjusted). d by seed: |S(LSTM)| - |S(hybrid)| for the two FLV signatures (positive:")
    print("smaller absolute bias in the hybrid), logNSE(hybrid) - logNSE(LSTM) for logNSE.")
    print("Alternative: d > 0.")
    head = f"  {'Signature':<15} {'Hybrid':<25} {'Mean d (2 decimals)':<22}"
    print(f"{head} {'d > 0':>6} {'W+':>5}  p")
    for test in tests:
        wins = f"{test.wins}/{len(test.seeds)}"
        print(
            f"  {TEST_LABEL[test.metric]:<15} {test.hybrid:<25}"
            f" {with_two_decimals(test.mean_diff):<22} {wins:>6} {test.w_plus:>5.1f}  {test.p:.10g}"
        )


def print_means(found: dict[str, float | str], values: dict[str, dict[int, Signatures]]) -> None:
    """Seed means of every signature for the configurations quoted in Sections 4.3 and 4.4."""
    print()
    print("Seed means by configuration (unrounded and, in parentheses, rounded to two decimals)")
    print("Tail GM above obs.: geometric mean of the simulated low-flow tail above that of the")
    print("observed tail, both regularized as in Eq. S1.")
    for pair in (("flv", "flv_y"), ("gm_pct", "lognse")):
        header = "".join(f"  {METRICS[metric]:<22}" for metric in pair)
        print(f"  {'Configuration':<27} {'Seeds':>5}{header}".rstrip())
        for config in CONFIGS:
            cells = "".join(
                f"  {with_two_decimals(float(found[f'mean:{metric}:{config}'])):<22}"
                for metric in pair
            )
            print(f"  {config:<27} {len(values[config]):>5}{cells}".rstrip())


def print_per_seed(values: dict[str, dict[int, Signatures]], tests: list[PairedTest]) -> None:
    """Signatures and paired differences by seed."""
    print()
    print("Values by seed (unrounded)")
    for config in CONFIGS:
        print(f"  {config}")
        print(
            f"    {'Seed':>4} {'FLV (Eq. S1), %':>16} {'FLV_Y (Eq. S2), %':>18}"
            f" {'Tail GM, %':>11} {'logNSE':>10} {'Tail min, m3/s':>15}"
        )
        for seed, sig in values[config].items():
            print(
                f"    {seed:>4} {sig.flv:>+16.6f} {sig.flv_y:>+18.6f} {sig.gm_pct:>+11.6f}"
                f" {sig.lognse:>+10.6f} {sig.sim_min:>15.4f}"
            )
    print()
    print("Paired differences d by seed (positive favors the hybrid)")
    for test in tests:
        print(f"  {TEST_LABEL[test.metric]}, {test.hybrid}")
        pairs = [f"{seed}: {d:+.6f}" for seed, d in zip(test.seeds, test.diff, strict=True)]
        for start in range(0, len(pairs), 5):
            print("    " + "   ".join(pairs[start : start + 5]))


def print_comparison(found: dict[str, float | str]) -> tuple[list[str], list[tuple]]:
    """Recomputed values against the manuscript; returns mismatches and rounding distances."""
    print()
    print("Comparison with the manuscript. Recomputed: rounded half away from zero from the")
    print("unrounded value to the precision of the printed value. TOPMODEL: LSTM_DUH_Topmodel;")
    print("PeOnly: LSTM_DUH_Topmodel_PeOnly.")
    print(f"  {'Quantity':<48} {'Unrounded':>13} {'Recomputed':>10} {'Printed':>9}  Status")
    mismatches, distances = [], []
    where = None
    for item in MANUSCRIPT:
        if item.where != where:
            where = item.where
            print(f"  {where}")
        unrounded, shown, ok, distance = recompute(item, found)
        status = "ok" if ok else "DIFFERS"
        print(f"    {item.what:<46} {unrounded:>13} {shown:>10} {item.text:>9}  {status}")
        if not ok:
            mismatches.append(f"{item.where}, {item.what}: manuscript {item.text}, data {shown}")
        if distance is not None:
            distances.append((distance, item, unrounded, shown))
    print(f"  {len(MANUSCRIPT) - len(mismatches)} of {len(MANUSCRIPT)} printed values match.")
    for line in mismatches:
        print(f"    DIFFERS: {line}")
    return mismatches, distances


def print_checks(
    runs: dict[str, list[Run]],
    values: dict[str, dict[int, Signatures]],
    tests: list[PairedTest],
    found: dict[str, float | str],
    distances: list[tuple],
) -> list[str]:
    """Consistency checks of the input and of the computation; returns the failed checks."""
    failed = []
    print()
    print("Checks")
    all_runs = [run for found_runs in runs.values() for run in found_runs]
    by_seeds: dict[tuple[int, ...], list[str]] = {}
    for config in CONFIGS:
        by_seeds.setdefault(tuple(run.seed for run in runs[config]), []).append(config)
    print_wrapped(f"Runs read: {len(all_runs)}")
    for seeds, configs in by_seeds.items():
        print_wrapped(f"seeds {number_list(list(seeds))}: {', '.join(configs)}", indent="    ")

    sigs = [sig for by_seed in values.values() for sig in by_seed.values()]
    dropped = sum(run.obs.size for run in all_runs) - sum(sig.n for sig in sigs)
    print_wrapped(
        f"Finite pairs per run: {found['n']} ({dropped} non-finite pairs excluded in total);"
        f" values per tail: {found['k']}"
    )
    reference = all_runs[0].obs
    same_obs = all(np.array_equal(run.obs, reference, equal_nan=True) for run in all_runs)
    print_wrapped(f"Observed series identical in all runs: {'yes' if same_obs else 'NO'}")
    if not same_obs:
        failed.append("observed series differ between runs")
    sim_min = min(sig.sim_min for sig in sigs)
    obs_min = min(sig.obs_min for sig in sigs)
    positive = sim_min > 0 and obs_min > 0
    print_wrapped(
        f"Smallest retained value: simulated {sim_min:.4f} m3/s, observed {obs_min:.4f} m3/s"
        f" ({'all positive, so Eq. S2 needs no regularization' if positive else 'NOT positive'})"
    )
    if not positive:
        failed.append("non-positive values in a low-flow tail")

    zeros = sum(test.zeros for test in tests)
    ties = [f"{TEST_LABEL[t.metric]}, {t.hybrid}" for t in tests if t.ties]
    scipy_ok = all(t.scipy_p is None or t.scipy_p == t.p for t in tests)
    print_wrapped(
        f"Wilcoxon: {zeros} zero differences; "
        + ("tied absolute differences in " + "; ".join(ties) if ties else "no tied differences")
    )
    print_wrapped(
        "Exact p-values equal to scipy.stats.wilcoxon(method='exact'):"
        f" {'yes' if scipy_ok else 'NO'}"
    )
    if not scipy_ok:
        failed.append("exact p-values differ from scipy.stats.wilcoxon")

    # Float64 recomputation: same rounded values, seed counts and p-values.
    values64 = {
        config: {run.seed: signatures(run, np.float64) for run in runs[config]}
        for config in CONFIGS
    }
    tests64 = [paired_test(values64, metric, hybrid) for metric, hybrid in TESTS]
    found64 = summarize(values64, tests64)
    gaps = {
        metric: max(
            abs(getattr(values[c][s], metric) - getattr(values64[c][s], metric))
            for c in CONFIGS
            for s in values[c]
        )
        for metric in METRICS
    }
    changed = [
        item.what for item in MANUSCRIPT if recompute(item, found)[1] != recompute(item, found64)[1]
    ]
    changed += [
        f"{TEST_LABEL[a.metric]}, {a.hybrid}"
        for a, b in zip(tests, tests64, strict=True)
        if (a.wins, a.p) != (b.wins, b.p)
    ]
    gap_text = ", ".join(f"{GAP_LABEL[metric]} {gaps[metric]:.1e}" for metric in METRICS)
    print_wrapped(
        f"float64 recomputation, max |difference| by seed: {gap_text} (percentage points except"
        " logNSE)"
    )
    print_wrapped(
        "rounded values, seed counts and p-values unchanged: "
        + ("yes" if not changed else "NO (" + "; ".join(changed) + ")"),
        indent="    ",
    )
    if changed:
        failed.append("float64 recomputation changes a rounded value, a count or a p-value")

    plain = []
    for config in (REFERENCE, PEONLY):
        value = float(np.mean([sig.gm_plain_pct for sig in values[config].values()]))
        plain.append(f"{config} {value:+.6f} ({round_half_up(value, 1)})")
    print_wrapped(
        "Tail geometric mean above observed without the 0.001 regularization (seed mean, %):"
        f" {plain[0]}, {plain[1]}"
    )

    print_wrapped(
        "Closest to a rounding boundary (distance in units of the last printed digit, from 0 on"
        " the boundary to 0.5):"
    )
    for distance, item, unrounded, shown in sorted(distances, key=lambda entry: entry[0])[:3]:
        print_wrapped(
            f"{item.where}, {item.what}: {unrounded} -> {shown} (distance {distance:.3f})",
            indent="    ",
        )
    return failed


def write_csv(
    path: Path, values: dict[str, dict[int, Signatures]], tests: list[PairedTest]
) -> None:
    """Write the signatures by seed to ``path`` and the tests to ``<stem>_tests.csv``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "configuration",
                "seed",
                "n_pairs",
                "k_tail",
                "flv_eq_s1_pct",
                "flv_yilmaz_eq_s2_pct",
                "tail_gm_above_obs_pct",
                "tail_gm_plain_above_obs_pct",
                "lognse",
                "sim_tail_min_m3s",
                "obs_tail_min_m3s",
            ]
        )
        for config in CONFIGS:
            for seed, sig in values[config].items():
                writer.writerow(
                    [
                        config,
                        seed,
                        sig.n,
                        sig.k,
                        repr(sig.flv),
                        repr(sig.flv_y),
                        repr(sig.gm_pct),
                        repr(sig.gm_plain_pct),
                        repr(sig.lognse),
                        repr(sig.sim_min),
                        repr(sig.obs_min),
                    ]
                )
    tests_path = path.with_name(f"{path.stem}_tests.csv")
    with tests_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["signature", "hybrid", "reference", "seeds", "mean_d", "wins", "n", "w_plus", "p"]
        )
        for test in tests:
            writer.writerow(
                [
                    test.metric,
                    test.hybrid,
                    REFERENCE,
                    " ".join(str(seed) for seed in test.seeds),
                    repr(test.mean_diff),
                    test.wins,
                    len(test.seeds),
                    test.w_plus,
                    repr(test.p),
                ]
            )
    print(f"\nWrote {display_path(path)} and {display_path(tests_path)}")


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
        "--per-seed",
        action="store_true",
        help="also print the signatures and the paired differences of every seed",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        type=Path,
        const=DEFAULT_CSV,
        default=None,
        metavar="PATH",
        help="also write the values by seed to PATH and the tests to PATH with suffix _tests"
        " (default PATH: outputs/tables/flv_signatures.csv)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 if a value differs from the manuscript or a check fails",
    )
    args = parser.parse_args()

    runs = {config: load_runs(args.grade, config) for config in CONFIGS}
    missing = [config for config, config_runs in runs.items() if not config_runs]
    if missing:
        parser.error(f"no completed runs in {args.grade} for: {', '.join(missing)}")
    values = {
        config: {run.seed: signatures(run) for run in config_runs}
        for config, config_runs in runs.items()
    }
    tests = [paired_test(values, metric, hybrid) for metric, hybrid in TESTS]
    found = summarize(values, tests)
    dtypes = sorted(
        {str(run.sim.dtype) for config_runs in runs.values() for run in config_runs}
        | {str(run.obs.dtype) for config_runs in runs.values() for run in config_runs}
    )

    print("Low-flow signatures of the test period")
    print("Manuscript Section 4.3; Supplementary Section S11 (Eqs. S1 and S2) and Table S7")
    print(f"Grid: {display_path(args.grade)}")
    print_wrapped(
        f"Series as stored ({', '.join(dtypes)}), finite pairs only; each tail holds the"
        " k = floor(0.30 N) lowest values of its independently sorted series.",
        indent="",
    )
    print_table_s7(found, values)
    print_tests(tests)
    print_means(found, values)
    if args.per_seed:
        print_per_seed(values, tests)
    mismatches, distances = print_comparison(found)
    failed = print_checks(runs, values, tests, found, distances)
    if args.csv is not None:
        write_csv(args.csv, values, tests)
    return 1 if args.strict and (mismatches or failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
