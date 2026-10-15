#!/usr/bin/env python
"""Reproduce the seed-paired contrasts of Sections 4.2 to 4.4 and Supplementary Section S2.

Each contrast compares two configurations of the grid over the seeds they share (42 to 51).
The difference A minus B is formed seed by seed and tested with the exact Wilcoxon signed-rank
test, with the tail used in the manuscript. For every contrast the script prints the seed means
of A and B, the mean paired difference, the number of seeds in which A is better, the
signed-rank statistic W+ and the exact p-value, next to the values printed in the manuscript and
whether they match. It also prints the differences of seed means of the input decomposition
(Section 4.2) and of the seed ensembles (Section 4.3), which the manuscript reports without a
test, and three rows (marked "no p-value in the text") that support statements the text makes
without printing a test.

Tails (manuscript Section 3.6): one-sided (A better than B) where the hypothesis is directional
(physics adds skill: each hybrid against the pure neural model, each PeOnly configuration against
its raw-rainfall counterpart, the state-based generator against the event-based one), and
two-sided for the degradation of SCS-CN when the raw-rainfall shortcut is closed. Where the text
names the tail (Sections 4.2 and 4.4, Supplementary Sections S2 and S11) it is used as stated;
otherwise the output says whether the tail follows the Methods rule or is not stated in the text,
in which case the one-sided test is the one that gives the printed p-value. As in the manuscript,
p-values are not adjusted for multiple comparisons.

Metrics, as in Table 3 (``scripts/analysis/table3.py``):

- NSE: ``test_metrics.nse`` of each ``results.json``, written by ``scripts/train.py``.
- logNSE: NSE of ln(Q + 0.01 mean(Q_obs)), recomputed from ``predictions.npz``.
- FLV: modified low-flow log-tail bias (Supplementary Eq. S1), recomputed from
  ``predictions.npz``. Its contrasts test the reduction of the absolute bias, |FLV_B| - |FLV_A|
  (Supplementary Section S11), so a positive difference means that A has the smaller bias.
- NSE of the seed ensemble: NSE of the step-by-step mean of the predictions of the seeds.

logNSE, FLV and the ensemble are computed on the stored float32 series over the finite pairs, as
in ``scripts/consolida_sim.py`` and the figure generator. The original low-flow signature of
Yilmaz et al. (2008), cited in Section 4.3, and Supplementary Table S7 are reproduced by
``scripts/analysis/flv_signatures.py``.

The test is exact: zero differences are dropped, tied absolute differences get average ranks,
and the null distribution of W+ (sum of the ranks of the positive differences) is obtained by
counting the 2**n equally likely sign patterns, so that p = count / 2**n. The two-sided p-value
is twice the smaller tail, capped at 1. The results are cross-checked against
``scipy.stats.wilcoxon(method="exact")``.

Values are printed unrounded and, in parentheses, rounded half away from zero from the unrounded
value: two decimals for NSE and logNSE, one decimal for FLV (as in the text), and p-values as the
text prints them (two decimals at or above 0.1, three decimals below).

Reads ``outputs/grade_sim`` (``results.json`` and ``predictions.npz``) and writes nothing unless
``--csv`` is given.

Usage, from the repository root:
    python scripts/analysis/contrasts.py
    python scripts/analysis/contrasts.py --csv        # also write outputs/tables/contrasts.csv
    python scripts/analysis/contrasts.py --grade outputs/retrain --strict
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRADE = ROOT / "outputs" / "grade_sim"
DEFAULT_CSV = ROOT / "outputs" / "tables" / "contrasts.csv"

STATED = "tail stated in the text"
STATED_S11 = "tail stated in Supplementary Section S11"
RULE = "tail from the Methods rule (Section 3.6)"
UNSTATED = "tail not stated in the text; this tail gives the printed p-value"
NO_P = "no p-value in the text: "

METRIC_LABEL = {"nse": "NSE", "lognse": "logNSE", "flv": "FLV, %"}
METRIC_DECIMALS = {"nse": 2, "lognse": 2, "flv": 1}
TAIL_LABEL = {"greater": "one-sided (A better than B)", "two-sided": "two-sided"}
ITEM_LABEL = {
    "mean_a": "mean A",
    "mean_b": "mean B",
    "delta": "difference",
    "wins": "seeds",
    "p": "p",
    "sign": "sign of the difference",
    "significant": "significant at 0.05",
}


@dataclass(frozen=True)
class Contrast:
    """A seed-paired contrast A minus B and the values the manuscript prints for it.

    ``expected`` holds the printed values: ``mean_a`` and ``mean_b`` (seed means), ``delta``
    (mean paired difference), ``wins`` ("k/n", seeds in which A is better) and ``p``. Rows that
    support a statement made without numbers carry qualitative expectations instead: ``sign`` of
    the mean difference ("+" or "-") and ``significant`` ("no" means p >= 0.05).
    """

    section: str
    a: str
    b: str
    metric: str  # "nse", "lognse" or "flv" (reduction of |FLV|)
    tail: str  # "greater" or "two-sided"
    tail_source: str
    title: str
    expected: dict[str, str] = field(default_factory=dict)


# Every seed-paired contrast reported in Sections 4.2 to 4.4 and in Supplementary Section S2, in
# the order of the text.
# fmt: off
CONTRASTS = (
    Contrast(
        "4.2", "LSTM", "LSTM_Lumped", "nse", "greater", STATED,
        "spatially distributed rainfall against its catchment mean",
        {"mean_a": "0.76", "mean_b": "0.67", "delta": "+0.09", "wins": "8/10", "p": "0.053"},
    ),
    Contrast(
        "4.2", "LSTM_DUH", "LSTM", "nse", "greater", RULE,
        NO_P + "DUH routing; 'none of these rungs separates from the pure neural model'",
        {"mean_a": "0.77", "mean_b": "0.76", "significant": "no"},
    ),
    Contrast(
        "4.3", "LSTM_DUH_SCS", "LSTM", "nse", "greater", RULE,
        "SCS-CN generator against the pure neural model",
        {"mean_a": "0.77", "delta": "+0.01", "p": "0.42"},
    ),
    Contrast(
        "4.3", "LSTM_DUH_Topmodel", "LSTM", "nse", "greater", RULE,
        "TOPMODEL generator against the pure neural model",
        {"mean_a": "0.80", "delta": "+0.04", "p": "0.22"},
    ),
    Contrast(
        "4.3", "LSTM_DUH_Topmodel_Baseflow", "LSTM", "nse", "greater", RULE,
        "TOPMODEL with baseflow against the pure neural model",
        {"delta": "+0.06", "p": "0.097"},
    ),
    Contrast(
        "4.3", "LSTM_DUH_Topmodel_PeOnly", "LSTM", "nse", "greater", RULE,
        "best hybrid against the pure neural model",
        {"delta": "+0.07", "wins": "8/10", "p": "0.080"},
    ),
    Contrast(
        "4.3", "LSTM_DUH_SCS_PeOnly", "LSTM", "nse", "greater", RULE,
        NO_P + "'positive, except for SCS-CN with the shortcut closed'",
        {"sign": "-"},
    ),
    Contrast(
        "4.3", "LSTM_DUH_Topmodel_PeOnly", "LSTM", "flv", "greater", STATED_S11,
        "low-flow log-tail bias of the best hybrid against the pure neural model",
        {"mean_a": "4.2", "mean_b": "6.0", "wins": "9/10", "p": "0.007"},
    ),
    Contrast(
        "4.3", "LSTM_DUH_Topmodel", "LSTM", "flv", "greater", STATED_S11,
        "low-flow log-tail bias of TOPMODEL with raw rainfall against the pure neural model",
        {"mean_a": "5.6", "mean_b": "6.0", "wins": "5/10", "p": "0.35"},
    ),
    Contrast(
        "4.3", "LSTM_DUH_Topmodel_PeOnly", "LSTM", "lognse", "greater", RULE,
        "logarithmic NSE of the best hybrid against the pure neural model",
        {"delta": "+0.09", "p": "0.065"},
    ),
    Contrast(
        "4.3", "LSTM_DUH_Topmodel", "LSTM", "lognse", "greater", RULE,
        "logarithmic NSE of TOPMODEL with raw rainfall against the pure neural model",
        {"delta": "+0.06", "p": "0.14"},
    ),
    Contrast(
        "4.4", "LSTM_DUH_Topmodel", "LSTM_Topmodel", "nse", "greater", UNSTATED,
        "explicit DUH routing over the routing-free TOPMODEL variant",
        {"mean_a": "0.80", "mean_b": "0.80", "p": "0.58"},
    ),
    Contrast(
        "4.4", "LSTM_Topmodel", "LSTM", "nse", "greater", RULE,
        "routing-free TOPMODEL against the pure neural model",
        {"mean_a": "0.80", "delta": "+0.04", "p": "0.16"},
    ),
    Contrast(
        "4.4", "LSTM_SCS", "LSTM", "nse", "greater", RULE,
        NO_P + "routing-free SCS-CN 'does not even surpass the pure neural model'",
        {"mean_a": "0.74", "sign": "-"},
    ),
    Contrast(
        "4.4", "LSTM_DUH_SCS_PeOnly", "LSTM_DUH_SCS", "nse", "two-sided", STATED,
        "SCS-CN with the raw-rainfall shortcut closed against SCS-CN",
        {"mean_a": "0.59", "mean_b": "0.77", "delta": "-0.18", "wins": "2/10", "p": "0.014"},
    ),
    Contrast(
        "4.4", "LSTM_DUH_Topmodel_PeOnly", "LSTM_DUH_Topmodel", "nse", "greater", RULE,
        "TOPMODEL with the raw-rainfall shortcut closed against TOPMODEL",
        {"mean_a": "0.83", "mean_b": "0.80", "delta": "+0.03", "wins": "6/10", "p": "0.16"},
    ),
    Contrast(
        "4.4", "LSTM_DUH_Topmodel_PeOnly", "LSTM_DUH_SCS_PeOnly", "nse", "greater", RULE,
        "state-based against event-based generator, shortcut closed",
        {"delta": "+0.24", "wins": "10/10", "p": "0.001"},
    ),
    Contrast(
        "4.4", "LSTM_DUH_Topmodel_Baseflow", "LSTM_DUH_Topmodel", "nse", "greater", UNSTATED,
        "baseflow and mean saturation deficit as additional LSTM inputs",
        {"delta": "+0.02", "p": "0.22"},
    ),
    Contrast(
        "4.4", "LSTM_PET", "LSTM", "nse", "greater", UNSTATED,
        "potential evapotranspiration as an additional LSTM input",
        {"mean_a": "0.78", "delta": "+0.02", "p": "0.35"},
    ),
    Contrast(
        "S2", "LSTM_DUH_Manning", "LSTM_DUH", "nse", "greater", STATED,
        "Manning (roughness-scaled) against topographic concentration time",
        {"delta": "-0.02", "p": "0.78"},
    ),
)
# fmt: on

SECTIONS = {
    "4.2": "Section 4.2. The attribution chain: where the performance comes from",
    "4.3": "Section 4.3. What the embedded physics changes: low flows and robustness",
    "4.4": "Section 4.4. Generation versus routing: routing-free and shortcut-closed contrasts",
    "S2": "Supplementary Section S2. Sensitivity to the concentration time",
}

# Section 4.2, input decomposition (differences of seed means of the NSE, no test), as printed.
DECOMPOSITION_MEANS = {
    "LSTM_Lumped_RainOnly": "-0.09",
    "LSTM_Lumped_CalOnly": "0.17",
    "LSTM_Lumped": "0.67",
}
DECOMPOSITION_DIFFERENCES = (
    ("calendar", "LSTM_Lumped", "LSTM_Lumped_RainOnly", "+0.76"),
    ("rainfall", "LSTM_Lumped", "LSTM_Lumped_CalOnly", "+0.49"),
)
# Section 4.3, seed ensembles (no test): A, B, printed ensemble NSE of A and of B, printed
# difference ("" where the text gives none).
ENSEMBLE_CONTRASTS = (
    ("LSTM_DUH_Topmodel_PeOnly", "LSTM", "0.88", "0.84", "+0.04"),
    ("LSTM_DUH_SCS", "LSTM", "0.84", "0.84", ""),
)


@dataclass
class Run:
    """One completed run (configuration and seed)."""

    seed: int
    nse: float  # test_metrics.nse of results.json
    sim: np.ndarray  # simulated discharge of the test period, as stored (float32)
    obs: np.ndarray  # observed discharge of the test period, as stored (float32)


@dataclass(frozen=True)
class WilcoxonResult:
    """Exact Wilcoxon signed-rank test of the paired differences."""

    n: int  # number of non-zero differences
    w_plus: float  # sum of the ranks of the positive differences
    count: int  # p = count / 2**n
    p: float
    ties: bool  # tied absolute differences (average ranks were used)
    zeros: int  # zero differences dropped


@dataclass
class Outcome:
    """A contrast evaluated on the grid."""

    contrast: Contrast
    seeds: list[int]
    mean_a: float
    mean_b: float
    delta: float  # mean paired difference (positive: A better)
    wins: int  # seeds with a positive difference
    test: WilcoxonResult
    scipy_p: float | None  # scipy.stats.wilcoxon(method="exact"), when applicable


@dataclass
class Tally:
    """Comparison of recomputed values with the values printed in the manuscript."""

    compared: int = 0
    mismatches: list[str] = field(default_factory=list)

    def check(self, label: str, printed: str, data: str) -> bool:
        """Record one comparison and return whether the two texts are equal."""
        self.compared += 1
        if printed != data:
            self.mismatches.append(f"{label}: manuscript {printed}, data {data}")
        return printed == data


# ---------------------------------------------------------------------------- metrics


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency over the finite pairs."""
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


def flv(sim: np.ndarray, obs: np.ndarray) -> float:
    """Modified low-flow log-tail bias, Supplementary Eq. S1 (as ``_fdc`` of consolida_sim.py).

    100 (A_sim - A_obs) / |A_obs|, with A the sum of ln(Q + 0.001) over the k = floor(0.30 N)
    lowest values of each independently sorted series.
    """
    k = int(0.3 * len(obs))
    a_sim = np.log(np.sort(sim)[:k] + 1e-3).sum()
    a_obs = np.log(np.sort(obs)[:k] + 1e-3).sum()
    return float(100 * (a_sim - a_obs) / (abs(a_obs) + 1e-9))


def run_metrics(run: Run) -> dict[str, float]:
    """Per-seed values of the contrasted metrics, and the NSE recomputed in float64 (a check)."""
    ok = np.isfinite(run.sim) & np.isfinite(run.obs)
    sim, obs = run.sim[ok], run.obs[ok]
    return {
        "nse": run.nse,
        "lognse": log_nse(sim, obs),
        "flv": flv(sim, obs),
        "nse_recomputed": nse(sim.astype(np.float64), obs.astype(np.float64)),
    }


def ensemble_nse(runs: dict[int, Run]) -> float:
    """NSE of the seed ensemble (``_pred_mean`` and ``_nse_ensemble`` of the figure generator).

    The predictions of the seeds are averaged step by step, truncated to the shortest series
    (all have the same length here), and scored against the observed series of the first seed.
    """
    ordered = [runs[seed] for seed in sorted(runs)]
    n = min(len(run.sim) for run in ordered)
    mean_sim = np.mean([run.sim[:n] for run in ordered], axis=0)
    return nse(mean_sim, ordered[0].obs[:n])


# ---------------------------------------------------------------------------- test


def exact_wilcoxon(diff: np.ndarray, tail: str) -> WilcoxonResult:
    """Exact Wilcoxon signed-rank test of the null hypothesis of differences symmetric about 0.

    ``tail`` is "greater" (positive differences), "less" or "two-sided". Zero differences are
    dropped (Wilcoxon's convention, ``zero_method="wilcox"`` in scipy) and tied absolute
    differences receive average ranks. The null distribution of W+ is built exactly by counting
    the 2**n equally likely sign patterns (ranks are doubled so that half ranks stay integers).
    """
    d = np.asarray(diff, dtype=float)
    zeros = int(np.sum(d == 0))
    d = d[d != 0]
    n = d.size
    if n == 0 or n > 60:
        raise ValueError(f"the exact test needs 1 to 60 non-zero differences, got {n}")
    ranks = stats.rankdata(np.abs(d))
    doubled = np.rint(2 * ranks).astype(np.int64)
    w_obs = int(doubled[d > 0].sum())
    counts = np.zeros(int(doubled.sum()) + 1, dtype=np.int64)  # sign patterns by 2 W+
    counts[0] = 1
    for rank in doubled:
        shifted = np.zeros_like(counts)
        shifted[rank:] = counts[: counts.size - rank]
        counts = counts + shifted
    total = 1 << n
    upper, lower = int(counts[w_obs:].sum()), int(counts[: w_obs + 1].sum())
    if tail == "greater":
        count = upper
    elif tail == "less":
        count = lower
    elif tail == "two-sided":
        count = min(2 * min(upper, lower), total)
    else:
        raise ValueError(f"unknown tail: {tail}")
    ties = len(np.unique(np.abs(d))) < n
    return WilcoxonResult(n, w_obs / 2, count, count / total, ties, zeros)


def scipy_exact_p(diff: np.ndarray, tail: str) -> float | None:
    """p-value of scipy.stats.wilcoxon(method="exact"), for differences without zeros or ties."""
    d = np.asarray(diff, dtype=float)
    if np.any(d == 0) or len(np.unique(np.abs(d))) < d.size:
        return None
    return float(stats.wilcoxon(d, alternative=tail, method="exact").pvalue)


def evaluate(
    contrast: Contrast, values: dict[str, dict[int, dict[str, float]]], metric: str | None = None
) -> Outcome:
    """Evaluate one contrast over the seeds shared by A and B.

    The paired differences are positive when A is better: A minus B for NSE and logNSE,
    |FLV_B| - |FLV_A| for FLV. ``metric`` overrides the metric of the contrast (used to repeat
    the NSE contrasts with the NSE recomputed from the predictions).
    """
    key = metric or contrast.metric
    runs_a, runs_b = values[contrast.a], values[contrast.b]
    seeds = sorted(set(runs_a) & set(runs_b))
    if len(seeds) < 2:
        raise RuntimeError(f"{contrast.a} and {contrast.b} share fewer than two seeds")
    x = np.array([runs_a[seed][key] for seed in seeds])
    y = np.array([runs_b[seed][key] for seed in seeds])
    diff = np.abs(y) - np.abs(x) if contrast.metric == "flv" else x - y
    return Outcome(
        contrast=contrast,
        seeds=seeds,
        mean_a=float(x.mean()),
        mean_b=float(y.mean()),
        delta=float(diff.mean()),
        wins=int(np.sum(diff > 0)),
        test=exact_wilcoxon(diff, contrast.tail),
        scipy_p=scipy_exact_p(diff, contrast.tail),
    )


# ---------------------------------------------------------------------------- input


def seed_folders(grade: Path) -> list[tuple[int, Path]]:
    """Folders seedNN of the grid, by increasing seed."""
    found = []
    for path in grade.glob("seed*"):
        match = re.fullmatch(r"seed(\d+)", path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def load_runs(grade: Path, config: str) -> dict[int, Run]:
    """Completed runs of one configuration, by seed.

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
        metrics = json.loads(found[0].read_text(encoding="utf-8"))["test_metrics"]
        with np.load(found[0].with_name("predictions.npz")) as npz:
            sim, obs = npz["pred"][:, 0], npz["target"][:, 0]
        runs[seed] = Run(seed, float(metrics["nse"]), sim, obs)
    return runs


# ---------------------------------------------------------------------------- formatting


def round_half_up(value: float, decimals: int) -> Decimal:
    """Round half away from zero, from the shortest repr of the unrounded value."""
    step = Decimal(1).scaleb(-decimals)
    rounded = Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_UP)
    return abs(rounded) if rounded == 0 else rounded  # no negative zero


def fmt(value: float, decimals: int, signed: bool = False) -> str:
    """Rounded value as text; an explicit plus sign when ``signed`` and the value is not 0."""
    rounded = round_half_up(value, decimals)
    return f"{rounded:+}" if signed and rounded != 0 else f"{rounded}"


def p_text(p: float) -> str:
    """p-value rounded as in the manuscript: two decimals at or above 0.1, three below."""
    return fmt(p, 2 if p >= 0.1 else 3)


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


def observed(outcome: Outcome) -> dict[str, str]:
    """The quantities of ``Contrast.expected``, computed and formatted as in the text."""
    nd = METRIC_DECIMALS[outcome.contrast.metric]
    sign = "+" if outcome.delta > 0 else "-" if outcome.delta < 0 else "0"
    return {
        "mean_a": fmt(outcome.mean_a, nd),
        "mean_b": fmt(outcome.mean_b, nd),
        "delta": fmt(outcome.delta, nd, signed=True),
        "wins": f"{outcome.wins}/{len(outcome.seeds)}",
        "p": p_text(outcome.test.p),
        "sign": sign,
        "significant": "yes" if outcome.test.p < 0.05 else "no",
    }


def print_outcome(outcome: Outcome, tally: Tally) -> None:
    """Print one contrast (test, unrounded and rounded values) and check it against the text."""
    c, t = outcome.contrast, outcome.test
    nd = METRIC_DECIMALS[c.metric]
    if c.metric == "flv":
        name = f"{c.a} against {c.b}, reduction of |FLV|"
        delta_label, wins_label = "mean reduction of |FLV|", "smaller |FLV| for A"
    else:
        name = f"{c.a} minus {c.b}, {METRIC_LABEL[c.metric]}"
        delta_label, wins_label = "difference", "A > B"
    notes = []
    if t.zeros:
        notes.append(f"{t.zeros} zero difference(s) dropped")
    if t.ties:
        notes.append("tied absolute differences, average ranks")
    print(f"[{c.section}] {name}")
    print(f"      {c.title}")
    print(f"      exact Wilcoxon signed-rank test, {TAIL_LABEL[c.tail]}; {c.tail_source}")
    print(
        f"      seeds {number_list(outcome.seeds)} (n = {len(outcome.seeds)});"
        f" mean A {outcome.mean_a:+.6f} ({fmt(outcome.mean_a, nd)});"
        f" mean B {outcome.mean_b:+.6f} ({fmt(outcome.mean_b, nd)})"
    )
    print(
        f"      {delta_label} {outcome.delta:+.6f} ({fmt(outcome.delta, nd, signed=True)});"
        f" {wins_label} in {outcome.wins} of {len(outcome.seeds)} seeds; W+ = {t.w_plus:g};"
        f" p = {t.p:.6f} = {t.count}/{1 << t.n} ({p_text(t.p)})"
        + (f"; {'; '.join(notes)}" if notes else "")
    )
    data = observed(outcome)
    label = f"[{c.section}] {c.a} vs {c.b} ({c.metric})"
    results = [
        (item, printed, tally.check(f"{label} {ITEM_LABEL[item]}", printed, data[item]))
        for item, printed in c.expected.items()
    ]
    printed_text = ", ".join(f"{ITEM_LABEL[item]} {printed}" for item, printed, _ in results)
    differing = [f"{ITEM_LABEL[item]}: data {data[item]}" for item, _, ok in results if not ok]
    verdict = "all match" if not differing else "DIFFERS (" + "; ".join(differing) + ")"
    print(f"      manuscript: {printed_text} -> {verdict}")


def status(ok: bool) -> str:
    """'ok' or 'DIFFERS'."""
    return "ok" if ok else "DIFFERS"


# ---------------------------------------------------------------------------- blocks


def print_decomposition(values: dict[str, dict[int, dict[str, float]]], tally: Tally) -> None:
    """Section 4.2: configuration means and differences of the input decomposition."""
    print()
    print("  Input decomposition (differences of seed means of the NSE; no test in the text)")
    means = {
        name: float(np.mean([v["nse"] for v in values[name].values()]))
        for name in DECOMPOSITION_MEANS
    }
    for name, printed in DECOMPOSITION_MEANS.items():
        data = fmt(means[name], 2)
        ok = tally.check(f"[4.2] mean NSE of {name}", printed, data)
        print(
            f"    {name:<22} mean NSE {means[name]:+.6f} ({data});"
            f" manuscript {printed} -> {status(ok)}"
        )
    for label, a, b, printed in DECOMPOSITION_DIFFERENCES:
        diff = means[a] - means[b]
        data = fmt(diff, 2, signed=True)
        ok = tally.check(f"[4.2] {label} contribution", printed, data)
        print(
            f"    {label:<8} {a} minus {b} = {diff:+.6f} ({data});"
            f" manuscript {printed} -> {status(ok)}"
        )


def print_ensembles(runs: dict[str, dict[int, Run]], tally: Tally) -> None:
    """Section 4.3: ensemble NSE of the configurations compared in the text."""
    print()
    print("  Seed ensembles (NSE of the seed-mean series; no test in the text)")
    for a, b, printed_a, printed_b, printed_diff in ENSEMBLE_CONTRASTS:
        ens_a, ens_b = ensemble_nse(runs[a]), ensemble_nse(runs[b])
        diff = ens_a - ens_b
        oks = [
            tally.check(f"[4.3] ensemble NSE of {a}", printed_a, fmt(ens_a, 2)),
            tally.check(f"[4.3] ensemble NSE of {b}", printed_b, fmt(ens_b, 2)),
        ]
        if printed_diff:
            data = fmt(diff, 2, signed=True)
            oks.append(tally.check(f"[4.3] ensemble {a} minus {b}", printed_diff, data))
        print(
            f"    {a} {ens_a:.6f} ({fmt(ens_a, 2)}) minus {b} {ens_b:.6f} ({fmt(ens_b, 2)})"
            f" = {diff:+.6f} ({fmt(diff, 2, signed=True)})"
        )
        printed = f"{printed_a} against {printed_b}" + (f", {printed_diff}" if printed_diff else "")
        print(f"      manuscript {printed} -> {'all match' if all(oks) else 'DIFFERS'}")


def print_baseflow_flv(values: dict[str, dict[int, dict[str, float]]], tally: Tally) -> None:
    """Section 4.4: the baseflow variant 'reproduces the same FLV (4.2%)' as PeOnly."""
    print()
    print("  Related statement: the baseflow variant 'reproduces the same FLV (4.2%)' as PeOnly")
    for name in ("LSTM_DUH_Topmodel_Baseflow", "LSTM_DUH_Topmodel_PeOnly"):
        mean_flv = float(np.mean([v["flv"] for v in values[name].values()]))
        ok = tally.check(f"[4.4] mean FLV of {name}", "4.2", fmt(mean_flv, 1))
        print(
            f"    {name:<27} mean FLV {mean_flv:.6f} ({fmt(mean_flv, 1)});"
            f" manuscript 4.2 -> {status(ok)}"
        )


def print_manning_spread(outcome: Outcome, values: dict[str, dict[int, dict[str, float]]]) -> None:
    """Supplementary S2: 'the difference lies within the seed-to-seed spread'."""
    names = (outcome.contrast.a, outcome.contrast.b)
    sds = {name: float(np.std([v["nse"] for v in values[name].values()], ddof=1)) for name in names}
    within = abs(outcome.delta) < min(sds.values())
    print()
    print("  Related statement: 'the difference lies within the seed-to-seed spread'")
    print(
        f"    |difference| {abs(outcome.delta):.6f}; sample sd of the NSE: "
        + ", ".join(f"{name} {sd:.6f}" for name, sd in sds.items())
        + f" -> {'smaller than both' if within else 'NOT smaller than both'}"
    )


def print_checks(
    runs: dict[str, dict[int, Run]],
    values: dict[str, dict[int, dict[str, float]]],
    outcomes: list[Outcome],
    tally: Tally,
) -> None:
    """Consistency checks of the input and of the test, and the summary against the text."""
    all_runs = [run for found in runs.values() for run in found.values()]
    lengths = sorted({len(run.obs) for run in all_runs})
    finite = sorted({int(np.sum(np.isfinite(run.sim) & np.isfinite(run.obs))) for run in all_runs})
    reference = all_runs[0].obs
    same_obs = all(np.array_equal(run.obs, reference, equal_nan=True) for run in all_runs)
    nse_gap = max(
        abs(per_seed["nse"] - per_seed["nse_recomputed"])
        for found in values.values()
        for per_seed in found.values()
    )
    recomputed_same = True
    for outcome in outcomes:
        if outcome.contrast.metric == "nse":
            alt = evaluate(outcome.contrast, values, metric="nse_recomputed")
            recomputed_same &= (alt.wins, alt.test.count) == (outcome.wins, outcome.test.count)
    scipy_gaps = [abs(o.scipy_p - o.test.p) for o in outcomes if o.scipy_p is not None]
    irregular = [o for o in outcomes if o.test.zeros or o.test.ties]
    flv_values = [
        values[name][seed]["flv"]
        for o in outcomes
        if o.contrast.metric == "flv"
        for name in (o.contrast.a, o.contrast.b)
        for seed in o.seeds
    ]
    seed_sets: dict[tuple[int, ...], list[str]] = {}
    for config, found in runs.items():
        seed_sets.setdefault(tuple(sorted(found)), []).append(config)

    print()
    print("Checks")
    print(f"  Runs read: {len(all_runs)} ({len(runs)} configurations)")
    for seeds, names in seed_sets.items():
        print(f"    seeds {number_list(list(seeds))}: {len(names)} configurations")
    print(
        f"  Test time steps per run: {', '.join(map(str, lengths))};"
        f" finite pairs per run: {', '.join(map(str, finite))}"
    )
    print(f"  Observed series identical in all runs: {'yes' if same_obs else 'NO'}")
    print(
        "  NSE of results.json against NSE recomputed from predictions.npz in float64:"
        f" max |difference| = {nse_gap:.1e}"
    )
    print(
        "    the NSE contrasts repeated with the recomputed NSE give the same seed counts and"
        f" p-values: {'yes' if recomputed_same else 'NO'}"
    )
    if scipy_gaps:
        print(
            f"  scipy.stats.wilcoxon(method='exact') on {len(scipy_gaps)} of {len(outcomes)}"
            f" contrasts: max |difference in p| = {max(scipy_gaps):.1e}"
        )
    print(
        "  Contrasts with zero or tied differences: "
        + (", ".join(f"{o.contrast.a} vs {o.contrast.b}" for o in irregular) or "none")
    )
    if flv_values:
        positive = all(value > 0 for value in flv_values)
        print(
            "  FLV of the runs in the FLV contrasts: "
            + ("all positive, so |FLV| = FLV" if positive else "NOT all positive")
        )
    print(
        f"  Manuscript: {tally.compared - len(tally.mismatches)} of {tally.compared} printed"
        " values match the values recomputed here."
    )
    for line in tally.mismatches:
        print(f"    DIFFERS: {line}")


def write_csv(path: Path, outcomes: list[Outcome]) -> None:
    """Write one row per contrast, unrounded and rounded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # fmt: off
        writer.writerow([
            "section", "a", "b", "metric", "tail", "tail_source", "n", "seeds",
            "mean_a", "mean_b", "delta", "wins", "w_plus", "p_count", "p_total", "p",
            "mean_a_rounded", "mean_b_rounded", "delta_rounded", "p_rounded",
            "manuscript", "matches_manuscript",
        ])
        # fmt: on
        for o in outcomes:
            c, nd = o.contrast, METRIC_DECIMALS[o.contrast.metric]
            data = observed(o)
            # fmt: off
            writer.writerow([
                c.section, c.a, c.b, c.metric, c.tail, c.tail_source, len(o.seeds),
                " ".join(map(str, o.seeds)), repr(o.mean_a), repr(o.mean_b), repr(o.delta),
                o.wins, o.test.w_plus, o.test.count, 1 << o.test.n, repr(o.test.p),
                fmt(o.mean_a, nd), fmt(o.mean_b, nd), fmt(o.delta, nd, signed=True),
                p_text(o.test.p),
                "; ".join(f"{item}={value}" for item, value in c.expected.items()),
                all(data[item] == value for item, value in c.expected.items()),
            ])
            # fmt: on


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
        "--csv",
        nargs="?",
        type=Path,
        const=DEFAULT_CSV,
        default=None,
        metavar="PATH",
        help="also write the contrasts to a CSV file (default path: outputs/tables/contrasts.csv)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 if any value differs from the manuscript",
    )
    args = parser.parse_args()

    configs = sorted(
        {c.a for c in CONTRASTS}
        | {c.b for c in CONTRASTS}
        | set(DECOMPOSITION_MEANS)
        | {name for row in ENSEMBLE_CONTRASTS for name in row[:2]}
    )
    runs = {config: load_runs(args.grade, config) for config in configs}
    missing = [config for config, found in runs.items() if not found]
    if missing:
        parser.error(f"no completed runs in {args.grade} for: {', '.join(missing)}")
    values = {
        config: {seed: run_metrics(run) for seed, run in found.items()}
        for config, found in runs.items()
    }

    print("Seed-paired contrasts of Sections 4.2 to 4.4 and Supplementary Section S2")
    print("Test period, telemetric product. Differences A minus B formed seed by seed; exact")
    print("Wilcoxon signed-rank test; p-values unadjusted. Values unrounded and (rounded).")
    print(f"Grid: {display_path(args.grade)}")

    tally = Tally()
    outcomes: list[Outcome] = []
    for section, heading in SECTIONS.items():
        print()
        print(heading)
        if section == "4.2":
            print_decomposition(values, tally)
        for contrast in (c for c in CONTRASTS if c.section == section):
            outcome = evaluate(contrast, values)
            outcomes.append(outcome)
            print()
            print_outcome(outcome, tally)
        if section == "4.3":
            print_ensembles(runs, tally)
        elif section == "4.4":
            print_baseflow_flv(values, tally)
        elif section == "S2":
            for outcome in (o for o in outcomes if o.contrast.section == "S2"):
                print_manning_spread(outcome, values)

    print_checks(runs, values, outcomes, tally)

    if args.csv is not None:
        write_csv(args.csv, outcomes)
        print(f"\nWrote {display_path(args.csv)}")

    return 1 if args.strict and tally.mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
