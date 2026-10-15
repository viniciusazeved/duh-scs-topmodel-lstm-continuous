#!/usr/bin/env python
"""Score the GR4H test simulations at the scored hours of the neural models (CPU, no refitting).

Manuscript Section 4.6 and Supplementary Information Section S6. The GR4H benchmark, calibrated on
NSE and, as a control, on KGE (``scripts/gr4h/run_gr4h_airgr.R`` and ``run_gr4h_airgr_kge.R``), is
evaluated at the same 6,323 valid hourly timestamps as the neural models, from 10 April 2025 01:00
to 31 December 2025 00:00 UTC, with the formulas and units of the neural evaluation. The script

1. rebuilds the neural evaluation mask from the test split of ``data/dataset_58585000_telem.h5``
   with the rule of ``AblationDataset._create_valid_indices`` in ``scripts/train.py`` (lookback
   240 h, horizon 1 h): a window is scored when the discharge of its target hour and the
   precipitation of all its 240 input hours are finite;
2. checks that the observed discharge of the dataset at those hours (float32) is exactly the
   ``target`` of the 30 ``predictions.npz`` of LSTM, LSTM_Lumped and LSTM_DUH_Topmodel_PeOnly
   (seeds 42 to 51), and recomputes the test NSE of LSTM seed 42 against its results.json;
3. joins the GR4H simulations (mm/h) to the scored hours and converts them to m3/s with the
   3,117 km2 area that produced the calibration inputs, Q[m3/s] = Q[mm/h] * 3117 / 3.6; the
   observed discharge exported by the R scripts, converted back, must match the dataset within
   the six-decimal resolution of the CSV;
4. computes NSE, KGE (and its r), RMSE, PBIAS, FHV, FLV and logNSE with the functions of the
   neural evaluation: ``compute_metrics`` of ``scripts/train.py`` and ``_fdc`` and ``_lognse`` of
   ``scripts/consolida_sim.py``. FLV is the modified log-tail signature of Supplementary Section
   S11, in m3/s. As sensitivity only, the report adds the metrics over all finite hours of the
   GR4H CSV (6,561 h, in mm/h and in m3/s) and with the exact sum of the sub-catchment areas
   (3,117.13 km2) instead of 3,117 km2.

The three functions are taken from the source files by parsing them and executing only their
definitions, because importing ``train.py`` would import torch and create ``outputs/ablation``.

Values are printed unrounded and rounded half away from zero from the unrounded value: to two
decimals and, where the manuscript quotes another precision, also to that precision, next to the
value printed in the manuscript. The calibration criteria come from ``gr4h_parametros*.txt``, where
the R scripts store them with four decimals (the R scripts also print them unrounded).

Reads ``data/dataset_58585000_telem.h5``, ``outputs/grade_sim`` (results.json and
predictions.npz), ``scripts/train.py``, ``scripts/consolida_sim.py`` and, in ``--sim-dir``
(default ``outputs/gr4h``), ``gr4h_sim_teste.csv``, ``gr4h_sim_teste_kge.csv``,
``gr4h_parametros.txt`` and ``gr4h_parametros_kge.txt``. Writes to ``--out`` (default
``outputs/retrain/gr4h``, not versioned; the versioned copies are in ``outputs/gr4h``):
``auditoria_gr4h.json`` (metrics, checks and the SHA-256 of every input), ``predicoes_mascara_comum.csv`` (observed and GR4H discharge in
m3/s at the 6,323 scored hours) and ``horas_validas_excluidas_GR4H_NSE.csv`` and ``_KGE.csv`` (the
finite hours of the GR4H CSV outside the mask).

Usage, from the repository root (the exit status is 1 if a recomputed value differs from the
manuscript):
    python scripts/gr4h/audit_gr4h_common_mask.py   # scores the versioned simulations
    python scripts/gr4h/audit_gr4h_common_mask.py --sim-dir outputs/retrain/gr4h

The second form scores a new calibration made by the two R scripts, which write to
``outputs/retrain/gr4h`` by default.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H5 = ROOT / "data" / "dataset_58585000_telem.h5"
DEFAULT_GRADE = ROOT / "outputs" / "grade_sim"
DEFAULT_SIM_DIR = ROOT / "outputs" / "gr4h"
DEFAULT_OUT = ROOT / "outputs" / "retrain" / "gr4h"
TRAIN_PY = ROOT / "scripts" / "train.py"
CONSOLIDA_PY = ROOT / "scripts" / "consolida_sim.py"
R_SCRIPTS = (
    ROOT / "scripts" / "gr4h" / "run_gr4h_airgr.R",
    ROOT / "scripts" / "gr4h" / "run_gr4h_airgr_kge.R",
)
GR4H_INPUT = ROOT / "data" / "gr4h" / "serie_horaria_gr4h.csv"

MODELS = ("LSTM", "LSTM_Lumped", "LSTM_DUH_Topmodel_PeOnly")  # neural pairs of Section 4.6
SEEDS = range(42, 52)
LOOKBACK = 240  # hours of forcing in each window (Table 2)
HORIZON = 1
AREA_KM2 = 3117.0  # area that converted the observed discharge to mm/h for the R scripts
CALIBRATIONS = (("NSE", ""), ("KGE", "_kge"))  # criterion, suffix of the R output files

# Values printed in the manuscript: (calibration, quantity) -> (text, decimals, where).
MANUSCRIPT = {
    ("NSE", "calibration"): ("0.75", 2, "Section 4.6; SI S6"),
    ("NSE", "nse"): ("0.45", 2, "Section 4.6; SI S6"),
    ("NSE", "kge"): ("0.45", 2, "SI S6"),
    ("NSE", "flv"): ("+0.3", 1, "SI S6, %"),
    ("NSE", "fhv"): ("-42.3", 1, "SI S6, %"),
    ("KGE", "nse"): ("0.48", 2, "Section 4.6; SI S6"),
    ("KGE", "kge"): ("0.50", 2, "SI S6"),
    ("KGE", "flv"): ("-3.1", 1, "SI S6, %"),
    ("KGE", "fhv"): ("-40.1", 1, "SI S6, %"),
}
MANUSCRIPT_MASK = {
    "count": 6323,
    "first_utc": "2025-04-10T01:00:00+00:00",
    "last_utc": "2025-12-31T00:00:00+00:00",
}  # Section 4.6; SI S6

Metrics = dict[str, float]


def display_path(path: Path) -> str:
    """Path relative to the repository root when possible."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_functions(path: Path, names: Sequence[str]) -> dict[str, Callable]:
    """Execute only the named top-level function definitions of a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError(f"functions {sorted(names)} not all found in {display_path(path)}")
    namespace: dict = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def dates_summary(timestamps: pd.DatetimeIndex) -> dict:
    """Count, first and last timestamp (ISO 8601, UTC)."""
    return {
        "count": len(timestamps),
        "first_utc": timestamps[0].isoformat() if len(timestamps) else None,
        "last_utc": timestamps[-1].isoformat() if len(timestamps) else None,
    }


def read_parameters(path: Path) -> dict[str, float]:
    """X1 to X4 and the calibration criterion stored by the R scripts (KEY=value lines)."""
    values = {}
    for line in path.read_text(encoding="utf-8").split():
        key, value = line.split("=")
        values[key] = float(value)
    return values


def round_half_up(value: float, decimals: int) -> Decimal:
    """Round half away from zero, from the shortest repr of the unrounded value."""
    step = Decimal(1).scaleb(-decimals)
    rounded = Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_UP)
    return abs(rounded) if rounded == 0 else rounded  # no negative zero


def neural_mask(flow: np.ndarray, precip: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """Target indices of the scored windows, as AblationDataset._create_valid_indices."""
    starts, rejected_target, rejected_precip = [], [], []
    for index in range(len(flow) - LOOKBACK - HORIZON + 1):
        if np.isnan(flow[index + LOOKBACK : index + LOOKBACK + HORIZON]).any():
            rejected_target.append(index + LOOKBACK)
            continue
        if np.isnan(precip[index : index + LOOKBACK]).any():
            rejected_precip.append(index + LOOKBACK)
            continue
        starts.append(index)
    return np.array(starts, dtype=np.int64) + LOOKBACK, rejected_target, rejected_precip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5, help="canonical dataset")
    parser.add_argument(
        "--grade",
        type=Path,
        default=DEFAULT_GRADE,
        help="results of the neural grid (outputs/grade_sim)",
    )
    parser.add_argument(
        "--sim-dir",
        type=Path,
        default=DEFAULT_SIM_DIR,
        help="folder with the outputs of the two R scripts",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="folder for the report and the CSVs (default outputs/retrain/gr4h)",
    )
    return parser.parse_args()


def run(args: argparse.Namespace, report: dict) -> None:
    """Fill the report and write the CSVs; raise on any failed check."""
    out = args.out
    compute_metrics = load_functions(TRAIN_PY, ["compute_metrics"])["compute_metrics"]
    signatures = load_functions(CONSOLIDA_PY, ["_fdc", "_lognse"])

    def metrics(pred: np.ndarray, target: np.ndarray) -> Metrics:
        if not np.isfinite(pred).all() or not np.isfinite(target).all():
            raise RuntimeError("non-finite values passed to the metrics")
        result = {key: float(value) for key, value in compute_metrics(pred, target).items()}
        result["fhv"], result["flv"] = signatures["_fdc"](pred, target)
        result["lognse"] = signatures["_lognse"](pred, target)
        return result

    def record(path: Path) -> None:
        report["input_hashes_sha256"][display_path(path)] = sha256(path)

    for path in [Path(__file__), args.h5, TRAIN_PY, CONSOLIDA_PY, *R_SCRIPTS, GR4H_INPUT]:
        record(path)

    # GR4H simulations over all finite hours of the CSV (the first computation, in mm/h).
    frames: dict[str, pd.DataFrame] = {}
    for calibration, suffix in CALIBRATIONS:
        csv = args.sim_dir / f"gr4h_sim_teste{suffix}.csv"
        parameters = args.sim_dir / f"gr4h_parametros{suffix}.txt"
        record(csv)
        record(parameters)
        values = read_parameters(parameters)
        report["calibration"][calibration] = {
            "parameters": {key: values[key] for key in ("X1", "X2", "X3", "X4")},
            "criterion": calibration,
            "criterion_value_4_decimals": values[f"{calibration}_cal"],
        }
        frame = pd.read_csv(csv)
        frame["timestamp_utc"] = pd.to_datetime(frame["Dates"], utc=True, errors="raise")
        if (
            frame["timestamp_utc"].duplicated().any()
            or not frame["timestamp_utc"].is_monotonic_increasing
        ):
            raise RuntimeError(f"duplicated or unsorted timestamps in {display_path(csv)}")
        frame = frame.set_index("timestamp_utc")
        valid = (np.isfinite(frame["Qsim_mm"]) & np.isfinite(frame["Qobs_mm"])).to_numpy()
        report["original_reproduction"][calibration] = {
            "unit": "mm/h",
            "metrics": metrics(
                frame["Qsim_mm"].to_numpy()[valid], frame["Qobs_mm"].to_numpy()[valid]
            ),
            "all_csv": dates_summary(frame.index),
            "finite_csv": dates_summary(frame.index[valid]),
            "note": (
                "all finite hours of the GR4H CSV, as first computed before the alignment to "
                "the neural mask; superseded by common_metrics"
            ),
        }
        frames[calibration] = frame

    # Neural evaluation mask of the test split.
    with h5py.File(args.h5, "r") as handle:
        flow = handle["test/streamflow"][:]
        precip = handle["test/precipitation"][:]
        timestamps = pd.to_datetime(handle["test/timestamps"][:], unit="s", utc=True)
        area = float(handle.attrs["area_total_km2"])
    if timestamps.has_duplicates:
        raise RuntimeError("duplicated timestamps in the test split of the dataset")
    indices, rejected_target, rejected_precip = neural_mask(flow, precip)
    target = np.asarray(flow[indices], dtype=np.float32).reshape(-1)
    if not np.isfinite(target).all():
        raise RuntimeError("the mask admitted a non-finite target")
    neural_dates = timestamps[indices]
    report["mask"] = {
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "test_h5": dates_summary(timestamps),
        "neural": dates_summary(neural_dates),
        "excluded_initial_lookback": LOOKBACK,
        "rejected_target_nan": len(rejected_target),
        "rejected_precip_window_nan": len(rejected_precip),
        "rule": (
            "a NaN target rejects the window; a NaN in any precipitation of its 240 input hours "
            "rejects it"
        ),
    }

    # The observed series of the mask is the target of the neural predictions.
    equality = []
    neural_nse_check = None
    for seed in SEEDS:
        for model in MODELS:
            model_root = args.grade / f"seed{seed}" / model
            candidates = list(model_root.glob("**/predictions.npz"))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"expected one predictions.npz in {display_path(model_root)}, "
                    f"found {len(candidates)}"
                )
            npz_path = candidates[0]
            result_path = npz_path.parent / "results.json"
            record(npz_path)
            record(result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["config"]["lookback"] != LOOKBACK or result["config"]["horizon"] != HORIZON:
                raise RuntimeError(f"unexpected lookback or horizon in {display_path(result_path)}")
            with np.load(npz_path) as data:
                observed = data["target"].reshape(-1)
                predicted = data["pred"].reshape(-1)
            if observed.dtype != np.float32 or not np.array_equal(observed, target):
                raise RuntimeError(f"NPZ target is not the float32 dataset series: {npz_path}")
            if not np.isfinite(predicted).all():
                raise RuntimeError(f"non-finite neural prediction: {npz_path}")
            if seed == 42 and model == "LSTM":
                computed = float(compute_metrics(predicted, observed)["nse"])
                stored = result["test_metrics"]["nse"]
                if abs(computed - stored) > 1e-7:
                    raise RuntimeError("test NSE of LSTM seed 42 not reproduced")
                neural_nse_check = {
                    "computed": computed,
                    "stored": stored,
                    "delta": computed - stored,
                }
            equality.append(
                {"seed": seed, "model": model, "count": len(observed), "target_float32_exact": True}
            )
    report["checks"]["npz_target_equality"] = equality
    report["checks"]["neural_seed42_NSE_reproduction"] = neural_nse_check

    # Conversion: 3117 km2 inverts the Qmm of the R inputs; the exact area is a sensitivity.
    factor = AREA_KM2 / 3.6
    exact_factor = area / 3.6
    report["units"] = {
        "area_exact_h5_km2": area,
        "area_historical_export_km2": AREA_KM2,
        "area_primary_inverse_export_km2": AREA_KM2,
        "mm_per_hour_to_m3_per_second_factor": factor,
        "sensitivity_exact_h5_factor": exact_factor,
        "FLV_original": "mm/h, eps=1e-3",
        "FLV_common": "m3/s, eps=1e-3",
        "target_tolerance": (
            "abs(delta) <= 0.5e-6*3117/3.6 + 2e-5 m3/s (CSV resolution and float32)"
        ),
        "common_observation": (
            "test streamflow of the dataset as float32, identical to the target of the 30 NPZ"
        ),
        "conversion_decision": (
            "3117 km2 inverts the Qmm series used by the R scripts and keeps the existing "
            "simulations; the exact area of the dataset rescales them and is reported as "
            "sensitivity only"
        ),
    }

    report["common_metrics"] = {}
    common = pd.DataFrame({"timestamp_utc": neural_dates, "Qobs_m3s": target})
    for calibration, frame in frames.items():
        if not neural_dates.isin(frame.index).all():
            raise RuntimeError(f"GR4H {calibration} does not cover the whole neural mask")
        joined = frame.loc[neural_dates]
        if not np.isfinite(joined[["Qsim_mm", "Qobs_mm"]].to_numpy()).all():
            raise RuntimeError(f"GR4H {calibration} has non-finite values in the neural mask")
        converted_target = joined["Qobs_mm"].to_numpy() * factor
        tolerance = np.full(len(target), 0.5e-6 * factor + 2e-5)
        delta = np.abs(converted_target - target)
        if not np.all(delta <= tolerance):
            raise RuntimeError(
                f"observed discharge of the CSV differs from the dataset in "
                f"{calibration}: max {delta.max()}"
            )
        prediction = joined["Qsim_mm"].to_numpy() * factor
        report["checks"][calibration + "_join"] = {
            "timestamps_unique": True,
            "full_neural_coverage": True,
            "finite_all_neural_hours": True,
            "target_max_abs_error_m3s": float(delta.max()),
            "target_max_error_over_tolerance": float(np.max(delta / tolerance)),
        }
        all_valid = (np.isfinite(frame["Qsim_mm"]) & np.isfinite(frame["Qobs_mm"])).to_numpy()
        outside = all_valid & ~frame.index.isin(neural_dates)
        report["common_metrics"][calibration] = {
            "unit": "m3/s",
            "metrics": metrics(prediction, target.astype(np.float64)),
            "before_all_valid_m3s": metrics(
                frame["Qsim_mm"].to_numpy()[all_valid] * factor,
                frame["Qobs_mm"].to_numpy()[all_valid] * factor,
            ),
            "after": dates_summary(neural_dates),
            "excluded_valid_csv_hours": dates_summary(frame.index[outside]),
            "exact_h5_area_sensitivity": {
                "area_km2": area,
                "interpretation": (
                    "explicit rescaling of the existing simulations, not the main result"
                ),
                "metrics": metrics(
                    joined["Qsim_mm"].to_numpy() * exact_factor, target.astype(np.float64)
                ),
            },
        }
        common["GR4H_" + calibration + "_m3s"] = prediction
        frame.loc[outside].copy().to_csv(out / f"horas_validas_excluidas_GR4H_{calibration}.csv")
    common.to_csv(out / "predicoes_mascara_comum.csv", index=False)


def compare_with_manuscript(report: dict) -> list[dict]:
    """Rows of recomputed against printed values."""
    rows = []
    mask = report["mask"]["neural"]
    for key, expected in MANUSCRIPT_MASK.items():
        rows.append(
            {
                "quantity": f"scored test hours, {key}",
                "manuscript": expected,
                "recomputed": mask[key],
                "match": mask[key] == expected,
                "where": "Section 4.6; SI S6",
            }
        )
    for (calibration, quantity), (text, decimals, where) in MANUSCRIPT.items():
        if quantity == "calibration":
            value = report["calibration"][calibration]["criterion_value_4_decimals"]
            label = f"GR4H {calibration} calibration, calibration {calibration} (4 decimals)"
        else:
            value = report["common_metrics"][calibration]["metrics"][quantity]
            label = f"GR4H {calibration} calibration, test {quantity.upper()}"
        rounded = round_half_up(value, decimals)
        rows.append(
            {
                "quantity": label,
                "manuscript": text,
                "recomputed": value,
                "recomputed_rounded": str(rounded),
                "decimals": decimals,
                "match": rounded == Decimal(text),
                "where": where,
            }
        )
    return rows


def print_summary(report: dict) -> None:
    """Numbers of Section 4.6 and SI S6, unrounded and rounded, against the manuscript."""
    mask = report["mask"]
    print(
        f"test split of the dataset: {mask['test_h5']['count']:,} h, "
        f"{mask['test_h5']['first_utc']} to {mask['test_h5']['last_utc']}"
    )
    print(
        f"neural mask: {mask['neural']['count']:,} scored h, {mask['neural']['first_utc']} to "
        f"{mask['neural']['last_utc']} (first {mask['excluded_initial_lookback']} h of lookback, "
        f"{mask['rejected_target_nan']} missing targets, "
        f"{mask['rejected_precip_window_nan']} windows with missing rainfall)"
    )
    checks = report["checks"]
    print(
        f"NPZ targets equal to the dataset series: {len(checks['npz_target_equality'])} of "
        f"{len(MODELS) * len(SEEDS)}; LSTM seed 42 test NSE recomputed "
        f"{checks['neural_seed42_NSE_reproduction']['computed']!r} = stored "
        f"{checks['neural_seed42_NSE_reproduction']['stored']!r}"
    )
    for calibration, _ in CALIBRATIONS:
        join = checks[calibration + "_join"]
        cal = report["calibration"][calibration]
        print(
            f"\nGR4H calibrated on {calibration}: X1 to X4 = "
            + ", ".join(f"{v:g}" for v in cal["parameters"].values())
            + f"; calibration {calibration} {cal['criterion_value_4_decimals']:.4f} "
            f"(stored with 4 decimals) -> "
            f"{round_half_up(cal['criterion_value_4_decimals'], 2)}"
        )
        print(
            f"  observed discharge, CSV against dataset: max |difference| "
            f"{join['target_max_abs_error_m3s']:.2e} m3/s "
            f"({join['target_max_error_over_tolerance']:.3f} of the tolerance)"
        )
        common = report["common_metrics"][calibration]
        print(f"  test, {common['after']['count']:,} scored hours, m3/s:")
        for key, label in (
            ("nse", "NSE"),
            ("kge", "KGE"),
            ("r", "r"),
            ("rmse", "RMSE m3/s"),
            ("pbias", "PBIAS %"),
            ("fhv", "FHV %"),
            ("flv", "FLV %"),
            ("lognse", "logNSE"),
        ):
            value = common["metrics"][key]
            line = f"    {label:<10}{value!r:>24}  2 decimals {round_half_up(value, 2)!s:>7}"
            quoted = MANUSCRIPT.get((calibration, key))
            if quoted is not None and quoted[1] != 2:
                line += f"  {quoted[1]} decimal(s) {round_half_up(value, quoted[1])!s:>6}"
            if quoted is not None:
                line += f"  manuscript {quoted[0]} ({quoted[2]})"
            print(line)
        before = common["before_all_valid_m3s"]
        exact = common["exact_h5_area_sensitivity"]["metrics"]
        n_finite = report["original_reproduction"][calibration]["finite_csv"]["count"]
        print(
            f"  sensitivity, all {n_finite:,} finite CSV hours: "
            f"NSE {before['nse']!r} ({round_half_up(before['nse'], 2)}), "
            f"KGE {before['kge']!r} ({round_half_up(before['kge'], 2)})"
        )
        print(
            f"  sensitivity, exact area {report['units']['area_exact_h5_km2']:.2f} km2: "
            f"NSE {exact['nse']!r} ({round_half_up(exact['nse'], 2)}), "
            f"KGE {exact['kge']!r} ({round_half_up(exact['kge'], 2)})"
        )
    print("\nagainst the manuscript:")
    for row in report["manuscript_comparison"]:
        shown = row.get("recomputed_rounded", row["recomputed"])
        print(
            f"  {'match' if row['match'] else 'MISMATCH':<9}{row['quantity']}: manuscript "
            f"{row['manuscript']}, recomputed {shown} ({row['where']})"
        )


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
        },
        "protocol": (
            "CPU only; existing GR4H simulations and neural predictions; no retraining or "
            "recalibration; the neural evaluation mask is not modified"
        ),
        "input_hashes_sha256": {},
        "calibration": {},
        "original_reproduction": {},
        "checks": {},
    }
    try:
        run(args, report)
        report["manuscript_comparison"] = compare_with_manuscript(report)
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "blocked"
        report["error"] = str(error)
        raise
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report_path = args.out / "auditoria_gr4h.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)
    print(
        f"\nwritten to {display_path(args.out)}: auditoria_gr4h.json, "
        "predicoes_mascara_comum.csv, horas_validas_excluidas_GR4H_NSE.csv, "
        "horas_validas_excluidas_GR4H_KGE.csv"
    )
    return 0 if all(row["match"] for row in report["manuscript_comparison"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
