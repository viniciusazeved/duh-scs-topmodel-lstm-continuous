#!/usr/bin/env python
"""Reproduce Table S4 of the Supplementary Information: metric panel over the validation period.

No retraining. The script reloads the ``best_model.pt`` checkpoints of ``outputs/grade_sim`` (10
seeds) of rungs 4 (LSTM, pure neural), 9 (LSTM_DUH_Topmodel) and 10 (LSTM_DUH_Topmodel_PeOnly),
runs inference over the canonical validation loader of ``scripts/train.py`` (30 June 2024 to
31 March 2025, 240 h windows, horizon 1 h) and computes the same metrics as in the test
evaluation: NSE and KGE with ``compute_metrics`` of ``scripts/train.py`` (through ``evaluate``),
and logNSE, PBIAS, FHV and FLV with ``assinaturas`` of ``scripts/consolida_sim.py``, the
implementation behind Table 3. FLV is the modified low-flow log-tail bias of Supplementary Section
S11 (Eq. S1). Table S4 reports, for each configuration, the mean and the population standard
deviation (divisor 10, ddof = 0) across seeds of NSE, KGE, FLV and FHV.

Built-in sanity check: the recomputed validation NSE of each run must agree with the
``best_val_nse`` recorded by training in its ``results.json`` (same checkpoint, same split);
differences above 0.01 are flagged. Inference on another device than the one used in training
(for example the CPU) changes the NSE of a run by a few 1e-4 at most (1.2e-4 for these 30 runs on
a CPU).

Origin: ``revisao2_paper1/c3_painel_validacao.py`` of the original project, with paths relative
to the repository, a choice of device and a new output file; the computation is unchanged. Its
output, transcribed from the standard output of the original job (three decimals for NSE and KGE,
one for FLV and FHV), is kept as a record in ``outputs/analysis/c3_painel_validacao.json`` and
``.csv``. This script never overwrites that record. It writes the unrounded values to ``--out``
(default ``outputs/tables/validation_panel.json``, not versioned) and compares them with Table S4
and with the record. The values of a run on a CPU (``--device cpu``) are versioned in
``outputs/analysis/validation_panel_recomputed.json``.

Rounded values are rounded half away from zero directly from the unrounded value (two decimals
for NSE and KGE, one for FLV and FHV, as in Table S4), never from an already rounded number.

Usage, from the repository root (needs torch; about 40 min on a laptop CPU, faster on a GPU):
    python scripts/analysis/validation_panel.py                 # CUDA if available, else CPU
    python scripts/analysis/validation_panel.py --device cpu    # CPU only (GPUs hidden)
    python scripts/analysis/validation_panel.py --grade outputs/retrain --out <file.json>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
H5 = ROOT / "data" / "dataset_58585000_telem.h5"
GRADE = ROOT / "outputs" / "grade_sim"
OUT = ROOT / "outputs" / "tables" / "validation_panel.json"
# Output of the original run, transcribed from its standard output: a record, never overwritten.
RECORD = ROOT / "outputs" / "analysis" / "c3_painel_validacao.json"
RECORD_CSV = RECORD.with_suffix(".csv")
# Default output folder of scripts/train.py, which creates it when the module is imported.
TRAIN_IMPORT_DIR = ROOT / "outputs" / "ablation"

# (rung in Tables 1, 3 and S4, configuration folder, model_type of the model factory)
CONFIGS = [
    (4, "LSTM", "lstm"),
    (9, "LSTM_DUH_Topmodel", "lstm_duh_base_topmodel"),
    (10, "LSTM_DUH_Topmodel_PeOnly", "lstm_duh_base_topmodel_peonly"),
]
SEEDS = list(range(42, 52))
LOOKBACK, HORIZON, BATCH_SIZE = 240, 1, 512  # windows of the continuous simulation (Table 2)
METRICS = ("nse", "kge", "lognse", "pbias", "fhv", "flv")
SANITY_TOLERANCE = 0.01  # |recomputed validation NSE - best_val_nse| above this is flagged

# Columns of Table S4: metric key, label, decimals printed, explicit sign on the mean.
COLUMNS = (
    ("nse", "NSE", 2, False),
    ("kge", "KGE", 2, False),
    ("flv", "FLV (%)", 1, True),
    ("fhv", "FHV (%)", 1, True),
)

# Table S4 as printed in the submitted Supplementary Information: mean and population standard
# deviation of each column, in the order of COLUMNS. Used only to report whether the recomputed
# values match.
MANUSCRIPT_TABLE_S4 = {
    "LSTM": ("0.75", "0.03", "0.77", "0.04", "+10.1", "1.7", "-19.3", "6.5"),
    "LSTM_DUH_Topmodel": ("0.73", "0.04", "0.74", "0.03", "+10.2", "1.6", "-22.8", "4.0"),
    "LSTM_DUH_Topmodel_PeOnly": ("0.69", "0.03", "0.74", "0.03", "+9.6", "1.5", "-21.6", "5.4"),
}
MANUSCRIPT_VALIDATION_WINDOWS = 5998  # Section 2.5 of the manuscript


# ---------------------------------------------------------------------------- set-up


@contextmanager
def skip_mkdir(folder: Path) -> Iterator[None]:
    """Make ``Path.mkdir`` a no-op for ``folder``, and for it only, while the block runs."""
    original = Path.mkdir
    target = folder.resolve()

    def mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.resolve() != target:
            original(self, *args, **kwargs)

    Path.mkdir = mkdir  # type: ignore[method-assign]
    try:
        yield
    finally:
        Path.mkdir = original  # type: ignore[method-assign]


def import_project_code() -> tuple[ModuleType, ModuleType, Callable[..., torch.nn.Module]]:
    """Import scripts/train.py, scripts/consolida_sim.py and the model factory of src/.

    src/ goes first on sys.path, so that the package of this repository is used even when another
    copy of ttd_scs_lstm is installed in the environment. scripts/train.py creates its default
    output folder, outputs/ablation, when it is imported; the panel only reads, so that folder is
    not created.
    """
    for folder in (SCRIPTS, SRC):  # SRC ends up first
        sys.path.insert(0, str(folder))
    with skip_mkdir(TRAIN_IMPORT_DIR):
        train = importlib.import_module("train")
    signatures = importlib.import_module("consolida_sim")
    models = importlib.import_module("ttd_scs_lstm.models.models")
    for module, folder in ((train, SCRIPTS), (signatures, SCRIPTS), (models, SRC)):
        origin = Path(module.__file__ or "").resolve()
        if not origin.is_relative_to(folder.resolve()):
            raise RuntimeError(f"{module.__name__} was imported from {origin}, not from {folder}")
    return train, signatures, models.create_model


def pick_device(requested: str | None) -> str:
    """The requested torch device; by default CUDA when available, else the CPU."""
    if requested:
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------- panel


def evaluate_runs(
    train: ModuleType,
    signatures: ModuleType,
    create_model: Callable[..., torch.nn.Module],
    grade: Path,
    h5: Path,
    device: str,
) -> tuple[list[dict[str, Any]], int]:
    """Validation metrics of every run of CONFIGS, and the number of validation windows."""
    _, val_loader, _, static = train.create_dataloaders(
        h5, lookback=LOOKBACK, horizon=HORIZON, batch_size=BATCH_SIZE
    )
    n_windows = len(val_loader.dataset)
    print(f"Validation windows: {n_windows} | device {device} | torch {torch.__version__}")
    print(f"Grid: {display_path(grade)} | data: {display_path(h5)}", flush=True)

    rows: list[dict[str, Any]] = []
    for rung, config, model_type in CONFIGS:
        for seed in SEEDS:
            run_dir = grade / f"seed{seed}" / config
            found = sorted(run_dir.glob("**/best_model.pt"))
            if not found:
                print(f"  [missing] {config} seed {seed}: no best_model.pt", flush=True)
                continue
            if len(found) > 1:
                raise RuntimeError(f"more than one best_model.pt under {run_dir}")
            ckpt = found[0]
            results = json.loads((ckpt.parent / "results.json").read_text(encoding="utf-8"))
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
            start = time.perf_counter()
            train.set_seed(seed)
            model = create_model(
                model_type=model_type,
                cn_values=static["cn_values"],
                tc_base_values=static["tc_base_values"],
                tc_manning_values=static["tc_manning_values"],
                area_km2=static["area_km2"],
                twi_dist=static.get("twi_dist"),
                twi_centers=static.get("twi_centers"),
                twi_mean=static.get("twi_mean"),
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                horizon=cfg["horizon"],
                device=device,
            )
            model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
            metrics, _, preds, targets = train.evaluate(model, val_loader, device)
            sig = signatures.assinaturas(preds, targets)
            nse = float(metrics["nse"])
            best_val = float(results.get("best_val_nse", float("nan")))
            diff = abs(nse - best_val)
            flag = "" if diff <= SANITY_TOLERANCE else "  [DIFFERS from best_val_nse]"
            rows.append(
                {
                    "rung": rung,
                    "config": config,
                    "seed": seed,
                    "checkpoint": display_path(ckpt),
                    "nse": nse,
                    "kge": float(metrics["kge"]),
                    "lognse": sig.get("lognse"),
                    "pbias": sig.get("pbias"),
                    "fhv": sig.get("fhv"),
                    "flv": sig.get("flv"),
                    "best_val_nse": best_val,
                    "abs_diff_best_val_nse": diff,
                }
            )
            print(
                f"  {config:<26} seed {seed}: NSE {nse:.6f} (best_val_nse {best_val:.6f},"
                f" |diff| {diff:.1e})  KGE {float(metrics['kge']):.6f}"
                f"  FLV {sig.get('flv'):+.4f}  FHV {sig.get('fhv'):+.4f}"
                f"  ({time.perf_counter() - start:.0f} s){flag}",
                flush=True,
            )
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return rows, n_windows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean and population standard deviation (ddof = 0) across seeds, per configuration."""
    agg: dict[str, dict[str, Any]] = {}
    for rung, config, _ in CONFIGS:
        runs = [row for row in rows if row["config"] == config]
        if not runs:
            continue
        entry: dict[str, Any] = {
            "rung": rung,
            "n_seeds": len(runs),
            "seeds": [row["seed"] for row in runs],
        }
        for key in (*METRICS, "best_val_nse"):
            values = np.array([row[key] for row in runs], dtype=float)
            entry[key] = {"mean": float(values.mean()), "sd_pop": float(values.std(ddof=0))}
        agg[config] = entry
    return agg


def seeds_ahead(rows: list[dict[str, Any]], key: str, first: str, second: str) -> tuple[int, int]:
    """Seeds in which ``first`` scores higher than ``second`` on ``key``, and seeds compared."""
    a = {row["seed"]: row[key] for row in rows if row["config"] == first}
    b = {row["seed"]: row[key] for row in rows if row["config"] == second}
    common = sorted(set(a) & set(b))
    return sum(a[seed] > b[seed] for seed in common), len(common)


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


def rounded_text(value: float, decimals: int, signed: bool) -> str:
    """Value as printed in Table S4 (explicit sign on the FLV and FHV means)."""
    if not math.isfinite(value):
        return "NA"
    rounded = round_half_up(value, decimals)
    return f"{rounded:+}" if signed else f"{rounded}"


def display_path(path: Path) -> str:
    """Path relative to the repository root when possible."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def table_s4_cells(agg: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The 24 numbers of Table S4: unrounded, rounded, manuscript value and rounding margin."""
    cells = []
    for rung, config, _ in CONFIGS:
        if config not in agg:
            continue
        printed = iter(MANUSCRIPT_TABLE_S4[config])
        for key, label, decimals, signed in COLUMNS:
            for stat, sign in (("mean", signed), ("sd_pop", False)):
                value = agg[config][key][stat]
                manuscript = next(printed)
                text = rounded_text(value, decimals, sign)
                cells.append(
                    {
                        "rung": rung,
                        "config": config,
                        "key": key,
                        "label": f"{label.split()[0]} {'mean' if stat == 'mean' else 'sd'}",
                        "stat": stat,
                        "value": value,
                        "two_decimals": rounded_text(value, 2, sign),
                        "text": text,
                        "manuscript": manuscript,
                        "ok": math.isfinite(value)
                        and round_half_up(value, decimals) == Decimal(manuscript),
                        "margin": rounding_margin(value, decimals),
                        "decimals": decimals,
                    }
                )
    return cells


# ---------------------------------------------------------------------------- report


def print_table_s4(agg: dict[str, dict[str, Any]]) -> None:
    """Table S4 rounded as in the Supplementary Information, then the unrounded values."""
    print()
    print("Table S4. Metric panel over the validation period")
    print("Mean +/- population standard deviation (ddof = 0) across seeds; FLV and FHV in percent.")
    print()
    head = "".join(f"  {label:<15}" for _, label, _, _ in COLUMNS)
    print(f"{'#':>2}  {'Configuration':<26} {'n':>2}{head}")
    for rung, config, _ in CONFIGS:
        if config not in agg:
            continue
        entry = agg[config]
        texts = []
        for key, _, decimals, signed in COLUMNS:
            mean = rounded_text(entry[key]["mean"], decimals, signed)
            sd = rounded_text(entry[key]["sd_pop"], decimals, False)
            texts.append(f"  {mean + ' +/- ' + sd:<15}")
        print(f"{rung:>2}  {config:<26} {entry['n_seeds']:>2}{''.join(texts)}")

    print()
    print("Unrounded values (mean and population sd across seeds; logNSE and PBIAS are not in")
    print("Table S4)")
    print(f"{'#':>2}  {'Configuration':<26} {'Metric':<7} {'Mean':>11} {'SD':>10}")
    for rung, config, _ in CONFIGS:
        for key in METRICS if config in agg else ():
            mean, sd = agg[config][key]["mean"], agg[config][key]["sd_pop"]
            print(f"{rung:>2}  {config:<26} {key:<7} {mean:>+11.6f} {sd:>10.6f}")


def print_checks(
    rows: list[dict[str, Any]],
    agg: dict[str, dict[str, Any]],
    cells: list[dict[str, Any]],
    n_windows: int,
    record: Path,
) -> tuple[int, int]:
    """Checks against training, Table S4 and the recorded panel; returns (mismatches, flagged)."""
    print()
    print("Checks")
    status = "matches" if n_windows == MANUSCRIPT_VALIDATION_WINDOWS else "DIFFERS from"
    print(
        f"  Validation windows: {n_windows} ({status} the {MANUSCRIPT_VALIDATION_WINDOWS:,}"
        " of the manuscript, Section 2.5)"
    )
    flagged = sum(1 for row in rows if not row["abs_diff_best_val_nse"] <= SANITY_TOLERANCE)
    largest = max((row["abs_diff_best_val_nse"] for row in rows), default=float("nan"))
    print(
        f"  Sanity: {len(rows) - flagged} of {len(rows)} runs agree with best_val_nse of"
        f" results.json within {SANITY_TOLERANCE} (max |difference| {largest:.1e})"
    )
    print("  best_val_nse of results.json (mean and population sd across seeds):")
    for config, entry in agg.items():
        mean, sd = entry["best_val_nse"]["mean"], entry["best_val_nse"]["sd_pop"]
        print(
            f"    {config:<26} {mean:.6f} -> {round_half_up(mean, 2)}"
            f" | sd {sd:.6f} -> {round_half_up(sd, 2)}"
        )
    for key, source in (("nse", "recomputed NSE"), ("best_val_nse", "best_val_nse")):
        ahead, n = seeds_ahead(rows, key, "LSTM", "LSTM_DUH_Topmodel_PeOnly")
        print(f"  LSTM ahead of LSTM_DUH_Topmodel_PeOnly in {ahead} of {n} seeds ({source})")

    mismatches = [cell for cell in cells if not cell["ok"]]
    print(
        f"  Table S4: {len(cells) - len(mismatches)} of {len(cells)} numbers match the manuscript"
    )
    print()
    print(
        f"{'#':>2}  {'Configuration':<26} {'Number':<9} {'Unrounded':>11} {'2 dp':>6}"
        f" {'Table S4':>8} {'Manuscript':>10}  {'Status':<7} {'Margin':>8}"
    )
    for cell in cells:
        print(
            f"{cell['rung']:>2}  {cell['config']:<26} {cell['label']:<9} {cell['value']:>+11.6f}"
            f" {cell['two_decimals']:>6} {cell['text']:>8} {cell['manuscript']:>10}"
            f"  {'ok' if cell['ok'] else 'DIFFERS':<7} {cell['margin']:>8.6f}"
        )
    print("  Margin: distance from the unrounded value to the nearest rounding boundary at the")
    print("  precision of Table S4 (0.005 is the largest possible margin at two decimals).")
    near = sorted(
        (cell for cell in cells if cell["margin"] <= 0.1 * 10 ** -cell["decimals"]),
        key=lambda cell: cell["margin"] * 10 ** cell["decimals"],
    )
    print("  Numbers within a tenth of the last printed digit of a rounding boundary:")
    for cell in near:
        print(
            f"    rung {cell['rung']} {cell['config']} {cell['label']}: {cell['value']:+.6f}"
            f" -> {cell['text']} (margin {cell['margin']:.6f})"
        )
    if not near:
        print("    none")

    if record.exists():
        recorded = json.loads(record.read_text(encoding="utf-8"))
        by_run = {(row["config"], row["seed"]): row for row in recorded.get("rows", [])}
        print()
        print(
            f"  Recorded panel ({display_path(record)}; per-run values rounded to three decimals"
            " for NSE and KGE and one for FLV and FHV):"
        )
        for key, label, _, _ in COLUMNS:
            diffs = [
                abs(row[key] - by_run[(row["config"], row["seed"])][key])
                for row in rows
                if key in by_run.get((row["config"], row["seed"]), {})
            ]
            if diffs:
                print(
                    f"    {label:<8} max |recomputed - recorded| over {len(diffs)} runs:"
                    f" {max(diffs):.6f}"
                )
        recorded_agg = recorded.get("agg", {})
        print("    Aggregates of the record (from its rounded per-run values) and recomputed:")
        for cell in cells:
            stat = "mean" if cell["stat"] == "mean" else "std"
            old = recorded_agg.get(cell["config"], {}).get(cell["key"], {}).get(stat)
            if old is None:
                continue
            print(
                f"    rung {cell['rung']:>2} {cell['config']:<26} {cell['label']:<9}"
                f" record {old:>+11.6f} | recomputed {cell['value']:>+11.6f}"
                f" | difference {cell['value'] - old:>+9.6f}"
            )
    else:
        print(f"  Recorded panel not found: {display_path(record)}")
    return len(mismatches), flagged


# ---------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--device",
        default=None,
        help="torch device, for example cpu, cuda or cuda:1 (default: cuda:0 if available, else"
        " cpu); with cpu the GPUs are hidden from torch",
    )
    parser.add_argument(
        "--grade",
        type=Path,
        default=GRADE,
        help="grid folder with seedNN/<configuration>/<class>/ (default: outputs/grade_sim)",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=H5,
        help="dataset with the validation split (default: data/dataset_58585000_telem.h5)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="JSON file for the unrounded values (default: outputs/tables/validation_panel.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 if any number differs from Table S4 or any run fails the sanity"
        " check",
    )
    args = parser.parse_args()
    if args.out.resolve() in {RECORD.resolve(), RECORD_CSV.resolve()}:
        parser.error(f"--out must not overwrite the recorded panel {display_path(args.out)}")
    if args.device == "cpu":
        # Hide the GPUs from torch before CUDA starts ("-1": an empty value is dropped on Windows).
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    train, signatures, create_model = import_project_code()
    device = pick_device(args.device)
    # train.py builds the data tensors of its loaders on its module-level DEVICE (CUDA whenever
    # available); keep them on the device chosen here.
    train.DEVICE = torch.device(device)

    rows, n_windows = evaluate_runs(train, signatures, create_model, args.grade, args.h5, device)
    if not rows:
        parser.error(f"no checkpoints found in {args.grade}")
    agg = aggregate(rows)
    cells = table_s4_cells(agg)
    print_table_s4(agg)
    mismatches, flagged = print_checks(rows, agg, cells, n_windows, RECORD)

    ahead, n_common = seeds_ahead(rows, "nse", "LSTM", "LSTM_DUH_Topmodel_PeOnly")
    ahead_bv, _ = seeds_ahead(rows, "best_val_nse", "LSTM", "LSTM_DUH_Topmodel_PeOnly")
    table_s4: dict[str, dict[str, Any]] = {}
    for cell in cells:
        column = table_s4.setdefault(cell["config"], {}).setdefault(cell["key"], {})
        column[cell["stat"]] = {"text": cell["text"], "manuscript": cell["manuscript"]}
    payload = {
        "description": "Table S4 recomputed: metrics over the validation period of rungs 4, 9 and"
        " 10 (10 seeds each) from the shipped checkpoints; unrounded values. Mean and population"
        " standard deviation (ddof = 0) across seeds.",
        "script": "scripts/analysis/validation_panel.py",
        "data": display_path(args.h5),
        "data_sha256": sha256(args.h5),
        "grade": display_path(args.grade),
        "validation_windows": n_windows,
        "lookback_h": LOOKBACK,
        "horizon_h": HORIZON,
        "batch_size": BATCH_SIZE,
        "device": device,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "sanity": {
            "tolerance": SANITY_TOLERANCE,
            "runs": len(rows),
            "flagged": flagged,
            "max_abs_diff_best_val_nse": max(row["abs_diff_best_val_nse"] for row in rows),
        },
        "lstm_ahead_of_peonly": {
            "seeds_compared": n_common,
            "recomputed_nse": ahead,
            "best_val_nse": ahead_bv,
        },
        "rows": rows,
        "agg": agg,
        "table_s4": table_s4,
        "table_s4_mismatches": mismatches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {display_path(args.out)}")
    return 1 if args.strict and (mismatches or flagged) else 0


if __name__ == "__main__":
    raise SystemExit(main())
