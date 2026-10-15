#!/usr/bin/env python
"""Supplementary Tables S5 and S6 (lumped ABC sweeps) and the cross-GPU check of Section S8.

Reads the logs that ``scripts/abc/lumped_lstm.py`` wrote in ``outputs/abc`` (horizon 1 h,
lookback 240 h, seeds 0 to 4):

- ``backbones_telem_gpu1.log``: architecture sweep with the telemetric forcing (Table S5);
- ``products_lstm_gpu1.log``: rainfall-product sweep with the LSTM (Table S6);
- ``products_lstm_gpu0.log``: the product sweep repeated on the other GPU, used only for the
  comparison of Section S8 (shared configuration: LSTM, telemetric forcing, coupled mode).

``gpu1`` is the RTX 2000 Ada, on which both tables were run; ``gpu0`` is the RTX 3000 Ada.
Modes: A, aggregated rainfall only; B, rainfall plus calendar (coupled); C, calendar only.

Each cell is computed from the per-seed test NSE of the ``PERSEED`` lines of the log: mean and
population standard deviation (ddof = 0, as in ``lumped_lstm.py``), printed unrounded and rounded
to two decimals from the unrounded value. The three-decimal summary that the log prints for each
mode is not rounded a second time: that would give 0.18 +/- 0.12 instead of 0.18 +/- 0.11 for the
Transformer in mode C (unrounded 0.18454 +/- 0.11479). Because the per-seed values carry four
decimals, a mean or a standard deviation can differ from the one computed at full precision by
at most 5e-5: cells closer than that to a two-decimal rounding boundary are flagged, and every
cell is checked against the three-decimal summary of the log. The script also prints the
differences between cells quoted in Section 4.1 and Supplementary Section S5.

``parse`` (the three-decimal summaries) and the constants ``LOGS_ARCH``, ``LOG_PROD``,
``BB_LABEL`` and ``PROD_LABEL`` are imported by ``fig_abc.py``, whose bars use those summaries;
they are kept as in the origin so that Figure S1 does not change.

Output: the tables are printed and written as Markdown to ``outputs/tables/tabela_abc.md``.

Origin: ``tabela_abc.py`` of the authors' working folder (23 June 2026; SHA-256
c067991d91883255360ed5ecd0ae70455e16b01ed44749480bbc98ecbc75ee84), which printed the
three-decimal summaries of the logs. Changes for this repository: paths relative to the
repository; tables from the per-seed values (new functions from ``parse_perseed`` on); the
Section S8 comparison and the differences quoted in the text; argparse; type hints; English text.

Usage, from the repository root:
    python scripts/figures/tabela_abc.py [--out FILE]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ABC = ROOT / "outputs" / "abc"

# Aligned rerun of 23 June 2026 (AdamW, scheduler, 5 seeds, lumped): the five architectures in a
# single log, on the RTX 2000 Ada (gpu1).
LOGS_ARCH = [
    ABC / "backbones_telem_gpu1.log",    # lstm, gru, transformer, dlinear, mlp
]
LOG_PROD = ABC / "products_lstm_gpu1.log"  # RTX 2000 Ada: Table S6
LOG_PROD_RTX3000 = ABC / "products_lstm_gpu0.log"  # RTX 3000 Ada: only for Section S8
OUT = ROOT / "outputs" / "tables" / "tabela_abc.md"

BB_LABEL = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "transformer": "Transformer",
            "dlinear": "DLinear", "mamba": "Mamba", "mlp": "MLP"}
PROD_LABEL = {"telem": "Telemetric (gauges)", "merge": "MERGE", "era5land": "ERA5-Land",
              "imerg": "IMERG", "mswep": "MSWEP"}


def parse(path: Path) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """(product, backbone) -> {mode: (mean, sd)} from the three-decimal summary of each mode line.

    Used by ``fig_abc.py`` for the bars of Figure S1. The tables use ``parse_perseed``.
    """
    data, cur = {}, None
    if not path.exists():
        return data
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        h = re.search(r"produto=(\S+)\s+backbone=(\S+)", ln)
        if h:
            cur = (h.group(1), h.group(2))
            data.setdefault(cur, {})
            continue
        m = re.match(r"\s+([ABC]):.*?@1h=(-?[\d.]+)\D+([\d.]+)", ln)
        if m and cur:
            data[cur][m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return data


# --------------------------------------------------------------------------- per-seed tables
MODES = ("A", "B", "C")  # column order of Tables S5 and S6
MODE_HEADER = {"A": "A: rainfall only", "B": "B: coupled", "C": "C: calendar only"}
PERSEED_MODE = {"chuva": "A", "ambos": "B", "tempo": "C"}  # labels written by lumped_lstm.py
TABLE_PROD_LABEL = {**PROD_LABEL, "telem": "Telemetric"}  # row labels of Table S6
PERSEED_STEP = 5e-5  # half of the last decimal of the per-seed values (four decimals)
LOG_STEP = 5e-4  # half of the last decimal of the log summaries (three decimals)

Key = tuple[str, str]  # (product, backbone)


@dataclass
class Cell:
    """One cell of Table S5 or S6, from the per-seed values of a log."""

    seeds: dict[int, float]  # seed -> test NSE at 1 h, four decimals as logged
    log_mean: float | None  # three-decimal summary of the log, for the consistency check
    log_sd: float | None

    @property
    def mean(self) -> float:
        """Unrounded mean over the seeds."""
        return float(np.mean(list(self.seeds.values())))

    @property
    def sd(self) -> float:
        """Unrounded population standard deviation over the seeds (ddof = 0)."""
        return float(np.std(list(self.seeds.values()), ddof=0))

    @property
    def near_boundary(self) -> bool:
        """True if the mean or the sd may round differently at full precision."""
        return any(_boundary_distance(v) < PERSEED_STEP for v in (self.mean, self.sd))

    @property
    def matches_log(self) -> bool:
        """True if mean and sd agree with the three-decimal summary of the log."""
        if self.log_mean is None or self.log_sd is None:
            return False
        tol = LOG_STEP + PERSEED_STEP + 1e-9
        return abs(self.mean - self.log_mean) <= tol and abs(self.sd - self.log_sd) <= tol


def _boundary_distance(x: float) -> float:
    """Distance from ``x`` to the nearest two-decimal rounding boundary (k + 0.5) / 100."""
    scaled = abs(x) * 100.0
    return abs(scaled - math.floor(scaled) - 0.5) / 100.0


def _num(x: float) -> str:
    """Unrounded value followed by its two-decimal rounding."""
    return f"{x:+.6f} ({x:+.2f})"


def parse_perseed(path: Path) -> dict[Key, dict[str, dict[int, float]]]:
    """(product, backbone) -> {mode: {seed: test NSE at 1 h}} from the PERSEED lines of a log.

    The mode of a PERSEED line follows from its label (chuva = A, ambos = B, tempo = C in
    ``lumped_lstm.py``) and must agree with the mode of the summary line just above it.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    data: dict[Key, dict[str, dict[int, float]]] = {}
    cur: Key | None = None
    mode_above: str | None = None
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        h = re.search(r"produto=(\S+)\s+backbone=(\S+)", ln)
        if h:
            cur, mode_above = (h.group(1), h.group(2)), None
            continue
        m = re.match(r"\s+([ABC]):", ln)
        if m:
            mode_above = m.group(1)
            continue
        p = re.match(r"\s+PERSEED\s+(\w+)\s+@(\d+)h:\s*(\S+)\s*$", ln)
        if p is None or cur is None:
            continue
        mode = PERSEED_MODE[p.group(1)]
        if mode != mode_above or p.group(2) != "1":
            raise ValueError(f"{path.name}: unexpected PERSEED line for {cur}: {ln.strip()}")
        seeds = {}
        for item in p.group(3).split(","):
            seed, value = item.split("=")
            seeds[int(seed)] = float(value)
        data.setdefault(cur, {})[mode] = seeds
    return data


def build_cells(path: Path) -> dict[Key, dict[str, Cell]]:
    """Cells of every (product, backbone) and mode of a log, with the log summaries attached."""
    summary = parse(path)
    cells: dict[Key, dict[str, Cell]] = {}
    for key, modes in parse_perseed(path).items():
        for mode, seeds in modes.items():
            mu_sd = summary.get(key, {}).get(mode)
            cells.setdefault(key, {})[mode] = Cell(seeds, *(mu_sd if mu_sd else (None, None)))
    return cells


def table_rows(cells: dict[Key, dict[str, Cell]], fix_idx: int, fix_val: str,
               var_idx: int) -> list[tuple[str, dict[str, Cell]]]:
    """Rows (varying item, cells) with key[fix_idx] == fix_val, by decreasing unrounded B mean."""
    rows = [(key[var_idx], c) for key, c in cells.items() if key[fix_idx] == fix_val and "B" in c]
    return sorted(rows, key=lambda r: -r[1]["B"].mean)


def _cell_text(cell: Cell, mode: str) -> str:
    """Two-decimal cell as in the manuscript: signed mean in modes A and C, plain in mode B."""
    return f"{cell.mean:{'.2f' if mode == 'B' else '+.2f'}} ± {cell.sd:.2f}"


def markdown_table(title: str, first_col: str, rows: list[tuple[str, dict[str, Cell]]],
                   labels: dict[str, str]) -> str:
    """Markdown version of Table S5 or S6 (mode B in bold, as in the manuscript)."""
    lines = [f"### {title}", "",
             f"| {first_col} | " + " | ".join(MODE_HEADER[m] for m in MODES) + " |",
             "|---|" + ":---:|" * len(MODES)]
    for item, cells in rows:
        texts = [_cell_text(cells[m], m) if m in cells else "n/a" for m in MODES]
        texts = [f"**{t}**" if m == "B" else t for m, t in zip(MODES, texts)]
        lines.append(f"| {labels.get(item, item)} | " + " | ".join(texts) + " |")
    return "\n".join(lines)


def detail_lines(rows: list[tuple[str, dict[str, Cell]]], labels: dict[str, str]) -> list[str]:
    """Unrounded mean and population sd of every cell, with the checks against the log."""
    out = []
    for item, cells in rows:
        for m in MODES:
            if m not in cells:
                continue
            c = cells[m]
            flags = ("" if c.matches_log else "  DIFFERS FROM LOG SUMMARY") + (
                "  NEAR ROUNDING BOUNDARY" if c.near_boundary else "")
            log = f"{c.log_mean:+.3f} ± {c.log_sd:.3f}" if c.log_mean is not None else "n/a"
            out.append(f"  {labels.get(item, item):<12} {m}  n={len(c.seeds)}  mean {_num(c.mean)}"
                       f"  sd {c.sd:.6f} ({c.sd:.2f})  -> {_cell_text(c, m):<14} log {log}{flags}")
    return out


def text_numbers(arch: list[tuple[str, dict[str, Cell]]],
                 prod: list[tuple[str, dict[str, Cell]]]) -> list[str]:
    """Differences between cells quoted in Section 4.1 and Supplementary Section S5."""
    b = {item: c["B"].mean for item, c in arch}
    best, worst = max(b, key=b.get), min(b, key=b.get)
    out = ["Differences quoted in Section 4.1 and Section S5 (from the unrounded means):",
           f"  coupled mode, best minus worst architecture ({BB_LABEL[best]} minus "
           f"{BB_LABEL[worst]}): {_num(b[best] - b[worst])}"]
    for item, c in sorted(arch, key=lambda r: r[1]["B"].mean - r[1]["A"].mean):
        gain = c["B"].mean - c["A"].mean
        out.append(f"  coupling gain B minus A, {BB_LABEL[item]:<11}: {_num(gain)}")
    out.append(f"  DLinear B over LSTM B: {_num(b['dlinear'] / b['lstm'])}")
    pb = {item: c["B"].mean for item, c in prod}
    for item, _ in prod:
        if item != "telem":
            drop = pb["telem"] - pb[item]
            out.append(f"  mode B, telemetric minus {TABLE_PROD_LABEL[item]:<9}: {_num(drop)}")
    return out


def card_lines(log_rtx2000: Path, log_rtx3000: Path) -> list[str]:
    """Section S8: the shared configuration (LSTM, telemetric, mode B) on the two GPUs."""
    cards = {"RTX 3000 Ada": log_rtx3000, "RTX 2000 Ada": log_rtx2000}
    cell = {name: build_cells(path)[("telem", "lstm")]["B"] for name, path in cards.items()}
    out = ["Section S8: LSTM, telemetric forcing, mode B, seeds 0 to 4, on the two GPUs:"]
    for name, path in cards.items():
        c = cell[name]
        vals = list(c.seeds.values())
        out.append(f"  {name} ({path.name}): mean {_num(c.mean)}, population sd {c.sd:.6f} "
                   f"({c.sd:.2f}), range {min(vals):.4f} to {max(vals):.4f}; "
                   f"log summary {c.log_mean:.3f} ± {c.log_sd:.3f}")
    diff = cell["RTX 2000 Ada"].mean - cell["RTX 3000 Ada"].mean
    out.append(f"  difference, RTX 2000 Ada minus RTX 3000 Ada: {_num(diff)}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Supplementary Tables S5 and S6 and the Section S8 check, from the ABC logs.")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="Markdown file (default: outputs/tables/tabela_abc.md)")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass

    arch_cells: dict[Key, dict[str, Cell]] = {}
    for log in LOGS_ARCH:
        arch_cells.update(build_cells(log))
    prod_cells = build_cells(LOG_PROD)
    arch = table_rows(arch_cells, fix_idx=0, fix_val="telem", var_idx=1)
    prod = table_rows(prod_cells, fix_idx=1, fix_val="lstm", var_idx=0)

    t5 = markdown_table("Table S5. Architecture sweep (telemetric forcing)", "Architecture",
                        arch, BB_LABEL)
    t6 = markdown_table("Table S6. Rainfall-product sweep (LSTM)", "Product", prod,
                        TABLE_PROD_LABEL)
    note = ("Simulation NSE at 1 h over the test period, mean ± population standard deviation "
            "across 5 seeds, from the per-seed values of the logs, rounded to two decimals.")
    md = "\n\n".join(["# Lumped ABC sweeps: Tables S5 and S6", note, t5, t6]) + "\n"

    report = [md,
              f"Table S5 cells ({', '.join(p.name for p in LOGS_ARCH)}):",
              *detail_lines(arch, BB_LABEL), "",
              f"Table S6 cells ({LOG_PROD.name}):",
              *detail_lines(prod, TABLE_PROD_LABEL), "",
              *text_numbers(arch, prod), "",
              *card_lines(LOG_PROD, LOG_PROD_RTX3000)]
    print("\n".join(report))
    n_cells = sum(len(c) for _, c in arch) + sum(len(c) for _, c in prod)
    flagged = [c for _, cs in arch + prod for c in cs.values()
               if c.near_boundary or not c.matches_log]
    print(f"\n{n_cells} cells ({2 * n_cells} numbers); {len(flagged)} flagged "
          "(near a rounding boundary or differing from the log summary).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    try:
        shown = args.out.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"[OK] {shown}")


if __name__ == "__main__":
    main()
