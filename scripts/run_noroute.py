#!/usr/bin/env python
"""Runner of LSTM_Topmodel and LSTM_SCS (runoff generation without DUH routing, impulse unit hydrograph;
manuscript Section 4.4), for continuous simulation.

Each configuration is trained with horizon 1 (the next hour from a 240 h lookback) and the
window slides over the record, so the NSE at 1 h over the test windows is the
continuous-simulation NSE: streamflow is generated from rainfall alone and observed streamflow
is never an input. Calls scripts/train.py with --horizon 1. A run gets a .done marker only
when its results.json holds a finite test NSE, and runs that already have one are skipped, so
the shipped results in outputs/grade_sim are never retrained in place.

Usage, from the repository root:
  smoke test (1 epoch, writes to outputs/smoke):
      python scripts/run_noroute.py --test --seeds 42
  protocol of the manuscript (150 epochs, patience 25, batch 512), into a new folder:
      python scripts/run_noroute.py --seeds 42 43 44 45 46 47 48 49 50 51 --out outputs/retrain
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "scripts" / "train.py"
DATA = ROOT / "data" / "dataset_58585000_telem.h5"
OUT = ROOT / "outputs" / "grade_sim"   # shipped results (runs with a .done marker are skipped)
SMOKE = ROOT / "outputs" / "smoke"     # default output of --test

# (internal model_type, display name) of the configurations trained by this runner
MODELS = [
    ("lstm_topmodel", "LSTM_Topmodel"),
    ("lstm_scs", "LSTM_SCS"),
]


def valido(run_dir):
    cands = list(run_dir.glob("**/results.json"))
    if not cands:
        return False, "sem results.json"
    try:
        r = json.loads(cands[0].read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"ilegivel ({e})"
    v = r.get("test_metrics", {}).get("nse", None)   # main metric = @1h (horizon=1) = simulation NSE
    if v is None or not (v == v) or abs(v) == float("inf"):
        return False, f"test NSE nao-finito ({v})"
    return True, f"simNSE={v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--test", action="store_true",
                    help="smoke test: 1 epoch, written to outputs/smoke unless --out is given")
    ap.add_argument("--out", default=None,
                    help="output folder, relative to the repository root "
                         "(default outputs/grade_sim, or outputs/smoke with --test)")
    a = ap.parse_args()
    if a.test:
        a.epochs, a.patience = 1, 1
    out = Path(a.out) if a.out else (SMOKE if a.test else OUT)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    summary, tg = [], time.time()
    print("=" * 70)
    print(f"  SIMULATION GRID (window, horizon=1) | seeds={a.seeds} | epochs={a.epochs} "
          f"pat={a.patience}{' [TEST]' if a.test else ''}")
    print("=" * 70)
    for seed in a.seeds:
        for mt, disp in MODELS:
            run_dir = out / f"seed{seed}" / disp
            done = run_dir / ".done"
            if done.exists() and not a.test:
                print(f"  [SKIP] seed{seed}/{disp}")
                summary.append((seed, disp, "skip", 0))
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(TRAIN), "--model", mt, "--data-path", str(DATA),
                   "--seed", str(seed), "--output-dir", str(run_dir),
                   "--batch-size", str(a.batch_size), "--epochs", str(a.epochs),
                   "--patience", str(a.patience), "--horizon", "1"]
            if a.test:
                cmd.append("--test")
            print(f"\n>>> seed{seed} | {disp} ({mt}) | {datetime.now():%H:%M:%S}")
            t0 = time.time()
            r = subprocess.run(cmd, cwd=str(ROOT))
            dt = time.time() - t0
            ok_out, msg = valido(run_dir)
            ok = r.returncode == 0 and ok_out
            if ok and not a.test:
                done.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {dt:.0f}s | {msg}\n")
            print(f"<<< seed{seed} | {disp} | {'ok' if ok else 'FAIL'} | {timedelta(seconds=int(dt))}"
                  f"{'' if ok else '  [rc=' + str(r.returncode) + ' | ' + msg + ']'}")
            summary.append((seed, disp, "ok" if ok else "fail", round(dt)))
    print(f"\n  FIM | total {timedelta(seconds=int(time.time()-tg))}")
    n_ok = sum(1 for _, _, st, _ in summary if st in ("ok", "skip"))
    for seed, disp, st, dt in summary:
        print(f"  seed{seed:>3} {disp:<30} {st:>4} {timedelta(seconds=dt)}")
    print(f"  {n_ok}/{len(summary)} OK")
    return 0 if n_ok == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
