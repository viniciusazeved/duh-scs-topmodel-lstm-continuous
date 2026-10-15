"""C3 (revisao pre-submissao Paper 1): painel de metricas no periodo de VALIDACAO.

Sem retreinar: carrega os best_model.pt existentes da grade_sim (10 seeds) para as
configs 4 (LSTM, neural puro), 9 (LSTM_DUH_Topmodel) e 10 (LSTM_DUH_Topmodel_PeOnly),
roda inferencia no val_loader canonico e computa as MESMAS metricas do teste
(NSE, KGE via train.compute_metrics; logNSE/PBIAS/FHV/FLV via consolida_sim.assinaturas,
a implementacao exata usada na Tabela 2 do paper).

Sanidade embutida: o NSE de validacao recomputado deve bater com o best_val_nse do
results.json de cada run (mesmo checkpoint, mesma particao) — divergencia > 0.01 e flag.

Codigo original intocado (so leitura). Saida: c3_painel_validacao.{json,csv} nesta pasta.
Uso (gpuq, cuda:0): uv run --project D:/TTD_SCS_LSTM python D:/TTD_SCS_LSTM/revisao2_paper1/c3_painel_validacao.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS = r"D:/TTD_SCS_LSTM/ablacao_skill/scripts"
SRC = r"D:/TTD_SCS_LSTM/ablacao_skill/src"
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, SRC)

import train as T  # train.py CANONICO do ablacao_skill  # noqa: E402
import consolida_sim as CS  # assinaturas/_fdc do paper  # noqa: E402
from ttd_scs_lstm.models.models import create_model  # noqa: E402

H5 = r"D:/TTD_SCS_LSTM/ablacao_skill/data/dataset_58585000_telem.h5"
GRADE = Path(r"D:/TTD_SCS_LSTM/ablacao_skill/outputs/grade_sim")
OUT = Path(r"D:/TTD_SCS_LSTM/revisao2_paper1")

# (numero na Tabela 2, nome da pasta, model_type do factory)
CONFIGS = [
    (4, "LSTM", "lstm"),
    (9, "LSTM_DUH_Topmodel", "lstm_duh_base_topmodel"),
    (10, "LSTM_DUH_Topmodel_PeOnly", "lstm_duh_base_topmodel_peonly"),
]
SEEDS = list(range(42, 52))


def main() -> int:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _, val_loader, _, static = T.create_dataloaders(H5, lookback=240, horizon=1, batch_size=512)
    print(f"val: {len(val_loader.dataset)} janelas | device {device}", flush=True)

    rows = []
    for num, disp, mt in CONFIGS:
        for seed in SEEDS:
            run_dir = GRADE / f"seed{seed}" / disp
            cands = list(run_dir.glob("**/best_model.pt"))
            if not cands:
                print(f"  [FALTA] {disp} seed{seed}: sem best_model.pt", flush=True)
                continue
            ckpt = cands[0]
            rj = json.loads((ckpt.parent / "results.json").read_text(encoding="utf-8"))
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
            T.set_seed(seed)
            model = create_model(
                model_type=mt,
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
            ).to(device)
            model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
            met, _, preds, targets = T.evaluate(model, val_loader, device)
            ass = CS.assinaturas(preds, targets)
            best_val = rj.get("best_val_nse", float("nan"))
            delta = abs(met["nse"] - best_val)
            flag = "" if delta <= 0.01 else f"  [DIVERGE do best_val_nse {best_val:.4f} por {delta:.4f}]"
            rows.append({
                "n": num, "config": disp, "seed": seed,
                "nse": met["nse"], "kge": met["kge"],
                "lognse": ass.get("lognse"), "pbias": ass.get("pbias"),
                "fhv": ass.get("fhv"), "flv": ass.get("flv"),
                "best_val_nse_treino": best_val, "delta_sanidade": delta,
            })
            print(f"  {disp:28s} seed{seed}: NSE {met['nse']:.3f} KGE {met['kge']:.3f} "
                  f"FLV {ass.get('flv'):+.1f} FHV {ass.get('fhv'):+.1f}{flag}", flush=True)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # agregacao media +- dp entre seeds
    print("\n=== painel de VALIDACAO (media +- dp, 10 seeds) ===", flush=True)
    agg = {}
    for num, disp, _ in CONFIGS:
        rs = [r for r in rows if r["config"] == disp]
        if not rs:
            continue
        agg[disp] = {"n_tabela": num, "n_seeds": len(rs)}
        for k in ("nse", "kge", "lognse", "pbias", "fhv", "flv"):
            v = np.array([r[k] for r in rs], dtype=float)
            agg[disp][k] = {"mean": float(v.mean()), "std": float(v.std())}
        a = agg[disp]
        print(f"  {num:>2} {disp:28s} ({len(rs)} seeds): "
              f"NSE {a['nse']['mean']:.3f}+-{a['nse']['std']:.3f} | "
              f"KGE {a['kge']['mean']:.3f}+-{a['kge']['std']:.3f} | "
              f"FLV {a['flv']['mean']:+.1f}+-{a['flv']['std']:.1f} | "
              f"FHV {a['fhv']['mean']:+.1f}+-{a['fhv']['std']:.1f}", flush=True)

    json.dump({"rows": rows, "agg": agg}, open(OUT / "c3_painel_validacao.json", "w"), indent=2, default=float)
    import csv
    with open(OUT / "c3_painel_validacao.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_diverge = sum(1 for r in rows if r["delta_sanidade"] > 0.01)
    print(f"\nsanidade: {len(rows) - n_diverge}/{len(rows)} runs batem com best_val_nse (tol 0.01)", flush=True)
    print(f"salvo: {OUT / 'c3_painel_validacao.json'} + .csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
