# DUH-SCS-TOPMODEL-LSTM — Continuous Simulation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21016683.svg)](https://doi.org/10.5281/zenodo.21016683)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A differentiable hybrid rainfall–runoff model for **hourly continuous simulation**: a
**distributed unit hydrograph (DUH)** for routing, coupled with **SCS-CN** and **TOPMODEL**
runoff generation and an integrating **LSTM**. This repository is the full reproducibility
package for a multi-seed nested-baseline ablation in the Preto River catchment (Manuel Duarte,
ANA gauge 58585000, ~3,117 km², Brazilian Atlantic Forest).

> **Non-autoregressive premise:** the model simulates the streamflow series `Q` from
> precipitation only — observed streamflow is never an input. The continuous series emerges
> from sliding a one-step (horizon = 1) window over the record.

This repository accompanies the article *"Where and how does embedded physics help a neural
streamflow simulator? A differentiable routing-and-runoff model coupled to an LSTM, tested by
multi-seed ablation in continuous simulation"* (Azevedo, Oliveira & Fagundes, *Journal of Hydrology*,
manuscript in preparation). It is the continuous-simulation companion of the forecasting package
[`duh-scs-topmodel-lstm-forecasting-ar`](https://github.com/viniciusazeved/duh-scs-topmodel-lstm-forecasting-ar).

## What is in the study

An **attribution chain of nested baselines**, from pure physics to the pure network:

- **13 configurations** — from `LSTM_Lumped_CalOnly` / `LSTM_Lumped_RainOnly` (information floor)
  through the coupled `LSTM_Lumped`, the distributed `LSTM`, the routed `LSTM_DUH`, the runoff
  generators (`_SCS`, `_Topmodel`, with `_PeOnly` and `_Baseflow` variants), down to the
  physics-only baselines (`Phys_DUH_SCS`, `Phys_DUH_Topmodel`).
- **10 random seeds** (42–51), with paired Wilcoxon tests.
- Auxiliary contrasts (same protocol and seeds): `LSTM_Topmodel` / `LSTM_SCS` (generation without
  routing), `LSTM_PET` (evaporative forcing), `LSTM_DUH_Manning` (concentration-time sensitivity,
  Supplementary).

A separate **ABC experiment** (`scripts/abc/`, 5 seeds) sweeps **network architectures**
(LSTM, GRU, Transformer, MLP, DLinear) and **precipitation products** (telemetric, MERGE,
ERA5-Land, IMERG, MSWEP) in a lumped setting, to separate the role of information from that of
architecture and forcing.

The central finding: in continuous simulation forced by precipitation alone, **skill arises first
from information** — the coupling of rainfall and the calendar takes NSE from negative (rainfall
alone) to 0.665, while architecture and embedded physics are secondary on the mean. Physics does
not lift mean NSE significantly here, but it helps at **low flows** (volume bias) and, suggestively,
in **robustness**; the best hybrid reaches NSE 0.831 (seed mean) and 0.885 (ensemble).

## Repository layout

```
duh-scs-topmodel-lstm-continuous/
├── src/ttd_scs_lstm/        # model package
│   ├── models/              #   models.py (configurations), topmodel_diff.py (differentiable TOPMODEL)
│   └── data/                #   dataset.py, temporal.py
├── scripts/
│   ├── run_sim.py           # attribution-chain runner (13 configs x seeds, horizon=1)
│   ├── train.py             # training of one configuration / seed
│   ├── run_noroute.py       # LSTM_Topmodel / LSTM_SCS (generation without routing)
│   ├── run_pet.py           # LSTM_PET (evaporative forcing)
│   ├── run_manning.py       # LSTM_DUH_Manning (concentration-time sensitivity, SI)
│   ├── consolida_sim.py     # consolidates results.json across seeds/configs
│   ├── _tabela_artigo.py    # builds the attribution-chain table (mean +/- sd, ensemble)
│   ├── abc/                 # ABC experiment: backbones.py (architectures), lumped_lstm.py (products)
│   └── figures/             # gerar_figuras_sim.py, fig_abc.py, tabela_abc.py, fig_arquitetura_sim.py
├── data/
│   ├── dataset_58585000_telem.h5    # processed telemetric dataset (245 sub-catchments)
│   └── processed/twi_attrs.npz      # TWI histograms for the TOPMODEL layer
└── outputs/
    ├── grade_sim/           # per-run results.json + predictions.npz (10 seeds x configs)
    └── abc/                 # ABC logs (backbones_telem_gpu1.log = Table 3; products_lstm_gpu1.log = Table 4)
```

The per-run model weights (`best_model.pt`, ~67 MB total) are **not** stored in git — they are
archived on Zenodo (see *Data availability*). The repository keeps the lightweight `results.json`,
the consolidated tables, and the per-run `predictions.npz`, which are enough to rebuild every table
and figure.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

**PyTorch is intentionally not pinned** (the build depends on your CUDA setup):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Training was run on NVIDIA RTX 2000/3000 Ada GPUs; CPU works but is slow.

## Reproducing the experiments

**A single run** (smoke test, one epoch):

```bash
uv run python scripts/run_sim.py --test --seeds 42
```

**The attribution chain** (13 configurations x 10 seeds) and the auxiliary contrasts:

```bash
uv run python scripts/run_sim.py     --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 100 --patience 20
uv run python scripts/run_noroute.py --seeds 42 43 44 45 46 47 48 49 50 51   # LSTM_Topmodel, LSTM_SCS
uv run python scripts/run_pet.py     --seeds 42 43 44 45 46 47 48 49 50 51   # LSTM_PET
uv run python scripts/run_manning.py --seeds 42 43 44 45 46 47 48 49 50 51   # LSTM_DUH_Manning (SI)
```

**The ABC experiment** (5 seeds; the architecture and product sweeps; needs the gridded datasets,
see *Data availability*):

```bash
uv run python scripts/abc/backbones.py     # Table 3 (architectures)
uv run python scripts/abc/lumped_lstm.py   # Table 4 (precipitation products)
```

**Tables and figures** are rebuilt from the shipped results:

```bash
uv run python scripts/consolida_sim.py
uv run python scripts/_tabela_artigo.py
uv run python scripts/figures/gerar_figuras_sim.py   # Figures 5, 7-10 (chain, robustness, FDC, hydrograph, parameters)
uv run python scripts/figures/fig_abc.py             # Figure 6 (ABC)
```

Key configuration (fixed across the chain): horizon 1 (sliding one-step continuous simulation, no
streamflow feedback), lookback 240 h, batch 512, up to 150 epochs, early stopping (patience 25 on
the validation NSE of the simulated series), AdamW (weight decay 1e-5; learning rate 1e-3 for the
LSTM and 1e-2 for the physical parameters), ReduceLROnPlateau scheduler, loss = MSE(log1p) +
0.01·MSE(linear), gradient clipping (norm 1.0), determinism enabled. All timestamps are UTC.

> **Note on paths.** The model code (`src/`, `scripts/train.py`, `scripts/run_*.py`) runs from a
> clone. Some analysis and figure scripts (`scripts/figures/`, `scripts/abc/`) carry absolute paths
> from the development machine (`D:\…`); adjust the paths at the top before running. The shipped
> `results.json` / `predictions.npz` are enough to rebuild the tables and figures without re-running
> the experiments.

## Dataset format

`dataset_58585000_telem.h5`:

```
ottobacia/   area, cn_2022, tc, twi        (245 sub-catchments)
train/ val/ test/   precipitation (T, 245), streamflow (T,), pet (T, 245), timestamps (T,)
```

Temporal split (70/15/15): train 2021-01-01 → 2024-06, val → 2025-03, test → 2025-12-31.

## Data availability

The processed dataset and the lightweight results are in this repository; the per-run model weights
are archived on Zenodo:

> **Zenodo DOI:** [10.5281/zenodo.21016683](https://doi.org/10.5281/zenodo.21016683)

The gridded precipitation products used in the ABC experiment (MERGE, ERA5-Land, IMERG, MSWEP) and
the underlying raw data are publicly available from their original providers: streamflow telemetry
and the BHAE_CN-2022 curve-number product from ANA; MERGE from CPTEC/INPE; ERA5-Land from ECMWF;
IMERG from NASA GPM; MSWEP from GloH2O; land cover from MapBiomas; and the ANADEM digital terrain
model from IPH/UFRGS and ANA.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite both the article and this archived repository.

## License

[MIT](LICENSE) © 2026 Vinicius Azevedo, Paulo Tarso S. Oliveira, Hugo de Oliveira Fagundes.
