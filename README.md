# DUH-SCS-TOPMODEL-LSTM: continuous simulation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21016682.svg)](https://doi.org/10.5281/zenodo.21016682)
[![Code: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA.md)

Reproducibility package of the manuscript

> Azevedo, V., Oliveira, P. T. S., & Fagundes, H. O. *Understanding the role of physics in
> differentiable hydrological models for hourly streamflow prediction.* Submitted to the
> *Journal of Hydrology* (2026).

Release v1.1.0 corresponds to the submitted manuscript. The concept DOI
[10.5281/zenodo.21016682](https://doi.org/10.5281/zenodo.21016682) resolves to the latest release;
each release also has its own version DOI on Zenodo. Changes since v1.0.0 are listed in
[CHANGELOG.md](CHANGELOG.md). The forecasting study has a companion package,
[`duh-scs-topmodel-lstm-forecasting-ar`](https://github.com/viniciusazeved/duh-scs-topmodel-lstm-forecasting-ar).

Contents: [installation](#installation), [reproducing the manuscript](#reproducing-the-manuscript),
[artefact map](#artefact-map), [datasets](#datasets),
[data provenance](scripts/data_prep/README.md), [GIS layers](data/gis/README.md),
[licences](#licences), [citation](#citation).

## Model and study design

A differentiable hybrid rainfall-runoff model for hourly continuous simulation: a distributed unit
hydrograph (DUH) for routing, SCS-CN or TOPMODEL runoff generation, and an integrating LSTM. Study
area: the Preto River catchment upstream of the Manuel Duarte gauge (ANA 58585000; 3,117 km²;
245 sub-catchments; Brazilian Atlantic Forest), with hourly data from 2021 to 2025.

Observed streamflow is never a model input; it is only the training target. The model simulates
hourly streamflow from rainfall, potential evapotranspiration (in the TOPMODEL layer, and as an
input of the auxiliary `LSTM_PET`), static sub-catchment attributes and calendar variables,
sliding a one-step window (horizon 1, lookback 240 h) over the record.

- Attribution chain of 13 nested configurations (manuscript Table 1), from the information floor
  (`LSTM_Lumped_CalOnly`, `LSTM_Lumped_RainOnly`) through the coupled lumped LSTM, the distributed
  `LSTM`, DUH routing and the SCS-CN and TOPMODEL generators (with `_PeOnly` and `_Baseflow`
  variants) to the physics-only baselines `Phys_DUH_SCS` and `Phys_DUH_Topmodel`.
- 10 random seeds (42 to 51) per configuration, except `Phys_DUH_Topmodel` (seeds 42 and 43),
  compared with seed-paired Wilcoxon tests.
- Auxiliary contrasts with the same protocol: the routing-free `LSTM_Topmodel` and `LSTM_SCS` and
  `LSTM_PET` (Section 4.4), and `LSTM_DUH_Manning` (Supplementary Sections S1 and S2).
  `LSTM_DUH_Manning_Fixed` was run alongside and is not reported in the manuscript.
- Two preparatory lumped sweeps, of network architectures and of rainfall products (5 seeds;
  Section 4.1 and Supplementary Section S5).
- A calibrated GR4H benchmark (Section 4.6 and Supplementary Section S6).

Main results over the test period (6,323 scored hours from 10 April to 31 December 2025;
manuscript Table 3):

| Configuration | NSE, mean ± sd over seeds | NSE, seed ensemble |
|---|---|---|
| Lumped LSTM, rainfall only | -0.09 ± 0.10 | -0.04 |
| Lumped LSTM, calendar only | 0.17 ± 0.05 | 0.20 |
| Lumped LSTM, rainfall and calendar | 0.67 ± 0.07 | 0.72 |
| Distributed LSTM (pure neural reference) | 0.76 ± 0.11 | 0.84 |
| LSTM + DUH + TOPMODEL with effective precipitation only (best hybrid) | 0.83 ± 0.06 | 0.88 |
| DUH + TOPMODEL without LSTM (seeds 42 and 43) | -0.40 | -0.40 |

Skill comes first from input information: rainfall alone performs poorly, the calendar encodings
capture the seasonality, and their coupling in the lumped LSTM reaches 0.67, while distributing the
rainfall over the 245 sub-catchments adds a modest gain. Network architecture and embedded physics
contribute little on average, although TOPMODEL reduces the low-flow bias (FLV from 6.0% to 4.2%)
and points to a more robust initialization. Without direct rainfall input the SCS-CN hybrid
degrades (NSE 0.77 to 0.59), whereas the TOPMODEL hybrid reaches the highest mean test NSE. The
calibrated GR4H benchmark reaches 0.45 on the same timestamps (0.48 when calibrated on KGE).

## Repository layout

```
src/ttd_scs_lstm/          model package: models/models.py, models/topmodel_diff.py, data/
scripts/
  train.py                 training and evaluation of one configuration and seed
  run_sim.py               attribution chain (13 configurations)
  run_noroute.py           LSTM_Topmodel, LSTM_SCS
  run_pet.py               LSTM_PET
  run_manning.py           LSTM_DUH_Manning, LSTM_DUH_Manning_Fixed
  run_all.sh               all runners with the protocol of the manuscript
  consolida_sim.py         quick consolidation of the grid (helper)
  _tabela_artigo.py        quick table, three decimals (helper)
  abc/                     preparatory sweeps: lumped_lstm.py (runner), backbones.py (module)
  analysis/                tables and statistics of the manuscript and of the SI
  gr4h/                    GR4H benchmark (R, airGR) and its evaluation on the common mask
  figures/                 generators of the figures and of Tables S5 and S6
  data_prep/               provenance of the processed datasets, as used (see its README)
data/
  dataset_58585000_telem.h5                    telemetric forcing used by all models
  dataset_58585000_{merge,era5land,imerg}.h5   gridded products (preparatory sweep only)
  processed/twi_attrs.npz                      TWI histograms of the TOPMODEL layer
  gauges/                                      27 telemetric rain gauges: metadata and hourly series
  gis/                                         sub-catchments, outline, Natural Earth (its README)
  gr4h/serie_horaria_gr4h.csv                  lumped GR4H input
outputs/
  grade_sim/seedNN/<configuration>/            config.json, summary.csv, all_results.json, .done
  grade_sim/seedNN/<configuration>/<model>/    results.json, predictions.npz, best_model.pt
  abc/                     logs of the preparatory sweeps
  gr4h/                    GR4H simulations, parameters and common-mask evaluation
  analysis/                validation panel (Table S4)
  logs/                    training logs of the grid and the launcher used
environment/frozen_env_2026-09-29.txt          packages of the environment that ran the grid
```

## Installation

```bash
uv sync --extra gis      # environment of the runs (uv.lock, Python 3.13); gis: Figures 1 and S2
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # or .../whl/cpu
```

PyTorch is not listed in `pyproject.toml` because the build depends on the CUDA setup. Install it
after `uv sync`, which removes the packages that are not in `uv.lock` (`uv run` keeps them). The
runs used torch 2.6.0+cu124 on NVIDIA RTX 2000 Ada and RTX 3000 Ada GPUs. Training results are
reproducible on the same card, not bitwise across cards (Supplementary Section S8).

## Reproducing the manuscript

### From the shipped results, without training

```bash
uv run python scripts/analysis/table3.py             # Table 3
uv run python scripts/analysis/contrasts.py          # paired tests, Sections 4.2 to 4.4 and S2
uv run python scripts/analysis/spread_tests.py       # seed-to-seed spread, Section 4.3
uv run python scripts/analysis/flv_signatures.py     # FLV and Yilmaz signatures, Table S7
uv run python scripts/analysis/learned_params.py     # learned parameters, Sections 4.5 and 5.3
uv run python scripts/analysis/test_period.py        # test-period diagnostics, Section 5.1
uv run python scripts/analysis/table_s3.py           # Table S3
uv run python scripts/analysis/validation_panel.py   # Table S4 (about 40 min on a CPU)
uv run python scripts/figures/tabela_abc.py          # Tables S5 and S6, from the sweep logs
uv run python scripts/figures/gerar_figuras_sim.py   # Figures 4 to 7 and S3 to S5
uv run python scripts/figures/fig_arquitetura_sim.py # Figure 3
uv run python scripts/figures/fig_abc.py             # Figure S1
uv run python scripts/figures/fig_produtos_evento.py # Figure S2 (MSWEP panel: see Datasets)
uv run python scripts/figures/fig_serie_vazao.py     # Figure 2
uv run python scripts/figures/mapa_localizacao.py    # Figure 1 (public layers: data/gis/README.md)
```

Each analysis script prints the unrounded values, the values rounded to the precision of the
manuscript and a comparison with the printed numbers (options: `--help`); the validation panel
reloads the 30 checkpoints of Table S4. Files go to `outputs/tables` (the CSV of the `--csv`
option, the JSON of the validation panel and the Markdown of Tables S5 and S6) and figures to
`outputs/figures`. With the locked environment and the Times New Roman font, the PNG files of
Figures 3 to 7, S1 and S3 to S5 are byte-identical to those of the submitted manuscript (checked on
Windows 11), and so are Figure 1, with the authors' copies of the public layers, and Figure S2,
with the MSWEP file. The PDF versions have the same content, with the embedded fonts in an order
that depends on where matplotlib is installed.

### GR4H benchmark

Requires R with the airGR package; the shipped simulations were made with R 4.6.1 and airGR 1.7.8.

```bash
uv run python scripts/gr4h/make_gr4h_input.py          # rebuilds the input from the h5, checks it
uv run python scripts/gr4h/audit_gr4h_common_mask.py   # scores the shipped simulations, 6,323 hours
Rscript scripts/gr4h/run_gr4h_airgr.R                  # NSE calibration, to outputs/retrain/gr4h
Rscript scripts/gr4h/run_gr4h_airgr_kge.R              # KGE calibration, same folder
# scores the new calibration
uv run python scripts/gr4h/audit_gr4h_common_mask.py --sim-dir outputs/retrain/gr4h
```

### Training

The runners skip runs that already have a `.done` marker, so the shipped results are not
overwritten; write new runs to a new folder. For 10 seeds, training takes from well under an hour
(lumped and distributed LSTM) to 6 to 8 GPU-hours (TOPMODEL configurations).

```bash
uv run python scripts/run_sim.py --test --seeds 42     # smoke test: one epoch, in outputs/smoke
bash scripts/run_all.sh outputs/retrain                # all configurations, seeds 42 to 51
```

Protocol (Table 2): batch 512, up to 150 epochs, early stopping with patience 25 on the validation
NSE, AdamW (learning rate 1e-3 for the network and 1e-2 for the physical parameters, weight decay
1e-5), ReduceLROnPlateau (factor 0.5, patience 7), gradient clipping at norm 1.0, loss
MSE(log1p) + 0.01 MSE(linear), cuDNN deterministic mode. `scripts/train.py` keeps defaults from an
earlier experiment; the runners pass the values above. In the manuscript rung 13
(`Phys_DUH_Topmodel`) was evaluated with seeds 42 and 43; `run_sim.py` trains every rung for every
seed it is given.

### Preparatory sweeps

Section 4.1 and Supplementary Section S5; seeds 0 to 4.

```bash
uv run python scripts/abc/lumped_lstm.py --H 1 --L 240 --seeds 5 --batch 512 \
    --backbones lstm,gru,transformer,mlp,dlinear --produtos telem          # Table S5
uv run python scripts/abc/lumped_lstm.py --H 1 --L 240 --seeds 5 --batch 512 \
    --backbones lstm --produtos telem,merge,era5land,imerg                 # Table S6 without MSWEP
```

The defaults of `lumped_lstm.py` (horizon 24, batch 1024) belong to the forecasting study. The
sweep logs record neither the batch size nor the data folder; rerunning seed 0 on the same GPU
models (October 2026) reproduced the logged values to four decimals only with batch 512 and the
telemetric file in `data/` (readings above 305 mm/h removed), on the RTX 2000 Ada for the logs of
Tables S5 and S6 and on the RTX 3000 Ada for the comparison of Section S8.

## Artefact map

| Manuscript artefact | Script | Data |
|---|---|---|
| Table 1 | `scripts/run_sim.py` (list `MODELS`) | |
| Table 2 | `scripts/train.py`, `scripts/run_all.sh` | `outputs/grade_sim/**/config.json`, `outputs/logs/` |
| Table 3 | `scripts/analysis/table3.py` | `outputs/grade_sim/` |
| Figure 1 | `scripts/figures/mapa_localizacao.py` | `data/gis/`, `data/gauges/`, public layers |
| Figure 2 | `scripts/figures/fig_serie_vazao.py` | `data/dataset_58585000_telem.h5` |
| Figure 3 | `scripts/figures/fig_arquitetura_sim.py` | |
| Figures 4 to 7 | `scripts/figures/gerar_figuras_sim.py` | `outputs/grade_sim/`, telemetric h5 |
| Sections 4.2 to 4.5, 5.1 and 5.3 | `scripts/analysis/` | `outputs/grade_sim/`, telemetric h5 |
| Section 4.6 and Section S6 | `scripts/gr4h/` | `data/gr4h/`, `outputs/gr4h/` |
| Tables S1 and S2 | `scripts/data_prep/tc/` | `ottobacia/tc_manning_h` in the h5 |
| Section S2 | `scripts/analysis/contrasts.py` | `outputs/grade_sim/` |
| Table S3 | `scripts/analysis/table_s3.py` | `outputs/grade_sim/` |
| Table S4 | `scripts/analysis/validation_panel.py` | `best_model.pt`; recorded output in `outputs/analysis/` |
| Tables S5 and S6, Figure S1 | `scripts/abc/lumped_lstm.py`<br>`scripts/figures/tabela_abc.py`<br>`scripts/figures/fig_abc.py` | `outputs/abc/backbones_telem_gpu1.log`<br>`outputs/abc/products_lstm_gpu1.log`<br>product h5 |
| Figure S2 | `scripts/figures/fig_produtos_evento.py` | product h5, `data/gis/ottobacias_tc.gpkg` |
| Section S7 | `scripts/data_prep/diagnostics/` | telemetric and product h5 |
| Section S8 | `scripts/figures/tabela_abc.py` | `outputs/abc/products_lstm_gpu0.log` (RTX 3000 Ada)<br>against `products_lstm_gpu1.log` (RTX 2000 Ada) |
| Figures S3 to S5 | `scripts/figures/gerar_figuras_sim.py` | `outputs/grade_sim/` |
| Section S11, Table S7, Eqs. S1 and S2 | `scripts/analysis/flv_signatures.py` | `outputs/grade_sim/` |

Other logs in `outputs/abc/`: `backbones_telem_gpu1b.log` (earlier run with the Adam optimizer,
superseded), `backbones_telem_gpu0.log` (incomplete), `backbones_telem_gpu0_mamba.log` and
`verify_gpu*.log` (headers only). `outputs/logs/phys_topmodel_gpu0.log` and `_gpu1.log` are the
interrupted seeds 48 and 44 of `Phys_DUH_Topmodel`, whose folders keep only `config.json`.

## Datasets

`data/dataset_58585000_telem.h5` (timestamps in UTC):

```
ottobacia/area_km2, cn_2022, tc_base_h, tc_manning_h   (245,)    sub-catchment attributes
{train,val,test}/precipitation   (T, 245)  mm/h; IDW of 27 ANA telemetric gauges, readings
                                           above 305 mm/h removed (root attribute cap_mm_h = 305)
{train,val,test}/pet             (T, 245)  mm/h; daily Hargreaves-Samani from ERA5-Land 2 m
                                           temperature, spread uniformly over the UTC day
{train,val,test}/streamflow      (T,)      m³/s; ANA 58585000, hourly mean of the 15 min record
{train,val,test}/timestamps      (T,)      UTC
```

Splits (UTC): train 2021-01-01 00:00 to 2024-06-30 00:00; validation 2024-06-30 01:00 to
2025-03-31 00:00; test 2025-03-31 01:00 to 2025-12-31 00:00. With lookback 240 h and horizon 1 the
valid windows are 27,292, 5,998 and 6,323; the first 240 h of each split serve only as lookback.
The `start`/`end` attributes of the split groups are in local time (UTC-3), and the root `created`
and `description` attributes are inherited from an earlier build; rely on `timestamps`. The TWI
histograms are in `data/processed/twi_attrs.npz`, not in the h5.

`dataset_58585000_{merge,era5land,imerg}.h5` share the attributes, PET, streamflow and timestamps
of the telemetric file and differ only in `precipitation` (area-weighted mean per sub-catchment).
MERGE has 125 hours without data (2023 to 2025), hence fewer valid windows. ERA5-Land and IMERG
were aggregated in Google Earth Engine (polygon reduction at 1 km scale, with the centroid value
for 8 and 7 sub-catchments smaller than a pixel).

MSWEP V3.16 is distributed by GloH2O under CC BY-NC 4.0, with terms that restrict redistribution,
so `dataset_58585000_mswep.h5` is not included.
`scripts/data_prep/products/montar_dataset_mswep_area.py` rebuilds it from the MSWEP files
obtained from GloH2O. Without it, the MSWEP runs of the sweep
cannot be retrained and its panel in Figure S2 is replaced by a note; the MSWEP row of Table S6
and its bar in Figure S1 are drawn from the logs in `outputs/abc/`.

## Notes on reproducibility

- Code provenance. `src/ttd_scs_lstm/models/models.py`, `scripts/train.py` and `scripts/run_sim.py`
  are the final versions used for the grid. They were last edited on 22 June 2026, after most runs,
  to add rung 13 and to fix the PET input check of `LSTM_PET`, which was rerun afterwards. Reloading
  the shipped checkpoints with this code reproduces the stored predictions of all 18 configurations
  (NSE differences below 3e-4).
- Determinism. The grid fixes the seeds and sets cuDNN to deterministic mode, without
  `torch.use_deterministic_algorithms`, and leaves TF32 at the PyTorch default; the preparatory
  sweeps use deterministic algorithms. Retraining on other hardware reproduces the results closely,
  not bitwise.
- Helpers. `scripts/consolida_sim.py` tests every contrast one-sided (alternative "greater") and
  its chain omits rung 13; the p-values of the manuscript, with the tails stated in the text, come
  from `scripts/analysis/contrasts.py`, and Table 3 from `scripts/analysis/table3.py`.

## Licences

Code: MIT ([LICENSE](LICENSE)). Data, model weights and results: CC BY 4.0, with the third-party
attributions listed in [LICENSE-DATA.md](LICENSE-DATA.md) (ANA; Copernicus ERA5-Land; ANADEM and
Copernicus WorldDEM-30; MapBiomas; CPTEC/INPE MERGE; NASA IMERG; Natural Earth).

## Citation

Cite the version you used with its version DOI, shown on its Zenodo page, or the concept DOI,
which covers all versions and resolves to the latest one:

> Azevedo, V., Oliveira, P. T. S., & Fagundes, H. O. (2026). DUH-SCS-TOPMODEL-LSTM (Continuous
> Simulation): a differentiable hybrid rainfall-runoff model for hourly continuous streamflow
> simulation [Software]. Zenodo. https://doi.org/10.5281/zenodo.21016682

and the article once published. Machine-readable metadata: [CITATION.cff](CITATION.cff).

## Use of generative AI

Parts of the model, training and analysis code were written with the assistance of generative AI
coding tools (Claude, Anthropic; Codex, OpenAI), and all code was reviewed and tested by the
authors.
