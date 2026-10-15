# Changelog

## v1.1.0 (2026-10-15): version of the manuscript submitted to the Journal of Hydrology

The manuscript "Understanding the role of physics in differentiable hydrological models for hourly
streamflow prediction" was submitted to the Journal of Hydrology on 13 October 2026.

### Added

- Model weights (`best_model.pt`) of the 172 completed runs. Version 1.0.0 stated that they were
  archived on Zenodo, which was not the case.
- Processed per-sub-catchment MERGE, ERA5-Land and IMERG datasets used in the rainfall-product
  sweep; rain-gauge metadata and hourly series; catchment and sub-catchment geometry.
- Analysis scripts for the tables, tests and statistics of the manuscript and of the Supplementary
  Information (`scripts/analysis/`): Table 3 and Table S3, seed-paired Wilcoxon contrasts with the
  tails reported in the text, Brown-Forsythe and bootstrap spread tests, FLV and Yilmaz signatures,
  learned-parameter summaries, test-period diagnostics and the validation panel of Table S4.
- GR4H benchmark (R, airGR), the script that rebuilds its input from the h5, and its evaluation on
  the common 6,323-hour mask.
- Figure generators of the submitted version and their inputs (`scripts/figures/`, `data/gis/`).
- Provenance of the processed datasets (`scripts/data_prep/`, with a README).
- Training logs of the grid and the launcher used (150 epochs, patience 25); frozen environment.
- `scripts/run_all.sh`, which trains every configuration with the protocol of the manuscript.
- `LICENSE-DATA.md` (CC BY 4.0 for data and weights, with third-party attributions) and
  `.zenodo.json`.

### Changed

- `README.md` and `CITATION.cff`: title and status of the manuscript, concept DOI, table and figure
  numbers of the submitted version, metrics with two decimals as in the manuscript, run commands
  (150 epochs and patience 25 instead of 100 and 20), dataset description.
- Runners: defaults set to the protocol of the manuscript; new `--out` option; the smoke test writes
  to `outputs/smoke` and no longer overwrites shipped results.
- `scripts/abc/lumped_lstm.py`: the default data folder is `data/`, and its docstring gives the
  commands of Tables S5 and S6 (batch 512, checked by rerunning seed 0 on the GPUs of the logs).
- Figure and analysis scripts use paths relative to the repository.
- `pyproject.toml`: direct and indirect dependencies pinned to the versions of the environment that
  ran the grid (Python 3.12 or later; optional `gis` extra for Figures 1 and S2); `uv.lock`
  regenerated.

### Unchanged

- Model code (`src/`), `scripts/train.py`, the processed telemetric dataset, the TWI file and all
  per-run results of v1.0.0.
