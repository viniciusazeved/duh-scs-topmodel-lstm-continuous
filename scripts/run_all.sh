#!/usr/bin/env bash
# Trains every configuration of the manuscript with its protocol (150 epochs, patience 25,
# batch 512, seeds 42 to 51) into the folder given as the first argument. From the repository root:
#     bash scripts/run_all.sh outputs/retrain
# Order of the original launcher (outputs/logs/orquestra_madrugada.sh): attribution chain, Manning,
# generation without routing, PET. In the manuscript rung 13 (Phys_DUH_Topmodel) was evaluated with
# seeds 42 and 43 only; run_sim.py trains it for every seed given.
set -euo pipefail
OUT="${1:?usage: bash scripts/run_all.sh <output folder>}"
PY="${PY:-uv run python}"
SEEDS="42 43 44 45 46 47 48 49 50 51"
PROTOCOL="--epochs 150 --patience 25 --batch-size 512"
$PY scripts/run_sim.py --seeds $SEEDS $PROTOCOL --out "$OUT"
$PY scripts/run_manning.py --seeds $SEEDS $PROTOCOL --out "$OUT"
$PY scripts/run_noroute.py --seeds $SEEDS $PROTOCOL --out "$OUT"
$PY scripts/run_pet.py --seeds $SEEDS $PROTOCOL --out "$OUT"
