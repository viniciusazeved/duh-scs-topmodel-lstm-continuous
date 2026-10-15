#!/usr/bin/env bash
# Orquestrador da madrugada (21/06): espera a grade principal (grade_sim, 120 .done) terminar e
# encadeia os tres experimentos engatilhados, pra GPU nao ficar ociosa:
#   1) Manning  (run_manning.py:  LSTM_DUH_Manning + _Fixed x 10 seeds)
#   2) sem-DUH  (run_noroute.py:  LSTM_Topmodel + LSTM_SCS x 10 seeds, IUH=impulso)
#   3) PET      (run_pet.py:      LSTM_PET x 10 seeds, forcante evaporativa como feature)
# Tudo cai em outputs/grade_sim/ e consolida com consolida_sim.py.
set -u
cd /d/TTD_SCS_LSTM/ablacao_skill || exit 1
LOG=outputs/orquestra_madrugada.log
UV="uv run --project /d/TTD_SCS_LSTM python"
SEEDS="42 43 44 45 46 47 48 49 50 51"
EP="--epochs 150 --patience 25"

echo "$(date '+%F %T') orquestrador iniciado; aguardando grade principal (>=120 .done)" >> "$LOG"
while [ "$(find outputs/grade_sim -name .done 2>/dev/null | wc -l)" -lt 120 ]; do sleep 300; done
echo "$(date '+%F %T') grade principal completa -> EXP 1: Manning" >> "$LOG"

$UV scripts/run_manning.py --seeds $SEEDS $EP >> outputs/sim_manning.log 2>&1
echo "$(date '+%F %T') Manning completo -> EXP 2: sem-DUH (geracao sem roteamento)" >> "$LOG"

$UV scripts/run_noroute.py --seeds $SEEDS $EP >> outputs/sim_noroute.log 2>&1
echo "$(date '+%F %T') sem-DUH completo -> EXP 3: PET (forcante evaporativa)" >> "$LOG"

$UV scripts/run_pet.py --seeds $SEEDS $EP >> outputs/sim_pet.log 2>&1
echo "$(date '+%F %T') PET completo -> FIM da fila da madrugada" >> "$LOG"
