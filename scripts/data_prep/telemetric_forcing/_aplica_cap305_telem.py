#!/usr/bin/env python
"""Aplica o cap305 ao telem da ablacao_v2 PRESERVANDO o PET.

Contexto: o cap130 (chute) cortava chuva real de verao (postos com 150-212 mm/h viravam
NaN antes do IDW). O cap correto e 305 (recorde mundial horario). O `cap305.h5` regerado
em outputs/qc_chuva_analise/ NAO tem PET (foi gerado do dataset_v2 antigo) -> os modelos
TOPMODEL/fisicos cairiam no fallback de climatologia. Solucao: pegar a base que TEM PET
(o telem cap130 da big ablacao) e enxertar SO a precip cap305 (unica coisa que difere).

Nao toca em nada oficial. Reconstroi ablacao_v2/data/dataset_58585000_telem.h5.
"""
import shutil
from pathlib import Path
import h5py
import numpy as np

BASE_COM_PET = Path(r"D:/TTD_SCS_LSTM/big_ablacao/data/dataset_58585000_telem.h5")  # cap130 + pet
PRECIP_CAP305 = Path(r"D:/TTD_SCS_LSTM/outputs/qc_chuva_analise/dataset_58585000_telem_cap305.h5")
DST = Path(r"D:/TTD_SCS_LSTM/ablacao_v2/data/dataset_58585000_telem.h5")

SPLITS = ["train", "val", "test"]

# 1. base limpa (com pet) -> destino
shutil.copy(BASE_COM_PET, DST)

# 2. enxerta a precip cap305
with h5py.File(PRECIP_CAP305, "r") as src, h5py.File(DST, "a") as dst:
    for sp in SPLITS:
        new = src[f"{sp}/precipitation"][:]
        del dst[f"{sp}/precipitation"]
        dst.create_dataset(f"{sp}/precipitation", data=new, compression="gzip")
    dst.attrs["precip_fonte"] = "telem ANA IDW(p=2) dinamico/hora por ottobacia | cap=305 mm/h (ablacao_v2)"
    dst.attrs["cap_mm_h"] = 305.0

# 3. VALIDACAO
print("=== VALIDACAO pos-enxerto ===")
ok = True
with h5py.File(DST, "r") as d, h5py.File(BASE_COM_PET, "r") as b, h5py.File(PRECIP_CAP305, "r") as c:
    # 3a. PET sobreviveu nos 3 splits?
    for sp in SPLITS:
        has = f"{sp}/pet" in d
        print(f"  [{sp}] pet presente: {has}")
        ok &= has
    # 3b. precip == cap305 exato?
    for sp in SPLITS:
        same = np.array_equal(d[f"{sp}/precipitation"][:], c[f"{sp}/precipitation"][:])
        print(f"  [{sp}] precip == cap305: {same}")
        ok &= same
    # 3c. test identico ao original (alinhamento de colunas)?  train/val: exatamente 1715 diff, todas >=
    for sp in SPLITS:
        a = d[f"{sp}/precipitation"][:]
        o = b[f"{sp}/precipitation"][:]
        diff = ~np.isclose(a, o, rtol=0, atol=1e-6)
        n = int(diff.sum())
        maior = int((a[diff] >= o[diff]).all()) if n else 1
        print(f"  [{sp}] vs cap130: {n} celulas diff | todas cap305>=cap130: {bool(maior)} | max {a.max():.1f}")
    # 3d. streamflow/timestamps/ottobacia intactos?
    for k in ["train/streamflow", "val/streamflow", "test/streamflow",
              "train/timestamps", "test/timestamps",
              "ottobacia/cn_2022", "ottobacia/area_km2", "ottobacia/tc_base_h", "ottobacia/tc_manning_h"]:
        same = np.array_equal(np.nan_to_num(d[k][:]), np.nan_to_num(b[k][:]))
        ok &= same
        if not same:
            print(f"  [ALERTA] {k} MUDOU (nao deveria)")
    print(f"  estaticos/vazao/timestamps intactos: {ok}")
print()
print("RESULTADO:", "OK — telem v2 = cap305 + PET preservado" if ok else "FALHOU — revisar")
