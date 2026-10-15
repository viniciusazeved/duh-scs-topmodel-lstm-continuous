#!/usr/bin/env python
"""Copia a PET (Hargreaves/ERA5 por ottobacia) de um dataset MD que ja a tem para o
dataset_mswep.h5. A PET independe da chuva (so t2m + latitude + dia), entao e identica
entre produtos que compartilham a mesma grade de timestamps. Valida os timestamps antes."""
import h5py

SRC = r"D:\TTD_SCS_LSTM\big_ablacao\data\dataset_58585000_telem.h5"   # ja tem PET
DST = r"D:\TTD_SCS_LSTM\big_ablacao\data\dataset_58585000_mswep.h5"   # falta PET

with h5py.File(SRC, "r") as fs, h5py.File(DST, "a") as fd:
    for s in ("train", "val", "test"):
        ts_s = fs[s]["timestamps"][:]
        ts_d = fd[s]["timestamps"][:]
        assert ts_s.shape == ts_d.shape and (ts_s == ts_d).all(), f"timestamps {s} divergem - NAO copiar!"
        pet = fs[s]["pet"][:]
        if "pet" in fd[s]:
            del fd[s]["pet"]
        fd[s].create_dataset("pet", data=pet, compression="gzip")
        print(f"  {s}: PET {pet.shape} copiada (timestamps conferem)")
print("OK - PET injetada no dataset_58585000_mswep.h5")
