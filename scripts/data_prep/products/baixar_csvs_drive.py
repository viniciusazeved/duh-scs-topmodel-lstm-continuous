#!/usr/bin/env python
"""Baixa CSVs de um folder do Google Drive direto pro disco local, reusando as
credenciais do Earth Engine (escopo Drive). Bypassa o Drive desktop (stream lento).

Uso:  uv run python baixar_csvs_drive.py <folder_no_drive> <prefixo>
  ex:  baixar_csvs_drive.py encruz_precip_era5land era5land_otto
Saida: D:\\TTD_SCS_LSTM\\encruzilhada\\data\\precip_csv\\<folder>\\
"""

from __future__ import annotations

import sys
from pathlib import Path

import ee

ee.Initialize(project="ggeantigravity")
creds = ee.data.get_persistent_credentials()
print("credenciais EE:", type(creds).__name__, "| scopes:", getattr(creds, "scopes", "?"))

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

svc = build("drive", "v3", credentials=creds, cache_discovery=False)

folder = sys.argv[1] if len(sys.argv) > 1 else "encruz_precip_era5land"
prefix = sys.argv[2] if len(sys.argv) > 2 else "era5land_otto"
dest = Path(rf"D:\TTD_SCS_LSTM\encruzilhada\data\precip_csv\{folder}")
dest.mkdir(parents=True, exist_ok=True)

# Busca GLOBAL por nome (robusta a folders homonimos que o GEE cria com tasks
# paralelas). O 1o arg 'folder' agora so define a subpasta de destino local.
files, token = [], None
while True:
    resp = svc.files().list(
        q=f"name contains '{prefix}' and trashed=false and mimeType='text/csv'",
        fields="nextPageToken, files(id,name,size)", pageSize=100, pageToken=token,
        spaces="drive",
    ).execute()
    files.extend(resp.get("files", []))
    token = resp.get("nextPageToken")
    if not token:
        break
# dedup por nome (folders homonimos podem repetir), fica o maior (completo)
best = {}
for f in files:
    nm = f["name"]
    if nm not in best or int(f.get("size", 0)) > int(best[nm].get("size", 0)):
        best[nm] = f
files = list(best.values())
print(f"{len(files)} arquivos '{prefix}' (busca global por nome)")

baixados, falhas = 0, []
for f in sorted(files, key=lambda x: x["name"]):
    out = dest / f["name"]
    sz = int(f.get("size", 0))
    if out.exists() and (not sz or out.stat().st_size == sz):
        baixados += 1
        continue
    ok = False
    for tent in range(4):  # resiliente a IncompleteRead / quedas de conexao
        try:
            with open(out, "wb") as fh:
                dl = MediaIoBaseDownload(
                    fh, svc.files().get_media(fileId=f["id"]), chunksize=10 * 1024 * 1024)
                done = False
                while not done:
                    _, done = dl.next_chunk(num_retries=5)
            if sz and out.stat().st_size != sz:
                raise IOError(f"size {out.stat().st_size} != {sz}")
            ok = True
            break
        except Exception as e:
            print(f"  retry {tent+1}/4 em {f['name']}: {type(e).__name__}")
            if out.exists():
                out.unlink()
    if ok:
        baixados += 1
        print(f"  baixado {f['name']} ({out.stat().st_size/1e6:.0f} MB) [{baixados}/{len(files)}]")
    else:
        falhas.append(f["name"])
        print(f"  FALHOU {f['name']} apos 4 tentativas")
print(f"\n[OK] {baixados}/{len(files)} CSVs em {dest}")
if falhas:
    print(f"[AVISO] {len(falhas)} falharam (re-rodar pega): {falhas}")
    raise SystemExit(1)
