"""LSTM LUMPED, mínima: chuva AGREGADA da bacia (1 série) -> vazão. Com vs sem feature temporal.

O teste mais limpo do efeito da feature de calendário. Entrada por timestep:
  modo A (3 colunas: index, chuva, vazão): só log1p(chuva agregada)            -> 1 canal
  modo B (4 colunas: +feature temporal):   log1p(chuva agregada) + hora + mês  -> 3 canais
Chuva agregada = média areal (ponderada por área) das 245 ottobacias.
Lookback 240h, prevê 24h. Reporta NSE por horizonte, média de N seeds.

Uso: uv run python src/lumped_lstm.py --seeds 5
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import datetime as dt
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import h5py
import numpy as np
import torch
import torch.nn as nn

from backbones import make_backbone, n_params

DATA = Path(__file__).resolve().parent.parent / "data"
L, H = 240, 24


def carrega(produto: str = "telem"):
    with h5py.File(DATA / f"dataset_58585000_{produto}.h5", "r") as f:
        area = f["ottobacia/area_km2"][:].astype(np.float32)        # (245,)
        out = {}
        for sp in ["train", "val", "test"]:
            P = np.clip(f[f"{sp}/precipitation"][:].astype(np.float32), 0, None)  # (T,245)
            Q = f[f"{sp}/streamflow"][:].astype(np.float32)
            TS = f[f"{sp}/timestamps"][:].astype(np.int64)
            Pag = (P * area[None, :]).sum(1) / area.sum()           # (T,) chuva média areal da bacia
            hora = np.array([dt.datetime.fromtimestamp(int(t), tz=dt.timezone.utc).hour for t in TS], np.float32) / 23.0
            mes = np.array([(dt.datetime.fromtimestamp(int(t), tz=dt.timezone.utc).month - 1) for t in TS], np.float32) / 11.0
            out[sp] = {"Pag": Pag, "Q": Q, "hora": hora, "mes": mes}
    return out


def janelas(Pag, Q):
    T = Q.shape[0]
    t_ini, t_fim = L - 1, T - 1 - H
    qn = np.concatenate([[0], np.cumsum(np.isnan(Q))])
    pn = np.concatenate([[0], np.cumsum(np.isnan(Pag))])
    ts = np.arange(t_ini, t_fim + 1)
    alvo = (qn[ts + 1 + H] - qn[ts + 1]) > 0
    look = (pn[ts + 1] - pn[ts - L + 1]) > 0
    return ts[~(alvo | look)]


def nse_np(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    den = np.sum((obs - obs.mean()) ** 2)
    return float(1 - np.sum((sim - obs) ** 2) / den) if den > 0 else float("nan")


class LumpedLSTM(nn.Module):
    def __init__(self, n_in, hid=64):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hid, num_layers=2, batch_first=True, dropout=0.1)
        self.dec = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hid, H))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.dec(h[-1])


def set_det(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def roda_seed(seed, dev, dados, idxs, usar_chuva, usar_tempo, backbone="lstm", batch=1024, epochs=150, pac=25):
    set_det(seed)
    T = {sp: {k: torch.as_tensor(np.nan_to_num(v), device=dev) for k, v in dados[sp].items()} for sp in dados}
    look = torch.arange(-(L - 1), 1, device=dev)
    fwd = torch.arange(1, H + 1, device=dev)
    n_in = int(usar_chuva) + 2 * int(usar_tempo)

    def batches(sp, idx, shuffle, g=None):
        order = torch.randperm(len(idx), generator=g, device=dev) if shuffle else torch.arange(len(idx), device=dev)
        for i in range(0, len(idx), batch):
            t = idx[order[i:i + batch]]
            li = t.view(-1, 1) + look.view(1, -1)
            canais = []
            if usar_chuva:
                canais.append(torch.log1p(T[sp]["Pag"][li]).unsqueeze(-1))           # (B,L,1)
            if usar_tempo:
                canais += [T[sp]["hora"][li].unsqueeze(-1), T[sp]["mes"][li].unsqueeze(-1)]
            x = torch.cat(canais, -1)
            y = T[sp]["Q"][t.view(-1, 1) + fwd.view(1, -1)]               # (B,H)
            yield x, y

    model = make_backbone(backbone, n_in, H=H, L=L).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)   # alinhado à grade
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=7, min_lr=1e-6)
    g = torch.Generator(device=dev).manual_seed(seed)
    it = {sp: torch.as_tensor(idxs[sp], device=dev) for sp in dados}
    best, ruim = -1e9, 0
    best_state = {k: t.clone() for k, t in model.state_dict().items()}   # inicial (nunca None)
    for ep in range(epochs):
        model.train()
        for x, y in batches("train", it["train"], True, g):
            pl = model(x)
            loss = nn.functional.mse_loss(pl, torch.log1p(torch.clamp(y, min=0))) \
                + 0.01 * nn.functional.mse_loss(torch.expm1(torch.clamp(torch.relu(pl), max=15.0)), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(loss):       # pula passo se divergiu (não propaga NaN)
                opt.step()
        model.eval(); s_, o_ = [], []
        with torch.no_grad():
            for x, y in batches("val", it["val"], False):
                s_.append(torch.expm1(torch.clamp(torch.relu(model(x)), max=15.0)).cpu().numpy()); o_.append(y.cpu().numpy())
        vi = min(5, H - 1)   # early-stopping: 6h no forecasting (H>=6), 1h na simulacao continua (H=1)
        v = nse_np(np.concatenate(s_)[:, vi], np.concatenate(o_)[:, vi])
        if np.isfinite(v):
            sched.step(v)   # scheduler igual ao da grade (no val NSE)
        if np.isfinite(v) and v > best + 1e-5:
            best, best_state, ruim = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            ruim += 1
            if ruim >= pac:
                break
    model.load_state_dict(best_state)
    model.eval(); s_, o_ = [], []
    with torch.no_grad():
        for x, y in batches("test", it["test"], False):
            s_.append(torch.expm1(torch.clamp(torch.relu(model(x)), max=15.0)).cpu().numpy()); o_.append(y.cpu().numpy())
    s, o = np.concatenate(s_), np.concatenate(o_)
    return {h: nse_np(s[:, h - 1], o[:, h - 1]) for h in [1, 3, 6, 12, 24] if h <= H}, ep + 1


MODOS = {
    "chuva": (True, False, "A: SO CHUVA AGREGADA (1 canal)"),
    "ambos": (True, True, "B: CHUVA+HORA/MES (3 canais)"),
    "tempo": (False, True, "C: SO FEATURE TEMPORAL, SEM CHUVA (2 canais)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-list", default="", help="sementes especificas, ex 0,1,2 (sobrepoe --seeds; p/ dividir entre GPUs)")
    ap.add_argument("--modo", default="chuva,tempo,ambos", help="A=chuva, C=tempo, B=ambos")
    ap.add_argument("--backbones", default="lstm", help="lista: lstm,gru,tcn,transformer,dlinear,mamba,mlp")
    ap.add_argument("--produtos", default="telem", help="lista: telem,merge,imerg,mswep,era5land")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--L", type=int, default=240, help="lookback (h)")
    ap.add_argument("--H", type=int, default=24, help="horizonte: 24=forecasting (default), 1=simulacao continua")
    ap.add_argument("--data", default="", help="diretorio com os .h5 (default: ../data)")
    a = ap.parse_args()
    global L, H, DATA, MH
    L, H = a.L, a.H
    MH = 6 if H >= 6 else H   # horizonte-resumo: 6h no forecasting, 1h na simulacao continua
    if a.data:
        DATA = Path(a.data)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modos = [m.strip() for m in a.modo.split(",")]
    bbs = [b.strip() for b in a.backbones.split(",")]
    produtos = [p.strip() for p in a.produtos.split(",")]
    seeds_iter = [int(x) for x in a.seed_list.split(",")] if a.seed_list.strip() else list(range(a.seeds))
    tabela = {}  # (produto, backbone, modo) -> (media@6h, std@6h)
    for prod in produtos:
        dados = carrega(prod)
        idxs = {sp: janelas(dados[sp]["Pag"], dados[sp]["Q"]) for sp in dados}
        print(f"\n##### PRODUTO={prod} | device={dev} | janelas: " +
              " ".join(f"{sp}={len(idxs[sp])}" for sp in idxs))
        for bb in bbs:
            npar = n_params(make_backbone(bb, 3, H=H, L=L))
            print(f"=== produto={prod} backbone={bb} (~{npar/1000:.0f}k params) ===")
            for chave in modos:
                usar_chuva, usar_tempo, rot = MODOS[chave]
                res = []
                for seed in seeds_iter:
                    met, ep = roda_seed(seed, dev, dados, idxs, usar_chuva, usar_tempo,
                                        backbone=bb, batch=a.batch, epochs=a.epochs)
                    res.append(met)
                m6 = [r[MH] for r in res]
                tabela[(prod, bb, chave)] = (float(np.mean(m6)), float(np.std(m6)))
                ext24 = np.mean([r[24] for r in res]) if H >= 24 else float("nan")
                print(f"  {rot}: @{MH}h={np.mean(m6):.3f}±{np.std(m6):.3f}  "
                      f"(@1h={np.mean([r[1] for r in res]):.3f} @24h={ext24:.3f})", flush=True)
                print(f"    PERSEED {chave} @{MH}h: " + ",".join(f"{sd}={r[MH]:.4f}" for sd, r in zip(seeds_iter, res)), flush=True)
    # tabela resumo (produto, backbone) x modo (NSE@6h)
    print(f"\n{'='*72}\nRESUMO NSE@{MH}h (média±σ, {len(seeds_iter)} seeds) — A/B/C por produto x arquitetura\n{'='*72}")
    print(f"{'produto':10s} {'backbone':12s} {'A:só chuva':>14} {'C:só tempo':>14} {'B:chuva+tempo':>15}")
    for prod in produtos:
        for bb in bbs:
            cels = []
            for chave in ["chuva", "tempo", "ambos"]:
                if (prod, bb, chave) in tabela:
                    m, s = tabela[(prod, bb, chave)]; cels.append(f"{m:.3f}±{s:.3f}")
                else:
                    cels.append("—")
            print(f"{prod:10s} {bb:12s} {cels[0]:>14} {cels[1]:>14} {cels[2]:>15}")


if __name__ == "__main__":
    main()
