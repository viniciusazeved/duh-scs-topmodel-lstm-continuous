"""Banco de backbones de sequência com interface comum, para o experimento A/B/C entre arquiteturas.

Interface: cada backbone é nn.Module com forward(x: (B, L, n_in)) -> pred_log (B, H).
Objetivo: testar se o padrão A/B/C (feature temporal domina) é robusto à arquitetura, não só LSTM.
Capacidade aproximadamente pareada (~50-110k params); o nº exato é reportado no experimento.
Paradigmas cobertos: recorrência (LSTM/GRU), convolução causal (TCN), atenção (Transformer), linear (DLinear).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _decoder(hid: int, H: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, H))


class LSTMBackbone(nn.Module):
    def __init__(self, n_in: int, H: int = 24, hid: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.LSTM(n_in, hid, num_layers=layers, batch_first=True, dropout=dropout)
        self.dec = _decoder(hid, H, dropout)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return self.dec(h[-1])


class GRUBackbone(nn.Module):
    def __init__(self, n_in: int, H: int = 24, hid: int = 72, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(n_in, hid, num_layers=layers, batch_first=True, dropout=dropout)
        self.dec = _decoder(hid, H, dropout)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.dec(h[-1])


class _Chomp(nn.Module):
    """Remove o padding extra à direita -> convolução estritamente causal."""
    def __init__(self, n):
        super().__init__(); self.n = n

    def forward(self, x):
        return x[:, :, :-self.n] if self.n > 0 else x


class TCNBackbone(nn.Module):
    """Temporal Convolutional Network: blocos residuais de conv1d causal dilatada (dilatação 2^k)."""
    def __init__(self, n_in: int, H: int = 24, ch: int = 48, levels: int = 7, k: int = 3, dropout: float = 0.1):
        super().__init__()
        camadas = []
        for i in range(levels):
            d = 2 ** i
            c_in = n_in if i == 0 else ch
            pad = (k - 1) * d
            camadas += [
                nn.Conv1d(c_in, ch, k, padding=pad, dilation=d), _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(ch, ch, k, padding=pad, dilation=d), _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*camadas)   # campo receptivo ~ 1 + 2*(k-1)*(2^levels-1) >= 240 com levels=7
        self.dec = _decoder(ch, H, dropout)

    def forward(self, x):
        y = self.net(x.transpose(1, 2))      # (B,ch,L)
        return self.dec(y[:, :, -1])         # última posição (causal)


class _PosEnc(nn.Module):
    def __init__(self, d, L_max=512):
        super().__init__()
        pe = torch.zeros(L_max, d)
        pos = torch.arange(L_max).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerBackbone(nn.Module):
    """Encoder Transformer com atenção causal (máscara triangular) + positional encoding."""
    def __init__(self, n_in: int, H: int = 24, d: int = 64, nhead: int = 4, layers: int = 2,
                 ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_in, d)
        self.pos = _PosEnc(d)
        enc = nn.TransformerEncoderLayer(d, nhead, ff, dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.dec = _decoder(d, H, dropout)

    def forward(self, x):
        z = self.pos(self.proj(x))           # (B,L,d)
        L = z.size(1)
        mask = torch.triu(torch.ones(L, L, device=z.device, dtype=torch.bool), diagonal=1)  # causal
        z = self.enc(z, mask=mask)
        return self.dec(z[:, -1])            # último passo (causal)


class _MovingAvg(nn.Module):
    def __init__(self, k=25):
        super().__init__(); self.k = k; self.pool = nn.AvgPool1d(k, stride=1, padding=0)

    def forward(self, x):                    # (B,L,C)
        pad = (self.k - 1) // 2
        xp = torch.cat([x[:, :1].repeat(1, pad, 1), x, x[:, -1:].repeat(1, self.k - 1 - pad, 1)], dim=1)
        return self.pool(xp.transpose(1, 2)).transpose(1, 2)


class DLinearBackbone(nn.Module):
    """DLinear (Zeng et al. 2023): decompõe em tendência (moving avg) + sazonal, linear em cada um.
    Baseline LINEAR — se reproduz o A/B/C, o achado independe de não-linearidade."""
    def __init__(self, n_in: int, H: int = 24, L: int = 240, k: int = 25, dropout: float = 0.1):
        super().__init__()
        self.decomp = _MovingAvg(k)
        self.lin_trend = nn.Linear(L * n_in, H)
        self.lin_seas = nn.Linear(L * n_in, H)

    def forward(self, x):                    # (B,L,n_in)
        trend = self.decomp(x)
        seas = x - trend
        B = x.size(0)
        return self.lin_trend(trend.reshape(B, -1)) + self.lin_seas(seas.reshape(B, -1))


def _parallel_scan(dA, dBu):
    """Scan associativo (Hillis-Steele) da recorrência linear h[t] = dA[t]*h[t-1] + dBu[t].
    Elemento (A,B) representa h_t = A*h_inicio + B; composição (Ap,Bp)⊕(Ac,Bc) = (Ac*Ap, Ac*Bp+Bc).
    Faz o scan em log2(L) passos vetorizados (vs L sequenciais). dA/dBu: (B,L,di,ds). Retorna h (mesmo shape)."""
    A = dA
    Bc = dBu
    L = dA.shape[1]
    d = 1
    while d < L:
        A_prev = torch.cat([torch.ones_like(A[:, :d]), A[:, :-d]], dim=1)   # x[t-d].A (identidade p/ t<d)
        B_prev = torch.cat([torch.zeros_like(Bc[:, :d]), Bc[:, :-d]], dim=1)  # x[t-d].B
        Bc = A * B_prev + Bc                                                # compose: Ac*Bp + Bc
        A = A * A_prev                                                      # compose: Ac*Ap
        d *= 2
    return Bc                                                               # h[t] (com h[-1]=0)


class MambaBlock(nn.Module):
    """Bloco Mamba (SSM seletivo S6) em PyTorch puro — sem o kernel CUDA da lib mamba-ssm
    (que não compila no Windows). Scan associativo paralelo (Hillis-Steele); a matemática é idêntica."""
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand: int = 1):
        super().__init__()
        d_inner = expand * d_model
        dt_rank = math.ceil(d_model / 16)
        self.d_inner, self.d_state, self.dt_rank = d_inner, d_state, dt_rank
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))          # A = -exp(A_log) (estável, negativo)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):                                # (B,L,d_model)
        B, Lq, _ = x.shape
        xin, z = self.in_proj(x).chunk(2, dim=-1)        # cada (B,L,d_inner)
        xin = self.conv1d(xin.transpose(1, 2))[:, :, :Lq].transpose(1, 2)   # conv causal
        xin = F.silu(xin)
        # parâmetros seletivos (dependentes da entrada)
        dbl = self.x_proj(xin)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                # (B,L,d_inner)
        A = -torch.exp(self.A_log)                       # (d_inner,d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)             # (B,L,d_inner,d_state) em (0,1)
        dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xin.unsqueeze(-1)
        h = _parallel_scan(dA, dBu)                      # (B,L,d_inner,d_state) — Hillis-Steele O(log L)
        y = (h * Cm.unsqueeze(2)).sum(-1) + xin * self.D   # (B,L,d_inner)
        return self.out_proj(y * F.silu(z))

    def _scan_recorrente(self, dA, dBu):
        """Referência (loop) para validar o scan paralelo."""
        B, Lq = dA.shape[0], dA.shape[1]
        h = torch.zeros(B, self.d_inner, self.d_state, device=dA.device)
        ys = []
        for t in range(Lq):
            h = dA[:, t] * h + dBu[:, t]
            ys.append(h)
        return torch.stack(ys, dim=1)


class MambaBackbone(nn.Module):
    def __init__(self, n_in: int, H: int = 24, d: int = 64, n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_in, d)
        self.blocks = nn.ModuleList([MambaBlock(d) for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_blocks)])
        self.dec = _decoder(d, H, dropout)

    def forward(self, x):                                # (B,L,n_in)
        z = self.proj(x)
        for blk, norm in zip(self.blocks, self.norms):
            z = z + blk(norm(z))                         # residual
        return self.dec(z[:, -1])                        # último passo (causal)


class MLPBackbone(nn.Module):
    """Controle definitivo: flatten + MLP, SEM nenhuma estrutura temporal (nem conv, nem recorrência)."""
    def __init__(self, n_in: int, H: int = 24, L: int = 240, hid: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(L * n_in, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, H),
        )

    def forward(self, x):
        return self.net(x)


_BACKBONES = {
    "lstm": LSTMBackbone, "gru": GRUBackbone, "tcn": TCNBackbone,
    "transformer": TransformerBackbone, "dlinear": DLinearBackbone, "mamba": MambaBackbone,
    "mlp": MLPBackbone,
}


def make_backbone(nome: str, n_in: int, H: int = 24, L: int = 240) -> nn.Module:
    nome = nome.lower()
    cls = _BACKBONES[nome]
    if nome in ("dlinear", "mlp"):
        return cls(n_in, H=H, L=L)
    return cls(n_in, H=H)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
