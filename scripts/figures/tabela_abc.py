#!/usr/bin/env python
"""Parseia os dois logs do ABC em simulacao (h=1) e gera as duas tabelas:
  (1) ARQUITETURA x A/B/C  (produto fixo = telem)        <- escolhe LSTM (arquitetura secundaria)
  (2) PRODUTOS    x A/B/C  (arquitetura fixa = lstm)      <- escolhe telem (forcante importa)
Robusto a runs PARCIAIS (so mostra celulas concluidas). Re-rodar quando o ABC fechar.

Uso: uv run --project D:/TTD_SCS_LSTM python scripts/tabela_abc.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Re-run alinhado 23/06 (AdamW/512/scheduler, 5 seeds, lumped): os 5 backbones num único log na GPU1.
LOGS_ARCH = [
    Path(r"D:\Artigo_JOH\entrega_ABC\outputs_sim\backbones_telem_gpu1.log"),    # lstm,gru,transformer,dlinear,mlp
]
LOG_PROD = Path(r"D:\Artigo_JOH\entrega_ABC\outputs_sim\products_lstm_gpu1.log")  # RTX 2000 (canonico, bate com a grade); gpu0=RTX 3000 e o cross-check
OUT = Path(r"D:\Artigo_JOH\artigo_simulacao\tabela_abc.md")

BB_LABEL = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "transformer": "Transformer",
            "dlinear": "DLinear", "mamba": "Mamba", "mlp": "MLP"}
PROD_LABEL = {"telem": "Telemetric (gauges)", "merge": "MERGE", "era5land": "ERA5-Land",
              "imerg": "IMERG", "mswep": "MSWEP"}


def parse(path: Path):
    """(produto, backbone) -> {A|B|C: (media, dp)} a partir do log."""
    data, cur = {}, None
    if not path.exists():
        return data
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        h = re.search(r"produto=(\S+)\s+backbone=(\S+)", ln)
        if h:
            cur = (h.group(1), h.group(2))
            data.setdefault(cur, {})
            continue
        m = re.match(r"\s+([ABC]):.*?@1h=(-?[\d.]+)\D+([\d.]+)", ln)
        if m and cur:
            data[cur][m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return data


def cell(d, mode):
    if mode in d:
        mu, sd = d[mode]
        return f"{mu:+.3f}±{sd:.3f}"
    return "—"


def tabela(data, fixar_key, fixar_val, varia_idx, label_map, titulo, escolha):
    """Monta markdown: linhas = eixo que varia; colunas A/B/C. fixar = (idx, val)."""
    rows = {}
    for (prod, bb), d in data.items():
        chave = (prod, bb)
        if chave[fixar_key] != fixar_val:
            continue
        var = chave[varia_idx]
        rows[var] = d
    if not rows:
        return f"### {titulo}\n\n(sem células concluídas ainda)\n"
    # ordena por B decrescente (quem tiver B), senao no fim
    def keyB(kv):
        return -(kv[1]["B"][0] if "B" in kv[1] else -9)
    ordenado = sorted(rows.items(), key=keyB)
    out = [f"### {titulo}", "",
           "| " + label_map.get("_col", "Item") + " | A: rainfall only | C: calendar only | B: rainfall + calendar |",
           "|---|:---:|:---:|:---:|"]
    for var, d in ordenado:
        out.append(f"| {label_map.get(var, var)} | {cell(d,'A')} | {cell(d,'C')} | **{cell(d,'B')}** |")
    out += ["", f"NSE@1h de simulação (média±dp, 5 sementes). {escolha}", ""]
    return "\n".join(out)


def main():
    arch = {}
    for lg in LOGS_ARCH:     # reúne os 3 logs; tcn vazio do log antigo é sobrescrito pelo novo
        arch.update(parse(lg))
    prod = parse(LOG_PROD)   # backbone=lstm, varia produto (idx 0)
    BB_LABEL["_col"] = "Architecture"
    PROD_LABEL["_col"] = "Rainfall product"
    t1 = tabela(arch, fixar_key=0, fixar_val="telem", varia_idx=1, label_map=BB_LABEL,
                titulo="Tabela A — Arquitetura × A/B/C (produto = telemétrico)",
                escolha="Modo B (acoplamento) ≫ A e C em toda arquitetura; o topo comprime → arquitetura secundária, segue-se com a LSTM.")
    t2 = tabela(prod, fixar_key=1, fixar_val="lstm", varia_idx=0, label_map=PROD_LABEL,
                titulo="Tabela B — Produtos de chuva × A/B/C (arquitetura = LSTM)",
                escolha="Telemétrico é o teto; produtos de grade perdem skill → forçante importa (rumo a PUB).")
    txt = ("# Tabelas do experimento ABC (simulação contínua, h=1)\n\n"
           "Ordem narrativa (Opção 2): arquitetura → escolhe LSTM; depois produtos com LSTM → escolhe telem.\n"
           "PARCIAL enquanto o ABC roda; re-rodar `tabela_abc.py` quando fechar.\n\n"
           + t1 + "\n" + t2)
    OUT.write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
