#!/usr/bin/env python
"""
Calculo de Tc com ajuste de Manning baseado em LULC.

Este script ajusta o tempo de concentracao (Tc) considerando a rugosidade
do escoamento superficial (coeficiente de Manning) derivada do uso do solo.

Metodologia:
1. Tc_base: Calculado por Maidment (V = f(slope, flow_acc))
2. Ajuste Manning: V_final = V_base × (n_ref / n_lulc)
3. Tc_manning = Tc_base × (n_lulc / n_ref)

Onde:
- n_ref = 0.035 (agropecuaria, usado como referencia no calculo original)
- n_lulc = coeficiente de Manning por classe LULC

Resultado:
- Floresta (n=0.15): Tc aumenta ~4.3x (mais lento)
- Urbano (n=0.015): Tc diminui ~2.3x (mais rapido)

Uso:
    python scripts/calculate_tc_manning.py
    python scripts/calculate_tc_manning.py --year 2022
"""

import os
import sys
from pathlib import Path
import json

# Fix PROJ conflict
import sysconfig
site_packages = sysconfig.get_path('purelib')
proj_dir = Path(site_packages) / "pyproj" / "proj_dir" / "share" / "proj"
if proj_dir.exists():
    os.environ["PROJ_LIB"] = str(proj_dir)
    os.environ["PROJ_DATA"] = str(proj_dir)

import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS as RioCRS

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
TC_BASE_FILE = DATA_DIR / "raw" / "rasters" / "tc_seconds.tif"
LULC_DIR = DATA_DIR / "processed" / "lulc"
OUTPUT_DIR = DATA_DIR / "processed" / "tc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Coeficientes de Manning por classe LULC
# Referencia: Chow (1959), Engman (1986)
MANNING_N = {
    1: 0.150,  # Floresta - alta rugosidade
    2: 0.100,  # Vegetacao Natural
    3: 0.035,  # Agropecuaria (REFERENCIA)
    4: 0.025,  # Nao Vegetada
    5: 0.030,  # Agua
    6: 0.035,  # Nao Observado (usa valor medio)
    7: 0.015,  # Urbano - baixa rugosidade
}

# Coeficiente de referencia (usado no calculo original do Tc)
N_REF = 0.035  # Agropecuaria

CLASS_NAMES = {
    1: "Floresta",
    2: "Vegetacao Natural",
    3: "Agropecuaria",
    4: "Nao Vegetada",
    5: "Agua",
    6: "Nao Observado",
    7: "Urbano"
}


def print_header(title: str) -> None:
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def calculate_tc_manning(tc_base_file: Path, lulc_file: Path, output_file: Path) -> dict:
    """
    Calcula Tc ajustado por Manning.

    Tc_manning = Tc_base × (n_lulc / n_ref)

    Onde maior Manning (mais rugoso) = maior Tc (mais lento)
    """
    print(f"\n  Carregando Tc base: {tc_base_file.name}")

    with rasterio.open(tc_base_file) as src_tc:
        tc_data = src_tc.read(1).astype(np.float32)
        tc_profile = src_tc.profile.copy()
        tc_transform = src_tc.transform
        tc_crs = src_tc.crs
        tc_shape = tc_data.shape

    print(f"    Shape: {tc_shape}")
    print(f"    Tc range: {np.nanmin(tc_data[tc_data > 0]):.0f} - {np.nanmax(tc_data):.0f} segundos")
    print(f"    Tc range: {np.nanmin(tc_data[tc_data > 0])/3600:.2f} - {np.nanmax(tc_data)/3600:.2f} horas")

    print(f"\n  Carregando LULC: {lulc_file.name}")

    with rasterio.open(lulc_file) as src_lulc:
        lulc_data = src_lulc.read(1)
        lulc_transform = src_lulc.transform
        lulc_crs = src_lulc.crs
        lulc_shape = lulc_data.shape

    print(f"    Shape: {lulc_shape}")

    # Verificar se precisamos reprojetar LULC para o grid do Tc
    if lulc_shape != tc_shape:
        print(f"\n  Reprojetando LULC para grid do Tc...")

        # Criar array para LULC reprojetado
        lulc_reprojected = np.zeros(tc_shape, dtype=np.uint8)

        reproject(
            source=lulc_data,
            destination=lulc_reprojected,
            src_transform=lulc_transform,
            src_crs=lulc_crs,
            dst_transform=tc_transform,
            dst_crs=tc_crs,
            resampling=Resampling.nearest  # Nearest para dados categoricos
        )

        lulc_data = lulc_reprojected
        print(f"    LULC reprojetado: {lulc_data.shape}")

    # Criar raster de Manning
    print(f"\n  Criando raster de Manning...")
    manning_raster = np.full(tc_shape, N_REF, dtype=np.float32)

    for class_id, n_value in MANNING_N.items():
        mask = lulc_data == class_id
        manning_raster[mask] = n_value
        count = np.sum(mask)
        if count > 0:
            print(f"    Classe {class_id} ({CLASS_NAMES[class_id]}): {count:,} pixels, n={n_value}")

    # Calcular fator de ajuste: n_lulc / n_ref
    # Maior Manning = maior fator = maior Tc (mais lento)
    adjustment_factor = manning_raster / N_REF

    print(f"\n  Fator de ajuste:")
    print(f"    Min: {np.min(adjustment_factor):.3f} (urbano, mais rapido)")
    print(f"    Max: {np.max(adjustment_factor):.3f} (floresta, mais lento)")
    print(f"    Mean: {np.mean(adjustment_factor):.3f}")

    # Calcular Tc ajustado
    # Tc_manning = Tc_base × (n_lulc / n_ref)
    tc_manning = tc_data * adjustment_factor

    # Manter nodata onde Tc_base era 0 ou nodata
    tc_manning[tc_data <= 0] = 0
    tc_manning[np.isnan(tc_data)] = np.nan

    # Estatisticas
    valid_mask = (tc_manning > 0) & ~np.isnan(tc_manning)
    tc_valid = tc_manning[valid_mask]
    tc_base_valid = tc_data[valid_mask]

    print(f"\n  Tc Manning:")
    print(f"    Range: {np.min(tc_valid):.0f} - {np.max(tc_valid):.0f} segundos")
    print(f"    Range: {np.min(tc_valid)/3600:.2f} - {np.max(tc_valid)/3600:.2f} horas")
    print(f"    Mean: {np.mean(tc_valid)/3600:.2f} horas")

    print(f"\n  Comparacao Tc_base vs Tc_manning:")
    print(f"    Tc_base mean:    {np.mean(tc_base_valid)/3600:.2f} horas")
    print(f"    Tc_manning mean: {np.mean(tc_valid)/3600:.2f} horas")
    print(f"    Diferenca:       {(np.mean(tc_valid) - np.mean(tc_base_valid))/3600:+.2f} horas ({(np.mean(tc_valid)/np.mean(tc_base_valid) - 1)*100:+.1f}%)")

    # Salvar
    print(f"\n  Salvando: {output_file}")

    tc_profile.update(
        dtype='float32',
        nodata=0,
        compress='lzw'
    )

    with rasterio.open(output_file, 'w', **tc_profile) as dst:
        dst.write(tc_manning, 1)

    print(f"    Tamanho: {output_file.stat().st_size / 1e6:.1f} MB")

    # Retornar estatisticas (converter para float nativo do Python)
    stats = {
        'tc_base': {
            'min_h': float(round(np.min(tc_base_valid) / 3600, 2)),
            'max_h': float(round(np.max(tc_base_valid) / 3600, 2)),
            'mean_h': float(round(np.mean(tc_base_valid) / 3600, 2)),
        },
        'tc_manning': {
            'min_h': float(round(np.min(tc_valid) / 3600, 2)),
            'max_h': float(round(np.max(tc_valid) / 3600, 2)),
            'mean_h': float(round(np.mean(tc_valid) / 3600, 2)),
        },
        'adjustment': {
            'mean_factor': float(round(np.mean(adjustment_factor), 3)),
            'diff_h': float(round((np.mean(tc_valid) - np.mean(tc_base_valid)) / 3600, 2)),
            'diff_pct': float(round((np.mean(tc_valid) / np.mean(tc_base_valid) - 1) * 100, 1)),
        }
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Calculate Tc with Manning adjustment")
    parser.add_argument("--year", type=int, default=2022,
                        help="Ano do LULC para usar (default: 2022)")
    parser.add_argument("--tc-base", type=str, default=str(TC_BASE_FILE),
                        help="Arquivo Tc base (segundos)")
    args = parser.parse_args()

    print_header("TTD-SCS-LSTM - Calculo Tc Manning")

    # Verificar arquivos
    tc_base_file = Path(args.tc_base)
    lulc_file = LULC_DIR / f"lulc_{args.year}.tif"
    output_file = OUTPUT_DIR / f"tc_manning_{args.year}.tif"

    if not tc_base_file.exists():
        print(f"\n  [ERRO] Tc base nao encontrado: {tc_base_file}")
        return 1

    if not lulc_file.exists():
        print(f"\n  [ERRO] LULC nao encontrado: {lulc_file}")
        print(f"  Execute primeiro: python scripts/process_lulc.py --year {args.year}")
        return 1

    # Calcular Tc Manning
    print_header(f"Calculo Tc Manning - LULC {args.year}")

    stats = calculate_tc_manning(tc_base_file, lulc_file, output_file)

    # Salvar estatisticas
    stats_file = OUTPUT_DIR / f"estatisticas_tc_manning_{args.year}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Estatisticas: {stats_file}")

    # Copiar Tc base para processed (para comparacao)
    tc_base_output = OUTPUT_DIR / "tc_base.tif"
    if not tc_base_output.exists():
        import shutil
        shutil.copy(tc_base_file, tc_base_output)
        print(f"  Tc base copiado: {tc_base_output}")

    # Summary
    print_header("RESUMO")
    print(f"\n  Arquivos gerados:")
    print(f"    - {output_file}")
    print(f"    - {stats_file}")

    print(f"\n  Coeficientes de Manning utilizados:")
    for class_id, n_value in MANNING_N.items():
        ratio = n_value / N_REF
        effect = "+" if ratio > 1 else ""
        print(f"    {CLASS_NAMES[class_id]}: n={n_value:.3f} (Tc {effect}{(ratio-1)*100:.0f}%)")

    print(f"\n  Resultado:")
    print(f"    Tc_base mean:    {stats['tc_base']['mean_h']:.2f} horas")
    print(f"    Tc_manning mean: {stats['tc_manning']['mean_h']:.2f} horas")
    print(f"    Diferenca:       {stats['adjustment']['diff_h']:+.2f} horas ({stats['adjustment']['diff_pct']:+.1f}%)")

    print(f"\n  [OK] Calculo concluido!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
