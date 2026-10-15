#!/usr/bin/env python
"""
Processamento LULC MapBiomas para TTD-SCS-LSTM.

1. Reclassificacao para 7 classes agregadas
2. Clip ao limite da bacia
3. Calculo de coeficientes de Manning por ottobacia
4. Exportacao de estatisticas

Classes Agregadas:
1 - Floresta           (Manning n = 0.150)
2 - Vegetacao Natural  (Manning n = 0.100)
3 - Agropecuaria       (Manning n = 0.035)
4 - Area Nao Vegetada  (Manning n = 0.025)
5 - Agua               (Manning n = 0.030)
6 - Nao Observado      (Manning n = 0.035)
7 - Area Urbana        (Manning n = 0.015)

Uso:
    python scripts/process_lulc.py
    python scripts/process_lulc.py --year 2022
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
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.crs import CRS as RioCRS
from rasterstats import zonal_stats

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MAPBIOMAS_DIR = DATA_DIR / "raw" / "lulc"
LULC_DIR = DATA_DIR / "processed" / "lulc"
LULC_DIR.mkdir(parents=True, exist_ok=True)

# Input files
BACIA_FILE = DATA_DIR / "raw" / "bacia" / "bacia_manuel_duarte.shp"
OTTO_FILE = DATA_DIR / "raw" / "ottobacias" / "ottobacias_cn.gpkg"

# Parameters
RESOLUTION = 30
CRS_UTM = "EPSG:31983"

# WKT for CRS
CRS_UTM_WKT = '''PROJCS["SIRGAS 2000 / UTM zone 23S",
    GEOGCS["SIRGAS 2000",
        DATUM["Sistema_de_Referencia_Geocentrico_para_las_AmericaS_2000",
            SPHEROID["GRS 1980",6378137,298.257222101]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",-45],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",10000000],
    UNIT["metre",1],
    AUTHORITY["EPSG","31983"]]'''

# ============================================================
# TABELA DE RECLASSIFICACAO MapBiomas -> 7 Classes
# ============================================================
# Baseado no MapBiomas Collection 9

RECLASS_MAP = {
    # 1 - Floresta
    1: 1,   # Floresta
    3: 1,   # Formacao Florestal
    4: 1,   # Formacao Savanica
    5: 1,   # Mangue
    6: 1,   # Floresta Alagavel
    49: 1,  # Restinga Arborizada

    # 2 - Vegetacao Natural Nao Florestal
    10: 2,  # Formacao Natural nao Florestal
    11: 2,  # Campo Alagado e Area Pantanosa
    12: 2,  # Formacao Campestre
    13: 2,  # Outras Formacoes nao Florestais
    32: 2,  # Apicum

    # 3 - Agropecuaria
    14: 3,  # Agropecuaria
    15: 3,  # Pastagem
    18: 3,  # Agricultura
    19: 3,  # Lavoura Temporaria
    20: 3,  # Cana
    21: 3,  # Mosaico Agricultura/Pastagem
    36: 3,  # Lavoura Perene
    39: 3,  # Soja
    40: 3,  # Arroz
    41: 3,  # Outras Lavouras Temporarias
    46: 3,  # Cafe
    47: 3,  # Citrus
    48: 3,  # Outras Lavouras Perenes
    62: 3,  # Algodao
    35: 3,  # Dende (Palma de Oleo)

    # 4 - Area Nao Vegetada
    22: 4,  # Area nao Vegetada
    23: 4,  # Praia e Duna
    25: 4,  # Outra Area nao Vegetada
    29: 4,  # Afloramento Rochoso
    30: 4,  # Mineracao

    # 5 - Agua
    26: 5,  # Corpo Dagua
    33: 5,  # Rio, Lago e Oceano

    # 6 - Nao Observado
    27: 6,  # Nao Observado

    # 7 - Area Urbana
    24: 7,  # Infraestrutura Urbana
}

# Nomes das classes agregadas
CLASS_NAMES = {
    1: "Floresta",
    2: "Vegetacao Natural",
    3: "Agropecuaria",
    4: "Nao Vegetada",
    5: "Agua",
    6: "Nao Observado",
    7: "Urbano"
}

# Coeficientes de Manning por classe
# Referencia: Chow (1959), Engman (1986)
MANNING_N = {
    1: 0.150,  # Floresta - alta rugosidade (serrapilheira, vegetacao densa)
    2: 0.100,  # Vegetacao Natural - rugosidade media-alta
    3: 0.035,  # Agropecuaria - pastagem/cultivo (rugosidade baixa-media)
    4: 0.025,  # Nao Vegetada - solo exposto (baixa rugosidade)
    5: 0.030,  # Agua - canal natural
    6: 0.035,  # Nao Observado - valor medio
    7: 0.015,  # Urbano - superficies impermeaveis (muito baixa)
}

# Cores para visualizacao (hex)
CLASS_COLORS = {
    1: "#1f8d49",  # Verde escuro - Floresta
    2: "#d6bc74",  # Bege - Veg Natural
    3: "#FFFFB2",  # Amarelo claro - Agropecuaria
    4: "#db4d4f",  # Vermelho - Nao Vegetada
    5: "#0000FF",  # Azul - Agua
    6: "#808080",  # Cinza - Nao Observado
    7: "#d4271e",  # Vermelho escuro - Urbano
}


def print_header(title: str) -> None:
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def reclassify_raster(data: np.ndarray) -> np.ndarray:
    """Reclassify MapBiomas codes to 7 aggregated classes."""
    result = np.zeros_like(data, dtype=np.uint8)

    for original, new in RECLASS_MAP.items():
        result[data == original] = new

    return result


def process_lulc(year: int, bacia: gpd.GeoDataFrame) -> tuple:
    """Process LULC for a single year, returns (output_file, stats)."""
    input_file = MAPBIOMAS_DIR / f"lulc_mapbiomas_{year}.tif"
    output_file = LULC_DIR / f"lulc_{year}.tif"

    if not input_file.exists():
        print(f"  [ERRO] Arquivo nao encontrado: {input_file}")
        return None, None

    # Check if already processed
    if output_file.exists():
        print(f"  [OK] LULC {year} ja processado")
        with rasterio.open(output_file) as src:
            data = src.read(1)
            valid = data[data > 0]

        stats = {}
        for class_id in range(1, 8):
            count = np.sum(valid == class_id)
            area_km2 = count * (RESOLUTION ** 2) / 1e6
            stats[class_id] = {
                'pixels': int(count),
                'area_km2': round(area_km2, 2),
                'percent': round(count / len(valid) * 100, 2) if len(valid) > 0 else 0
            }
        return output_file, stats

    print(f"  Processando LULC {year}...", end=" ", flush=True)

    try:
        # Garantir que bacia esta em UTM
        if bacia.crs and bacia.crs.is_geographic:
            bacia_utm = bacia.to_crs(CRS_UTM)
        else:
            bacia_utm = bacia

        # Read MapBiomas raster
        with rasterio.open(input_file) as src:
            # Clip to basin
            out_image, out_transform = mask(
                src,
                bacia_utm.geometry,
                crop=True,
                nodata=0
            )
            data = out_image[0]

            # Get profile
            profile = src.profile.copy()
            profile.update(
                height=data.shape[0],
                width=data.shape[1],
                transform=out_transform,
                dtype='uint8',
                nodata=0,
                compress='lzw',
                tiled=True
            )

        # Reclassify to 7 classes
        data_reclass = reclassify_raster(data)

        # Fix CRS to proper EPSG:31983
        crs_obj = RioCRS.from_wkt(CRS_UTM_WKT)
        profile['crs'] = crs_obj

        # Save
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(data_reclass, 1)

        # Calculate statistics
        valid = data_reclass[data_reclass > 0]
        stats = {}
        for class_id in range(1, 8):
            count = np.sum(valid == class_id)
            area_km2 = count * (RESOLUTION ** 2) / 1e6
            stats[class_id] = {
                'pixels': int(count),
                'area_km2': round(area_km2, 2),
                'percent': round(count / len(valid) * 100, 2) if len(valid) > 0 else 0
            }

        print(f"OK ({output_file.stat().st_size / 1024:.0f} KB)")
        return output_file, stats

    except Exception as e:
        print(f"ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def calculate_manning_by_ottobacia(lulc_file: Path, otto: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate weighted Manning coefficient for each ottobacia."""
    print("\n  Calculando Manning por ottobacia...")

    # Garantir que otto esta em UTM
    if otto.crs and otto.crs.is_geographic:
        otto_utm = otto.to_crs(CRS_UTM)
    else:
        otto_utm = otto.copy()

    # Calculate zonal histogram (count of each class per ottobacia)
    with rasterio.open(lulc_file) as src:
        affine = src.transform
        data = src.read(1)

    # Calculate zonal statistics for each class
    class_counts = {}
    for class_id in range(1, 8):
        # Create binary mask for this class
        class_data = (data == class_id).astype(np.float32)

        stats = zonal_stats(
            otto_utm.geometry,
            class_data,
            affine=affine,
            stats=['sum'],
            nodata=0
        )

        class_counts[class_id] = [s['sum'] if s['sum'] else 0 for s in stats]

    # Calculate weighted Manning coefficient for each ottobacia
    manning_values = []
    dominant_class = []

    for i in range(len(otto_utm)):
        total_pixels = sum(class_counts[c][i] for c in range(1, 8))

        if total_pixels == 0:
            # No data - use default value
            manning_values.append(0.035)
            dominant_class.append(3)  # Agropecuaria
            continue

        # Weighted average of Manning coefficients
        weighted_sum = 0
        max_count = 0
        dom_class = 3

        for class_id in range(1, 8):
            count = class_counts[class_id][i]
            weight = count / total_pixels
            weighted_sum += weight * MANNING_N[class_id]

            if count > max_count:
                max_count = count
                dom_class = class_id

        manning_values.append(round(weighted_sum, 4))
        dominant_class.append(dom_class)

    # Add to GeoDataFrame
    otto_utm['manning_n'] = manning_values
    otto_utm['lulc_class'] = dominant_class
    otto_utm['lulc_name'] = [CLASS_NAMES[c] for c in dominant_class]

    # Calculate fractions for each class
    for class_id in range(1, 8):
        total_pixels = sum(class_counts[c][i] for c in range(1, 8) for i in range(len(otto_utm)))
        otto_utm[f'frac_c{class_id}'] = [
            round(class_counts[class_id][i] / sum(class_counts[c][i] for c in range(1, 8)), 4)
            if sum(class_counts[c][i] for c in range(1, 8)) > 0 else 0
            for i in range(len(otto_utm))
        ]

    print(f"    Manning range: {min(manning_values):.4f} - {max(manning_values):.4f}")
    print(f"    Manning mean:  {np.mean(manning_values):.4f}")

    return otto_utm


def main():
    parser = argparse.ArgumentParser(description="Process MapBiomas LULC")
    parser.add_argument("--year", type=int, default=2022,
                        help="Ano para processar (default: 2022)")
    args = parser.parse_args()

    print_header("TTD-SCS-LSTM - Processamento LULC")

    # Load basin
    print("\nCarregando bacia...")
    if not BACIA_FILE.exists():
        print(f"  [ERRO] Arquivo nao encontrado: {BACIA_FILE}")
        return 1

    bacia = gpd.read_file(BACIA_FILE)
    if bacia.crs and bacia.crs.is_geographic:
        bacia_utm = bacia.to_crs(CRS_UTM)
    else:
        bacia_utm = bacia

    area_km2 = bacia_utm.geometry.area.sum() / 1e6
    print(f"  Area: {area_km2:,.0f} km2")

    # Process LULC
    print_header(f"Processamento LULC {args.year}")
    lulc_file, stats = process_lulc(args.year, bacia)

    if lulc_file is None:
        print("\n  [ERRO] Falha no processamento")
        return 1

    # Print statistics
    print("\n  Estatisticas por classe:")
    for class_id, name in CLASS_NAMES.items():
        s = stats[class_id]
        print(f"    {name}: {s['area_km2']:.1f} km2 ({s['percent']:.1f}%)")

    # Calculate Manning by ottobacia if file exists
    if OTTO_FILE.exists():
        print_header("Calculo de Manning por Ottobacia")
        otto = gpd.read_file(OTTO_FILE)
        print(f"  Ottobacias: {len(otto)}")

        otto_manning = calculate_manning_by_ottobacia(lulc_file, otto)

        # Save
        output_otto = DATA_DIR / "processed" / "ottobacias_manning.gpkg"
        otto_manning.to_file(output_otto, driver="GPKG")
        print(f"\n  Salvo: {output_otto}")

        # Also save Manning values as CSV for easy access
        manning_csv = LULC_DIR / f"manning_by_otto_{args.year}.csv"
        otto_cols = ['cobacia'] if 'cobacia' in otto_manning.columns else ['COBACIA']
        otto_cols = otto_cols[0] if otto_cols[0] in otto_manning.columns else otto_manning.columns[0]

        manning_df = otto_manning[[otto_cols, 'manning_n', 'lulc_class', 'lulc_name']].copy()
        manning_df = manning_df.rename(columns={otto_cols: 'cobacia'})
        manning_df.to_csv(manning_csv, index=False)
        print(f"  CSV: {manning_csv}")
    else:
        print(f"\n  [AVISO] Arquivo de ottobacias nao encontrado: {OTTO_FILE}")
        print("  Manning por ottobacia nao calculado.")

    # Save statistics
    stats_file = LULC_DIR / f"estatisticas_lulc_{args.year}.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'ano': args.year,
            'classes': CLASS_NAMES,
            'manning': MANNING_N,
            'estatisticas': stats
        }, f, indent=2)
    print(f"\n  Estatisticas: {stats_file}")

    # Summary
    print_header("RESUMO")
    print(f"\n  LULC processado: {lulc_file}")
    print(f"  Classes agregadas: 7")

    print("\n  Coeficientes de Manning:")
    for class_id, name in CLASS_NAMES.items():
        print(f"    {name}: n = {MANNING_N[class_id]:.3f}")

    print(f"\n  [OK] Processamento concluido!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
