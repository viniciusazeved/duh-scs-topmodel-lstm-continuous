#!/usr/bin/env python
"""
Agregar precipitação MERGE por ottobacia.

Este script calcula a precipitação média horária para cada ottobacia
a partir dos arquivos MERGE (grade 0.1°).

Processo:
1. Calcular pesos (fração de cada pixel MERGE em cada ottobacia)
2. Para cada hora, aplicar pesos e calcular média ponderada
3. Salvar em HDF5 para uso no modelo

Usage:
    python scripts/aggregate_merge_ottobacias.py --start 2021-01-01 --end 2025-12-31
    python scripts/aggregate_merge_ottobacias.py --year 2021
    python scripts/aggregate_merge_ottobacias.py --compute-weights-only
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import h5py
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuracoes
MERGE_DIR = Path("_archive/data/series/merge/raw")  # Onde estao os arquivos MERGE
OTTO_FILE = Path("data/raw/ottobacias/ottobacias_cn_2022.gpkg")  # Ottobacias CN 2022 (245)
OUTPUT_DIR = Path("data/processed")
WEIGHTS_FILE = OUTPUT_DIR / "merge_weights.npz"
OUTPUT_FILE = OUTPUT_DIR / "precipitation_distributed.h5"

# MERGE grid
MERGE_RES = 0.1  # graus


def get_merge_path(dt: datetime) -> Path:
    """Retorna caminho do arquivo MERGE para uma data/hora."""
    year = dt.strftime("%Y")
    filename = f"MERGE_CPTEC_{dt.strftime('%Y%m%d%H')}.grib2"
    return MERGE_DIR / year / filename


def load_merge_sample(merge_dir: Path) -> tuple:
    """Carrega um arquivo MERGE de exemplo para obter a grade."""
    # Encontrar um arquivo
    sample_files = list(merge_dir.rglob("MERGE_CPTEC_2021*.grib2"))
    if not sample_files:
        sample_files = list(merge_dir.rglob("MERGE_CPTEC_*.grib2"))

    if not sample_files:
        raise FileNotFoundError(f"Nenhum arquivo MERGE encontrado em {merge_dir}")

    ds = xr.open_dataset(sample_files[0], engine='cfgrib')

    # Coordenadas
    lats = ds.latitude.values
    lons = ds.longitude.values

    # Converter longitude de 0-360 para -180-180
    lons_180 = np.where(lons > 180, lons - 360, lons)

    ds.close()

    return lats, lons, lons_180


def compute_weights(otto: gpd.GeoDataFrame, lats: np.ndarray, lons_180: np.ndarray) -> dict:
    """
    Calcula pesos de cada pixel MERGE para cada ottobacia.

    Retorna dicionario com:
    - 'otto_ids': lista de IDs das ottobacias
    - 'pixel_indices': lista de arrays com indices (i,j) dos pixels para cada otto
    - 'pixel_weights': lista de arrays com pesos de cada pixel
    - 'lat_indices': indices de latitude usados
    - 'lon_indices': indices de longitude usados
    """
    print("Calculando pesos MERGE -> Ottobacias...")

    # Reprojetar ottobacias para WGS84
    otto_wgs = otto.to_crs("EPSG:4326")

    # Bounding box da bacia
    bounds = otto_wgs.total_bounds  # minx, miny, maxx, maxy

    # Indices da grade MERGE que cobrem a bacia (com margem)
    margin = 0.2  # graus
    lat_mask = (lats >= bounds[1] - margin) & (lats <= bounds[3] + margin)
    lon_mask = (lons_180 >= bounds[0] - margin) & (lons_180 <= bounds[2] + margin)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    lats_sub = lats[lat_indices]
    lons_sub = lons_180[lon_indices]

    print(f"  Grade MERGE recortada: {len(lat_indices)} x {len(lon_indices)} pixels")

    # Criar grid de pixels MERGE como poligonos
    from shapely.geometry import box

    pixel_polygons = []
    pixel_coords = []

    for i, lat in enumerate(lats_sub):
        for j, lon in enumerate(lons_sub):
            # Pixel MERGE centrado em (lat, lon)
            poly = box(
                lon - MERGE_RES/2,
                lat - MERGE_RES/2,
                lon + MERGE_RES/2,
                lat + MERGE_RES/2
            )
            pixel_polygons.append(poly)
            pixel_coords.append((lat_indices[i], lon_indices[j]))

    pixels_gdf = gpd.GeoDataFrame(
        {'pixel_idx': range(len(pixel_polygons)), 'coords': pixel_coords},
        geometry=pixel_polygons,
        crs="EPSG:4326"
    )

    print(f"  Pixels MERGE na regiao: {len(pixels_gdf)}")

    # Calcular intersecao de cada ottobacia com pixels MERGE
    otto_ids = []
    pixel_indices = []
    pixel_weights = []

    for idx, row in tqdm(otto_wgs.iterrows(), total=len(otto_wgs), desc="  Calculando pesos"):
        otto_geom = row.geometry
        otto_id = row.get('cobacia', idx)
        otto_area = otto_geom.area

        if otto_area == 0:
            continue

        # Encontrar pixels que intersectam esta ottobacia
        intersects = pixels_gdf[pixels_gdf.intersects(otto_geom)]

        if len(intersects) == 0:
            # Ottobacia muito pequena, usar pixel mais proximo
            centroid = otto_geom.centroid
            distances = pixels_gdf.geometry.distance(centroid)
            nearest_idx = distances.idxmin()

            otto_ids.append(otto_id)
            pixel_indices.append(np.array([pixels_gdf.loc[nearest_idx, 'coords']]))
            pixel_weights.append(np.array([1.0]))
            continue

        # Calcular area de intersecao com cada pixel
        coords_list = []
        weights_list = []

        for _, pixel_row in intersects.iterrows():
            intersection = otto_geom.intersection(pixel_row.geometry)
            inter_area = intersection.area

            if inter_area > 0:
                coords_list.append(pixel_row['coords'])
                weights_list.append(inter_area)

        if coords_list:
            weights_arr = np.array(weights_list)
            weights_arr = weights_arr / weights_arr.sum()  # Normalizar

            otto_ids.append(otto_id)
            pixel_indices.append(np.array(coords_list))
            pixel_weights.append(weights_arr)

    print(f"  Ottobacias processadas: {len(otto_ids)}")

    return {
        'otto_ids': otto_ids,
        'pixel_indices': pixel_indices,
        'pixel_weights': pixel_weights,
        'lat_indices': lat_indices,
        'lon_indices': lon_indices,
        'lats': lats,
        'lons_180': lons_180
    }


def save_weights(weights: dict, filepath: Path):
    """Salva pesos em arquivo NPZ."""
    # Converter listas para arrays de objetos
    np.savez_compressed(
        filepath,
        otto_ids=np.array(weights['otto_ids'], dtype=object),
        pixel_indices=np.array(weights['pixel_indices'], dtype=object),
        pixel_weights=np.array(weights['pixel_weights'], dtype=object),
        lat_indices=weights['lat_indices'],
        lon_indices=weights['lon_indices'],
        lats=weights['lats'],
        lons_180=weights['lons_180']
    )
    print(f"Pesos salvos em: {filepath}")


def load_weights(filepath: Path) -> dict:
    """Carrega pesos de arquivo NPZ."""
    data = np.load(filepath, allow_pickle=True)
    return {
        'otto_ids': data['otto_ids'].tolist(),
        'pixel_indices': data['pixel_indices'].tolist(),
        'pixel_weights': data['pixel_weights'].tolist(),
        'lat_indices': data['lat_indices'],
        'lon_indices': data['lon_indices'],
        'lats': data['lats'],
        'lons_180': data['lons_180']
    }


def read_merge_precip(filepath: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Le precipitacao de um arquivo MERGE."""
    try:
        ds = xr.open_dataset(filepath, engine='cfgrib')

        # Variavel de precipitacao (pode ser 'rdp' ou 'prec' ou 'tp')
        if 'rdp' in ds.data_vars:
            precip = ds['rdp'].values
        elif 'prec' in ds.data_vars:
            precip = ds['prec'].values
        elif 'tp' in ds.data_vars:
            precip = ds['tp'].values
        else:
            # Pegar primeira variavel
            var_name = list(ds.data_vars)[0]
            precip = ds[var_name].values

        ds.close()

        # Garantir que nao ha valores negativos
        precip = np.maximum(precip, 0)

        return precip

    except Exception as e:
        return None


def aggregate_precipitation(
    weights: dict,
    start: datetime,
    end: datetime,
    merge_dir: Path
) -> tuple:
    """
    Agrega precipitacao MERGE por ottobacia.

    Retorna:
    - timestamps: array de datetime
    - precip_otto: array (n_times, n_ottobacias) com precipitacao
    """
    otto_ids = weights['otto_ids']
    pixel_indices = weights['pixel_indices']
    pixel_weights = weights['pixel_weights']

    n_ottobacias = len(otto_ids)

    # Gerar lista de horas
    hours = []
    current = start
    while current <= end:
        hours.append(current)
        current += timedelta(hours=1)

    n_hours = len(hours)
    print(f"Agregando {n_hours:,} horas para {n_ottobacias:,} ottobacias...")

    # Arrays de saida
    timestamps = np.array(hours, dtype='datetime64[ns]')
    precip_otto = np.full((n_hours, n_ottobacias), np.nan, dtype=np.float32)

    # Processar cada hora
    missing = 0
    for t, dt in enumerate(tqdm(hours, desc="Processando")):
        filepath = get_merge_path(dt)

        if not filepath.exists():
            missing += 1
            continue

        # Ler precipitacao
        precip = read_merge_precip(filepath, weights['lats'], weights['lons_180'])

        if precip is None:
            missing += 1
            continue

        # Agregar para cada ottobacia
        for i, (pix_idx, pix_w) in enumerate(zip(pixel_indices, pixel_weights)):
            values = []
            for (lat_i, lon_i), w in zip(pix_idx, pix_w):
                val = precip[lat_i, lon_i]
                if not np.isnan(val):
                    values.append(val * w)

            if values:
                precip_otto[t, i] = np.sum(values)

    print(f"Arquivos faltantes: {missing} ({100*missing/n_hours:.1f}%)")

    return timestamps, precip_otto, otto_ids


def save_precipitation_h5(
    timestamps: np.ndarray,
    precip_otto: np.ndarray,
    otto_ids: list,
    filepath: Path
):
    """Salva precipitacao agregada em HDF5."""
    with h5py.File(filepath, 'w') as f:
        # Timestamps
        f.create_dataset('timestamps', data=timestamps.astype('int64'))

        # Precipitacao (n_times, n_ottobacias)
        f.create_dataset('precipitation', data=precip_otto, compression='gzip')

        # IDs das ottobacias
        otto_ids_encoded = [str(x).encode('utf-8') for x in otto_ids]
        f.create_dataset('otto_ids', data=otto_ids_encoded)

        # Metadata
        f.attrs['n_times'] = len(timestamps)
        f.attrs['n_ottobacias'] = len(otto_ids)
        f.attrs['start'] = str(timestamps[0])
        f.attrs['end'] = str(timestamps[-1])
        f.attrs['unit'] = 'mm/h'

    print(f"Precipitacao salva em: {filepath}")
    print(f"  Shape: {precip_otto.shape}")
    print(f"  Tamanho: {filepath.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Agregar MERGE por ottobacia")
    parser.add_argument("--start", type=str, help="Data inicio (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Data fim (YYYY-MM-DD)")
    parser.add_argument("--year", type=int, help="Processar ano inteiro")
    parser.add_argument("--compute-weights-only", action="store_true",
                        help="Apenas calcular pesos, nao agregar")
    parser.add_argument("--otto-file", type=str, default=str(OTTO_FILE),
                        help="Arquivo de ottobacias")
    parser.add_argument("--merge-dir", type=str, default=str(MERGE_DIR),
                        help="Diretorio dos arquivos MERGE")

    args = parser.parse_args()

    # Configurar paths
    otto_file = Path(args.otto_file)
    merge_dir = Path(args.merge_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AGREGACAO MERGE -> OTTOBACIAS")
    print("=" * 60)

    # Carregar ottobacias
    print(f"\nCarregando ottobacias de {otto_file}...")
    if not otto_file.exists():
        print(f"[ERRO] Arquivo nao encontrado: {otto_file}")
        print("Execute primeiro o notebook 02 para gerar o arquivo.")
        sys.exit(1)

    otto = gpd.read_file(otto_file)
    print(f"  Ottobacias: {len(otto)}")

    # Carregar grade MERGE
    print(f"\nCarregando grade MERGE de {merge_dir}...")
    lats, lons, lons_180 = load_merge_sample(merge_dir)
    print(f"  Grade: {len(lats)} x {len(lons)}")

    # Calcular ou carregar pesos
    if WEIGHTS_FILE.exists() and not args.compute_weights_only:
        print(f"\nCarregando pesos de {WEIGHTS_FILE}...")
        weights = load_weights(WEIGHTS_FILE)
        print(f"  Ottobacias nos pesos: {len(weights['otto_ids'])}")
    else:
        print("\nCalculando pesos...")
        weights = compute_weights(otto, lats, lons_180)
        save_weights(weights, WEIGHTS_FILE)

    if args.compute_weights_only:
        print("\n[OK] Pesos calculados. Use sem --compute-weights-only para agregar.")
        return

    # Determinar periodo
    if args.year:
        start = datetime(args.year, 1, 1, 0)
        end = datetime(args.year, 12, 31, 23)
    elif args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23)
    else:
        print("[ERRO] Especifique --start/--end ou --year")
        sys.exit(1)

    print(f"\nPeriodo: {start} a {end}")

    # Agregar
    timestamps, precip_otto, otto_ids = aggregate_precipitation(
        weights, start, end, merge_dir
    )

    # Salvar
    save_precipitation_h5(timestamps, precip_otto, otto_ids, OUTPUT_FILE)

    # Estatisticas
    print("\n" + "=" * 60)
    print("ESTATISTICAS")
    print("=" * 60)
    valid = ~np.isnan(precip_otto)
    print(f"Valores validos: {valid.sum():,} ({100*valid.mean():.1f}%)")
    print(f"Precip media: {np.nanmean(precip_otto):.2f} mm/h")
    print(f"Precip maxima: {np.nanmax(precip_otto):.1f} mm/h")
    print(f"Horas com chuva (>0.1mm): {(np.nanmean(precip_otto, axis=1) > 0.1).sum():,}")

    print("\n[OK] Agregacao concluida!")


if __name__ == "__main__":
    main()
