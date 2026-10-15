#!/usr/bin/env python
"""
Download MapBiomas LULC via Google Earth Engine.

Download de dados de uso e cobertura do solo (LULC) do MapBiomas
Collection 9 via Google Earth Engine.

Para este projeto (TTD-SCS-LSTM), baixamos apenas 2022 (ano base).

Uso:
    python scripts/download_lulc_mapbiomas.py
    python scripts/download_lulc_mapbiomas.py --year 2022
"""

import os
import sys
from pathlib import Path

# Fix PROJ conflict with PostgreSQL
import sysconfig
site_packages = sysconfig.get_path('purelib')
proj_dir = Path(site_packages) / "pyproj" / "proj_dir" / "share" / "proj"
if proj_dir.exists():
    os.environ["PROJ_LIB"] = str(proj_dir)
    os.environ["PROJ_DATA"] = str(proj_dir)
else:
    proj_dir2 = Path(sys.prefix) / "Library" / "share" / "proj"
    if proj_dir2.exists():
        os.environ["PROJ_LIB"] = str(proj_dir2)
        os.environ["PROJ_DATA"] = str(proj_dir2)

import argparse
import ee
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS as RioCRS
import urllib.request

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
BACIA_FILE = DATA_DIR / "raw" / "bacia" / "bacia_manuel_duarte.shp"
LULC_DIR = DATA_DIR / "raw" / "lulc"
LULC_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
RESOLUTION = 30  # metros
BUFFER_M = 500  # buffer para download
CRS_UTM = "EPSG:31983"
CRS_WGS84 = "EPSG:4326"

# MapBiomas Collection 9
MAPBIOMAS_ASSET = "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1"

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


def print_header(title: str) -> None:
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def initialize_gee() -> None:
    """Initialize Google Earth Engine."""
    print("\nInicializando Google Earth Engine...")
    try:
        ee.Initialize()
        print("  [OK] GEE inicializado")
    except Exception:
        print("  Autenticando GEE...")
        ee.Authenticate()
        ee.Initialize()
        print("  [OK] GEE autenticado e inicializado")


def get_bbox_wgs84(bacia: gpd.GeoDataFrame) -> list:
    """Get bounding box in WGS84 with buffer."""
    bacia_wgs84 = bacia.to_crs(CRS_WGS84)
    bounds = bacia_wgs84.total_bounds  # minx, miny, maxx, maxy

    buffer_deg = BUFFER_M / 111000
    bbox = [
        bounds[0] - buffer_deg,
        bounds[1] - buffer_deg,
        bounds[2] + buffer_deg,
        bounds[3] + buffer_deg
    ]
    return bbox


def download_year(year: int, region: ee.Geometry, output_dir: Path) -> Path:
    """Download MapBiomas LULC for a specific year."""
    output_file = output_dir / f"lulc_mapbiomas_{year}.tif"

    if output_file.exists():
        print(f"  [SKIP] {year} - arquivo ja existe")
        return output_file

    print(f"  Baixando {year}...", end=" ", flush=True)

    try:
        # Load MapBiomas
        mapbiomas = ee.Image(MAPBIOMAS_ASSET)

        # Select band for year
        band_name = f"classification_{year}"
        lulc_year = mapbiomas.select(band_name).clip(region)

        # Get download URL
        url = lulc_year.getDownloadURL({
            'scale': RESOLUTION,
            'crs': CRS_UTM,
            'region': region,
            'format': 'GEO_TIFF'
        })

        # Download
        temp_file = output_dir / f"temp_{year}.tif"
        urllib.request.urlretrieve(url, temp_file)

        # Fix CRS and save
        with rasterio.open(temp_file) as src:
            data = src.read(1)
            profile = src.profile.copy()

        # Update profile with correct CRS
        crs_obj = RioCRS.from_wkt(CRS_UTM_WKT)
        profile.update(
            crs=crs_obj,
            dtype='uint8',
            nodata=0,
            driver='GTiff',
            tiled=True,
            compress='lzw'
        )

        # Save with correct CRS
        data = data.astype(np.uint8)
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(data, 1)

        # Clean up
        temp_file.unlink()

        print(f"OK ({output_file.stat().st_size / 1024:.0f} KB)")
        return output_file

    except Exception as e:
        print(f"ERRO: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download MapBiomas LULC")
    parser.add_argument("--year", type=int, default=2022,
                        help="Ano para download (default: 2022)")
    parser.add_argument("--bacia", type=str, default=str(BACIA_FILE),
                        help="Arquivo shapefile da bacia")
    args = parser.parse_args()

    print_header("TTD-SCS-LSTM - Download MapBiomas LULC")

    # Initialize GEE
    initialize_gee()

    # Load basin
    print("\nCarregando bacia...")
    bacia_file = Path(args.bacia)
    if not bacia_file.exists():
        print(f"  [ERRO] Arquivo nao encontrado: {bacia_file}")
        return 1

    bacia = gpd.read_file(bacia_file)

    # Garantir que esta em UTM para calculo de area
    if bacia.crs and bacia.crs.is_geographic:
        bacia_utm = bacia.to_crs(CRS_UTM)
    else:
        bacia_utm = bacia

    area_km2 = bacia_utm.geometry.area.sum() / 1e6
    print(f"  Area: {area_km2:,.0f} km2")

    # Get bounding box
    bbox = get_bbox_wgs84(bacia)
    print(f"  Bbox (WGS84): [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")

    # Create GEE geometry
    region = ee.Geometry.Rectangle(bbox)

    # Download
    print_header(f"Download MapBiomas Collection 9 - Ano {args.year}")
    print(f"\n  Asset: {MAPBIOMAS_ASSET}")
    print(f"  Resolucao: {RESOLUTION}m")
    print(f"  CRS: {CRS_UTM}")
    print()

    result = download_year(args.year, region, LULC_DIR)

    # Summary
    print_header("RESUMO")
    print(f"\n  Diretorio: {LULC_DIR}")

    if result:
        print(f"  Arquivo: {result.name}")
        print(f"  Tamanho: {result.stat().st_size / 1024:.0f} KB")
        print(f"\n  [OK] Download concluido!")
        print("\n  >>> Proximo passo: python scripts/process_lulc.py <<<")
        return 0
    else:
        print(f"\n  [ERRO] Falha no download")
        return 1


if __name__ == "__main__":
    sys.exit(main())
