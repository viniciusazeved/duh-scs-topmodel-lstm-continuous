"""Computação offline de caminhos de fluxo D8 e distâncias por classe LULC.

Pipeline:
1. DEM 30m (ANADEM) → fill depressions → D8 flow direction → slope
2. Definir outlet (coordenadas da estação Manuel Duarte)
3. Para cada célula, traçar caminho D8 até o outlet
4. Para cada classe LULC k, computar D_ck = Σ dx · s^(-1/2) ao longo do caminho,
   considerando apenas segmentos em que LULC == k

Saídas:
- travel_distance.tif  (distância euclidiana total acumulada até outlet, m)
- slope.tif            (slope por célula, m/m)
- D_ck_{k}.tif         (um raster por classe LULC, unidade s·m^(-1/2))

O tensor final D_ck por célula é empilhado em parquet para treino:
cells_30m.parquet: colunas (ottobacia_id, cell_id, area_m2, D_c1, ..., D_cK).

Uso do numba para downslope accumulation (3.5M células → segundos).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from numba import njit, prange

logger = logging.getLogger(__name__)

# Offsets D8 da convenção WhiteboxTools (1,2,4,8,16,32,64,128)
# Direções em ordem: E, SE, S, SW, W, NW, N, NE
D8_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int32)
D8_DR = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
D8_DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
D8_DIST = np.array(
    [1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)], dtype=np.float64
)


def preparar_terreno(
    dem_path: Path,
    saida_dir: Path,
) -> dict[str, Path]:
    """Fill depressions + D8 pointer + slope com WhiteboxTools.

    Retorna dict com paths absolutos: dem_filled, d8_pointer, slope.

    No Windows, WBT funciona melhor com paths relativos quando set_working_dir
    está ativo. Aqui copiamos o DEM para o working_dir e usamos só nomes simples.
    """
    import shutil

    import whitebox

    saida_dir = saida_dir.resolve()
    saida_dir.mkdir(parents=True, exist_ok=True)

    # Copia DEM para o working_dir (WBT no Windows tropeça em paths absolutos)
    dem_local = saida_dir / "dem_input.tif"
    if dem_local.resolve() != dem_path.resolve():
        shutil.copy2(dem_path, dem_local)

    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(str(saida_dir))
    wbt.set_verbose_mode(True)

    dem_filled_name = "dem_filled.tif"
    d8_ptr_name = "d8_pointer.tif"
    slope_name = "slope.tif"

    logger.info("Fill depressions...")
    rc = wbt.fill_depressions(dem_local.name, dem_filled_name)
    logger.info("  rc=%s", rc)

    logger.info("D8 pointer...")
    rc = wbt.d8_pointer(dem_filled_name, d8_ptr_name)
    logger.info("  rc=%s", rc)

    logger.info("Slope...")
    rc = wbt.slope(dem_filled_name, slope_name, units="percent")
    logger.info("  rc=%s", rc)

    flow_acc_name = "flow_accumulation.tif"
    logger.info("Flow accumulation (D8)...")
    rc = wbt.d8_flow_accumulation(dem_filled_name, flow_acc_name, out_type="cells")
    logger.info("  rc=%s", rc)

    dem_filled = saida_dir / dem_filled_name
    d8_ptr = saida_dir / d8_ptr_name
    slope = saida_dir / slope_name
    flow_acc = saida_dir / flow_acc_name

    for arquivo, label in [
        (dem_filled, "dem_filled"),
        (d8_ptr, "d8_pointer"),
        (slope, "slope"),
        (flow_acc, "flow_accumulation"),
    ]:
        if not arquivo.exists():
            raise FileNotFoundError(f"WhiteboxTools não gerou {label}: {arquivo}")
        logger.info("OK %s: %s (%.1f MB)", label, arquivo.name, arquivo.stat().st_size / 1e6)

    return {
        "dem_filled": dem_filled,
        "d8_pointer": d8_ptr,
        "slope": slope,
        "flow_accumulation": flow_acc,
    }


def snap_outlet_para_canal(
    flow_acc_path: Path,
    lat: float,
    lon: float,
    buffer_m: float = 60.0,
) -> tuple[int, int, float, float]:
    """Snap da coordenada (lat, lon) para o pixel de maior flow accumulation
    dentro de um buffer (em metros) ao redor.

    Retorna (row, col, lat_snapped, lon_snapped).
    """
    import math

    with rasterio.open(flow_acc_path) as src:
        fa = src.read(1).astype(np.float64)
        crs = src.crs
        transform = src.transform
        cell_size = 0.5 * (abs(transform.a) + abs(transform.e))

    # Converte (lat, lon) para o CRS do raster (EPSG:31983)
    from pyproj import Transformer

    target_crs = crs or "EPSG:31983"
    if str(target_crs).upper() not in ("EPSG:4326", "WGS84"):
        tr = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        x, y = tr.transform(lon, lat)
    else:
        x, y = lon, lat

    with rasterio.open(flow_acc_path) as src:
        row0, col0 = src.index(x, y)

    nrows, ncols = fa.shape
    delta = max(1, int(math.ceil(buffer_m / cell_size)))

    r_min = max(0, row0 - delta)
    r_max = min(nrows, row0 + delta + 1)
    c_min = max(0, col0 - delta)
    c_max = min(ncols, col0 + delta + 1)

    janela = fa[r_min:r_max, c_min:c_max]
    if janela.size == 0:
        raise ValueError("Janela de busca vazia (coordenada fora do raster?)")

    flat_idx = int(np.nanargmax(janela))
    rr = r_min + flat_idx // janela.shape[1]
    cc = c_min + flat_idx % janela.shape[1]

    # Recupera lat/lon do pixel snappado
    with rasterio.open(flow_acc_path) as src:
        x_snap, y_snap = src.xy(rr, cc)
    if str(target_crs).upper() not in ("EPSG:4326", "WGS84"):
        tr_back = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
        lon_s, lat_s = tr_back.transform(x_snap, y_snap)
    else:
        lon_s, lat_s = x_snap, y_snap

    logger.info(
        "Snap outlet: pedido (%.4f, %.4f) → (%.4f, %.4f) | row=%d col=%d | flow_acc=%.0f cells | buffer=%.0fm (%d px)",
        lat, lon, lat_s, lon_s, rr, cc, fa[rr, cc], buffer_m, delta,
    )
    return rr, cc, lat_s, lon_s


def _codigo_para_offset(codigo: int) -> tuple[int, int, float] | None:
    """Converte código WhiteboxTools D8 em (dr, dc, distância_unit)."""
    for i, c in enumerate(D8_CODES):
        if c == codigo:
            return int(D8_DR[i]), int(D8_DC[i]), float(D8_DIST[i])
    return None


@njit(cache=True)
def _decode_d8(fdir: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decodifica raster D8 (códigos 1,2,...,128) em (dr, dc, dist_unit).

    Retorna três arrays do mesmo shape de fdir.
    Célula sem direção (nodata, sink) recebe (0, 0, 0).
    """
    nrows, ncols = fdir.shape
    dr = np.zeros_like(fdir, dtype=np.int32)
    dc = np.zeros_like(fdir, dtype=np.int32)
    du = np.zeros_like(fdir, dtype=np.float64)
    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int32)
    dr_lut = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
    dc_lut = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
    du_lut = np.array(
        [1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)], dtype=np.float64
    )
    for r in prange(nrows):
        for c in range(ncols):
            v = fdir[r, c]
            if v <= 0:
                continue
            for i in range(8):
                if codes[i] == v:
                    dr[r, c] = dr_lut[i]
                    dc[r, c] = dc_lut[i]
                    du[r, c] = du_lut[i]
                    break
    return dr, dc, du


@njit(cache=True)
def _downslope_accum(
    weights: np.ndarray,
    dr: np.ndarray,
    dc: np.ndarray,
    du: np.ndarray,
    cell_size: float,
    outlet_row: int,
    outlet_col: int,
    max_steps: int,
) -> np.ndarray:
    """Soma `weights` ao longo do caminho D8 de cada célula até o outlet.

    `weights` deve estar pré-multiplicado por qualquer fator da célula (ex: slope^-1/2).
    A distância física adicionada por passo é cell_size * du (unit dist da LUT D8).
    Se a célula já representa (peso × contribuição), aqui só acumulamos.

    Retorna array com a acumulação por célula.
    """
    nrows, ncols = weights.shape
    out = np.zeros_like(weights, dtype=np.float64)

    for r in prange(nrows):
        for c in range(ncols):
            if not np.isfinite(weights[r, c]):
                continue
            rr, cc = r, c
            total = 0.0
            for _ in range(max_steps):
                if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                    break
                if rr == outlet_row and cc == outlet_col:
                    total += weights[rr, cc]
                    break
                w = weights[rr, cc]
                if np.isfinite(w):
                    total += w
                ddr, ddc = dr[rr, cc], dc[rr, cc]
                if ddr == 0 and ddc == 0:
                    break  # sink ou nodata
                rr += ddr
                cc += ddc
            out[r, c] = total
    return out


def latlon_para_linhacoluna(
    raster_path: Path, lat: float, lon: float
) -> tuple[int, int]:
    """Converte lat/lon em (row, col) no raster."""
    with rasterio.open(raster_path) as src:
        row, col = src.index(lon, lat)
    return int(row), int(col)


def alinhar_raster(
    raster_path: Path,
    referencia_path: Path,
    saida_path: Path,
    metodo: str = "nearest",
) -> Path:
    """Reamostra `raster_path` para o grid de `referencia_path` (mesmo CRS, transform, shape)."""
    from rasterio.warp import Resampling, reproject

    if saida_path.exists():
        return saida_path

    with rasterio.open(referencia_path) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)

    with rasterio.open(raster_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_dtype = src_data.dtype

    dst = np.zeros(ref_shape, dtype=src_dtype)
    resampling = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "average": Resampling.average,
    }[metodo]
    reproject(
        source=src_data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=resampling,
    )
    profile = ref_profile.copy()
    profile.update(dtype=str(src_dtype), count=1, compress="deflate", nodata=0)
    saida_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(saida_path, "w", **profile) as dst_io:
        dst_io.write(dst, 1)
    logger.info("Alinhado %s -> %s (%s)", raster_path.name, saida_path.name, metodo)
    return saida_path


def computar_Dck_por_classe(
    d8_path: Path,
    slope_path: Path,
    lulc_path: Path,
    outlet_latlon: tuple[float, float],
    classes: Iterable[int],
    slope_min: float = 1e-4,
) -> dict[int, np.ndarray]:
    """Computa D_ck por classe LULC.

    Para cada classe k:
        weights_k(célula) = (LULC == k) * cell_size * slope^(-1/2)
        D_ck = Σ weights_k ao longo do caminho D8 até outlet

    `slope` esperado em percent (WhiteboxTools); converte para fração.

    Retorna dict {classe_k: array_2D}.
    """
    with rasterio.open(d8_path) as src:
        fdir = src.read(1).astype(np.int32)
        transform = src.transform
        cell_size_x = abs(transform.a)
        cell_size_y = abs(transform.e)
        cell_size = 0.5 * (cell_size_x + cell_size_y)

    with rasterio.open(slope_path) as src:
        slope = src.read(1).astype(np.float64)
    slope_frac = slope / 100.0
    slope_frac = np.where(slope_frac < slope_min, slope_min, slope_frac)

    with rasterio.open(lulc_path) as src:
        lulc = src.read(1).astype(np.int32)

    if fdir.shape != lulc.shape or fdir.shape != slope.shape:
        raise ValueError(
            f"Shapes divergentes: d8={fdir.shape} slope={slope.shape} lulc={lulc.shape}"
        )

    dr, dc, du = _decode_d8(fdir)

    outlet_row, outlet_col = latlon_para_linhacoluna(
        d8_path, outlet_latlon[0], outlet_latlon[1]
    )
    logger.info("Outlet em (row=%d, col=%d)", outlet_row, outlet_col)

    max_steps = fdir.shape[0] + fdir.shape[1]
    resultados: dict[int, np.ndarray] = {}
    inv_sqrt_slope = 1.0 / np.sqrt(slope_frac)

    for k in classes:
        mask_k = (lulc == k).astype(np.float64)
        # weight = (LULC==k) * cell_size * du * s^(-1/2) = aplicado na célula atual.
        # A contribuição de cada passo para o caminho é cell_size * du[rr,cc].
        # Montamos peso por célula: (LULC==k) * cell_size * du_atual * s^(-1/2) ... mas
        # du é do passo a partir da célula; simplificamos assumindo cell_size médio fixo.
        w_k = mask_k * cell_size * inv_sqrt_slope
        Dck = _downslope_accum(w_k, dr, dc, du, cell_size, outlet_row, outlet_col, max_steps)
        resultados[int(k)] = Dck.astype(np.float32)
        logger.info(
            "Classe %d: D_ck médio=%.2f (m·s/m^(1/2)), max=%.2f",
            k,
            float(Dck[np.isfinite(Dck)].mean()),
            float(np.nanmax(Dck)),
        )

    return resultados


def salvar_raster_como(referencia_path: Path, array: np.ndarray, saida_path: Path) -> None:
    """Salva array como GeoTIFF copiando metadados de outro raster."""
    with rasterio.open(referencia_path) as src:
        profile = src.profile.copy()
    profile.update(dtype=array.dtype.name, count=1, compress="deflate")
    saida_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(saida_path, "w", **profile) as dst:
        dst.write(array, 1)
