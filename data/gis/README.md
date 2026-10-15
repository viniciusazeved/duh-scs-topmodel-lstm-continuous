# GIS layers

Layers of the location map (Figure 1, `scripts/figures/mapa_localizacao.py`) and of the
sub-catchment maps of Figure S2 (`scripts/figures/fig_produtos_evento.py`). Coordinates are in
SIRGAS 2000 / UTM zone 23S (EPSG:31983) unless stated otherwise.

## Files in this folder

| File | Content | Source |
|---|---|---|
| `bacia_manuel_duarte.shp` (with `.shx`, `.dbf`, `.prj`, `.cpg`) | Outline of the catchment at the Manuel Duarte gauge (ANA 58585000), 3,117 km²; two polygons, the catchment and a 597 m² sliver of the raster conversion, dissolved by the scripts | Delineated by the authors from the ANADEM v1 terrain model in ArcGIS Pro (manual step 1 in `scripts/data_prep/README.md`) |
| `ottobacias_tc.gpkg` | The 245 sub-catchments, in the column order of the h5 datasets, with code (`cobacia`), area, curve number and concentration times | ANA BHAE sub-catchments clipped by the outline (manual step 2 in `scripts/data_prep/README.md`); attributes computed by the authors |
| `naturalearth_lowres/` | Countries at 1:110m (177 polygons, EPSG:4326) for the inset of Figure 1 | Natural Earth, public domain; copy of the `naturalearth_lowres` test file distributed with pyogrio 0.12.1 (originally the GeoPandas sample dataset) |

These files are released with the rest of `data/` under CC BY 4.0, with the attributions listed in
[LICENSE-DATA.md](../../LICENSE-DATA.md); Natural Earth is in the public domain. Figure 1 also
reads `data/gauges/`: the gauge metadata and the hourly series, which define the 27 telemetric
gauges of the IDW forcing (6 inside the catchment and 21 outside).

## Public layers not included (Figure 1)

Three public layers of the map are not redistributed here (about 60 MB). Put them in
`data/gis/external/` (a folder to create; it is not part of the package) under the expected names
below, or give their paths by argument or environment variable:

| Layer | Expected name in `data/gis/external/` | Argument | Environment variable |
|---|---|---|---|
| Terrain model | `dem.tif` | `--dem` | `FIG1_DEM` |
| Drainage network | `BH.gdb` (layer `hidro`) | `--drainage`, `--drainage-layer` | `FIG1_DRAINAGE` |
| State boundaries | `BR_UF_2025/BR_UF_2025.shp` | `--states` | `FIG1_STATES` |
| Folder of the three layers | `data/gis/external/` | `--layers-dir` | `FIG1_LAYERS_DIR` |

```bash
uv sync --extra gis      # before installing torch: uv sync removes it (README at the root)
uv run python scripts/figures/mapa_localizacao.py --layers-dir /path/to/layers
```

### Terrain model: ANADEM v1 (`dem.tif`)

- Product: ANADEM v1.0, a 30 m digital terrain model of South America, by IPH/UFRGS and ANA,
  licensed CC BY 4.0. Cite Laipelt, L., et al. (2024), Remote Sensing 16(13), 2321,
  doi:10.3390/rs16132321, and keep the Copernicus WorldDEM-30 notice given in
  [LICENSE-DATA.md](../../LICENSE-DATA.md).
- Download: tiles by MGRS zone, listed in ANA's metadata catalogue (record
  `93664c15-1ff8-4e87-bbed-2bb69d321309` at https://metadados.snirh.gov.br). The catchment lies in
  tile 23K: https://metadados.snirh.gov.br/files/anadem_v1_tiles/anadem_v1_23K.tif. The model is
  also available in Google Earth Engine as `projects/et-brasil/assets/anadem/v1`.
- What the script needs: a GeoTIFF in EPSG:31983 that covers the catchment, with elevations in
  metres. It masks the DEM with the catchment outline; cells that are not positive (nodata) are
  left out. The colour ramp runs from the 1st percentile of the elevations inside the catchment to
  their maximum.
- File used for the submitted figure: `dem.tif`, 26,424,576 bytes, SHA-256
  `633ca982d1d095fd48b05fddfc2f67a8996c2e5de8c2aeb814248e96147c882c`. It is ANADEM v1 projected to
  EPSG:31983 and clipped to the extent of the catchment in ArcGIS Pro: 3,940 x 1,585 cells of
  30.25 m x 30.24 m, bounds 534243.58, 7523943.44, 653413.65, 7571871.83 (nodata 0). Its CRS is
  stored as an unnamed `LOCAL_CS` ("SIRGAS 2000 / UTM zone 23S") without an EPSG code; the script
  takes such a file as EPSG:31983 and stops if the DEM declares another EPSG code. The `.tfw`,
  `.ovr` and `.aux.xml` files that ArcGIS wrote next to it are not needed.
- An equivalent file can be made with GDAL, for example
  `gdalwarp -t_srs EPSG:31983 -te 534243.58 7523943.44 653413.65 7571871.83 -tr 30.2462 30.2387 anadem_v1_23K.tif dem.tif`.
  The authors did not use this route, and the resampling of the ArcGIS projection was not
  recorded, so the hypsometry of such a file differs slightly from the submitted figure.
- The script sets `GDAL_PAM_ENABLED=NO` (unless already set), so that GDAL does not write
  `.aux.xml` files next to the DEM.

### Drainage network: BHO 2017 5k (`BH.gdb`, layer `hidro`)

- Product: ANA, Base Hidrográfica Ottocodificada Multiescalas 2017 5k (BHO 2017 5k), published on
  16 July 2018, free access; cite ANA as the source. The drainage reaches are the file
  `geoft_bho_2017_5k_trecho_drenagem.gpkg` of record `f7b1fc91-f5bc-4d0d-9f4f-f4e5061e5d8f` at
  https://metadados.snirh.gov.br (the record is marked as superseded by the Base Hidrográfica
  Atlas-Estudos, BHAE, and still lists the files).
- What the script needs: the reach lines and their field `nuareamont` (upstream drainage area,
  km²). It clips the reaches to the catchment and draws those with at least 60 km² of upstream
  area (115 reaches), with widths growing with the square root of that area.
- File used for the submitted figure: the file geodatabase `BH.gdb` (14,154,041 bytes) prepared in
  ArcGIS Pro. Its layer `hidro` holds 6,318 BHO 2017 5k reaches of the Paraíba do Sul region
  (version field "BHO bho_2017_v_01_05_5k de 2018-06-20") in EPSG:31983, with ANA's surface water
  availability attributes (version out2020), which the figure does not use. The geodatabase also
  holds the layers `bh_rp` and `bh_rps` (outlines of the Preto and Paraíba do Sul basins), used
  only by earlier layouts of the map.
- With the file from ANA: `--drainage geoft_bho_2017_5k_trecho_drenagem.gpkg --drainage-layer ""`
  (an empty layer name reads the first layer). The script reads the whole layer, so a regional
  clip around the catchment is faster.

### State boundaries: IBGE 2025 (`BR_UF_2025/BR_UF_2025.shp`)

- Product: IBGE, Malha Municipal Digital 2025, state boundaries (`BR_UF_2025.zip`), from
  https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais.html. Use is
  subject to IBGE's methodological note "Informações técnicas e legais para a utilização dos dados
  publicados"; cite IBGE as the source.
- What the script needs: the shapefile extracted from the archive into a folder `BR_UF_2025/`
  (27 polygons in SIRGAS 2000 geographic coordinates, EPSG:4674), with the field `SIGLA_UF`. It
  draws the border between Rio de Janeiro (RJ) and Minas Gerais (MG) and labels the two parts of
  the catchment.
- File used for the submitted figure: `BR_UF_2025.shp`, 19,753,124 bytes, SHA-256
  `23a9b54fc2dbfbe46298d3152ed4ccfdb60f9e7f178b00c50526741ecbe142d7`.

### Sub-catchments

The generator of the submitted figure read the sub-catchments from `ottobacias_cn_2022.gpkg`
(the ANA BHAE_CN2022 sub-catchments of the catchment, 11,276,288 bytes, SHA-256
`7af7838afbb968867e618c5b9d9d511bce441c36a685092ff6b77586e5d44005`). Its 245 geometries are
identical, in the same order and CRS, to those of `ottobacias_tc.gpkg`, and both files give the
same PNG, so the script reads the shipped file by default (`--sub-catchments` or
`FIG1_SUBCATCHMENTS` to change it).

### Reproducibility of the PNG

With the three files identified above and the versions pinned in `pyproject.toml` (Python 3.13,
matplotlib 3.10.8 and the `gis` extra: geopandas 1.1.2, rasterio 1.5.0, pyogrio 0.12.1,
shapely 2.1.2, pyproj 3.7.2), the script reproduces the submitted Figure 1 byte for byte (PNG
SHA-256 `16c9877582a8beace98919b3eaa546b0ffb3108c7f7bde7807e7c121316d16c3`; checked on Windows
with Python 3.13.10). Layers rebuilt from the public sources give the same map up to small
differences, since the projection of the DEM and the drawing order of the clipped reaches depend
on the files used.

## Figure S2

`fig_produtos_evento.py` draws the 245 polygons of `ottobacias_tc.gpkg` (in lon/lat) with the
rainfall of each product at 13:00 UTC on 1 November 2021, read from the h5 datasets in `data/`.
The outlet marker is the centroid of the sub-catchment with the shortest concentration time
(`ottobacia/tc_base_h` of the telemetric h5). The MSWEP dataset is not redistributed (see the
README at the root of the repository); its map is drawn when `--mswep` or the environment
variable `MSWEP_H5` points to the file. Without it the MSWEP map is replaced by a note, and the
other maps and the colour bar are those of the submitted figure.
