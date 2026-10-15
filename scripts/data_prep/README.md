# Provenance of the processed datasets

The scripts in this folder are copies of the code that built the processed datasets in `data/`,
kept as they were run, including the absolute paths of the authors' machine. They document how
each file was produced; they are not meant to run unchanged.

Rebuilding the chain requires credentials and raw data that this package does not redistribute
(an ANA HidroWebService account, a Google Earth Engine project with export to Google Drive, access
to MSWEP through GloH2O, and roughly 160 GiB of raw gridded products) and three manual GIS steps,
[listed at the end](#manual-gis-steps-arcgis-pro).

## Chain

1. Streamflow, static attributes and splits (`notebooks/`, code cells exported from Jupyter
   notebooks without their outputs).
   - `01_vazao_eda.py`: hourly streamflow at ANA gauge 58585000 as the mean of the 15 min record.
   - `02_ottobacias_cn_eda.py`: curve number `cn_2022`, field `ANA_med` of the ANA product
     BHAE_CN2022.
   - `04_tc_eda.py`: zonal means of the concentration-time rasters per sub-catchment.
   - `05_dataset_creation.py`: base dataset with streamflow, static attributes, timestamps and
     the train, validation and test splits.
2. Concentration time (`tc/`; Supplementary Tables S1 and S2).
   - `download_lulc_mapbiomas.py` and `process_lulc.py`: MapBiomas land use and land cover.
   - `calculate_tc_manning.py`: Manning-adjusted concentration time, Tc_manning = Tc_base x n /
     n_ref per pixel, with n_ref = 0.035.
   - The base travel-time raster (`tc_seconds.tif`, time of travel to the outlet) was produced in
     ArcGIS Pro (manual step 3).
3. Rain gauges (`gauges/`; output in `data/gauges/`).
   - `coletar_postos.py`: inventory of ANA gauges in Rio de Janeiro, Minas Gerais and São Paulo
     within 20 km of the catchment.
   - `screening_telem.py`: screening of the telemetric gauges.
   - `coletar_telem.py`: hourly series through the HidroWebService API. The gauge scripts import
     an ANA API client (`cabra_h.ana`) from another project of the first author, which is not part
     of this package, and need an ANA account.
4. Telemetric rainfall forcing (`telemetric_forcing/`; output `data/dataset_58585000_telem.h5`).
   - `montar_dataset_telem.py`: inverse-distance interpolation of the gauges onto the
     sub-catchments (this original version removes readings above 130 mm/h).
   - `montar_telem_cap.py`, run with `--cap 305`: the same interpolation with readings above
     305 mm/h removed at the gauges before interpolating (the forcing used by every model).
   - `_aplica_cap305_telem.py`: writes that precipitation into the file that already carried
     potential evapotranspiration.
5. Potential evapotranspiration (`pet/` and `products/`).
   - `products/exportar_precip_gee.py` (ERA5-Land 2 m temperature, mean per sub-catchment in
     Google Earth Engine, exported to Google Drive) and `products/baixar_csvs_drive.py`.
   - `pet/calcular_pet_otto.py`: daily Hargreaves-Samani with FAO-56 extraterrestrial radiation
     at the sub-catchment centroid latitude, spread uniformly over the UTC day.
   - `pet/_inject_pet_mswep.py`: copies the same PET into the product files.
6. Gridded rainfall products (`products/`; preparatory sweep of Supplementary Section S5).
   - MERGE: `download_merge.py` (hourly GRIB2 from the CPTEC/INPE server) and
     `aggregate_merge_ottobacias.py` (area weights between the 0.1 degree pixels and the
     sub-catchments).
   - ERA5-Land and IMERG: `exportar_precip_gee.py` (polygon reduction at 1 km scale in Google
     Earth Engine), `baixar_csvs_drive.py`, `consolidar_precip_area_md.py` (area-weighted mean,
     with the centroid value for sub-catchments smaller than a pixel) and
     `montar_dataset_precip.py`.
   - MSWEP: `montar_dataset_mswep_area.py`, from the files obtained from GloH2O. The resulting
     file is not redistributed (see the [main README](../../README.md#datasets)).
7. Topographic wetness index (`twi/`; output `data/processed/twi_attrs.npz`).
   - `flow_paths.py`: WhiteboxTools processing of the DEM (fill, D8 flow directions, slope and
     flow accumulation).
   - `twi.py` and `01_compute_twi.py`: index per cell and histograms per sub-catchment.
8. Diagnostics of Supplementary Section S7 (`diagnostics/`): `_lag_precip_vazao.py` and
   `_diag_timezone.py`. They read an earlier version of the telemetric forcing (130 mm/h); the
   diagnostics are the same with the forcing in `data/`.

## Manual GIS steps (ArcGIS Pro)

1. Catchment outline: polygon of the watershed raster derived from the DEM (RasterToPolygon), saved
   as `data/gis/bacia_manuel_duarte.shp` (3,117 km²).
2. Sub-catchments: the ANA BHAE sub-catchments clipped by that outline, 245 units, saved with
   their concentration time in `data/gis/ottobacias_tc.gpkg`.
3. Travel-time raster `tc_seconds.tif`, input of the concentration time of step 2 of the chain.

## Software used across the chain

Python with numpy, pandas, scipy, h5py, xarray, geopandas, shapely, pyproj, rasterio,
rasterstats, numba, whitebox (WhiteboxTools), earthengine-api, google-api-python-client,
matplotlib, plotly, rich and tqdm; the wetness-index scripts import the package
`topmodel_hud_lstm` from another project of the first author (its `twi.py` module is included
here).
