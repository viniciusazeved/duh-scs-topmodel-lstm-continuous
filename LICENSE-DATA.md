# Licence of the data and model weights

The code in this repository (`src/`, `scripts/`) is released under the MIT License
([LICENSE](LICENSE)).

The processed datasets (`data/`), the trained model weights (`outputs/grade_sim/**/best_model.pt`),
the per-run results and the logs (`outputs/`) are released under the Creative Commons Attribution
4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). They derive
from the third-party sources below, aggregated or interpolated to the 245 sub-catchments by the
authors. Keep these attributions when redistributing.

- Streamflow at gauge 58585000 and hourly telemetric rainfall at 27 gauges: Agência Nacional de
  Águas e Saneamento Básico (ANA), Rede Hidrometeorológica Nacional, HidroWeb/HidroWebService,
  accessed 2026. Reproduction is permitted provided the source is cited.
- Sub-catchment geometry, area and curve number: ANA, Base Hidrográfica Atlas-Estudos (BHAE) and
  Curva Número na Base Ottocodificada (BHAE_CN2022; Nota Técnica nº 9/2025/COMUC/SHE).
- ERA5-Land (2 m temperature for potential evapotranspiration; precipitation in
  `dataset_58585000_era5land.h5`): Contains modified Copernicus Climate Change Service information
  2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the
  Copernicus information or data it contains. Muñoz Sabater, J. (2019): ERA5-Land hourly data from
  1950 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS),
  doi:10.24381/cds.e2161bac.
- MERGE hourly precipitation, CPTEC/INPE, CC BY 4.0. Rozante, J. R., Moreira, D. S., de Goncalves,
  L. G. G., & Vila, D. A. (2010). Weather and Forecasting, doi:10.1175/2010WAF2222325.1.
- IMERG V07 half-hourly precipitation, NASA GPM and GES DISC, from the Google Earth Engine
  collection NASA/GPM_L3/IMERG_V07, which serves Final Run products and provisional Late Run
  products where the Final Run is not yet available. Huffman, G. J., Stocker, E. F., Bolvin, D. T.,
  Nelkin, E. J., & Tan, J. (2023), doi:10.5067/GPM/IMERG/3B-HH/07 (Final Run) and
  doi:10.5067/GPM/IMERG/3B-HH-L/07 (Late Run).
- ANADEM v1 digital terrain model (concentration time, TWI, Figure 1), IPH/UFRGS and ANA, CC BY 4.0.
  Laipelt, L., et al. (2024). Remote Sensing 16(13), 2321, doi:10.3390/rs16132321. Produced using
  Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018
  provided under COPERNICUS by the European Union and ESA; all rights reserved.
- Land use and land cover: MapBiomas Project, Collection 9, year 2022 (roughness classes of the
  Manning-adjusted concentration time `tc_manning_h`), and the MapBiomas version used by ANA in
  BHAE_CN2022. Souza, C. M., et al. (2020). Remote Sensing 12(17), 2735, doi:10.3390/rs12172735.
- Figure 1 inset: Natural Earth (public domain).

Figure 1 also reads public layers that are not redistributed here (the ANADEM terrain model, the
BHO 2017 5k drainage network of ANA and the IBGE 2025 state boundaries); their sources and terms
of use are listed in [data/gis/README.md](data/gis/README.md#public-layers-not-included-figure-1).

MSWEP V3.16 (GloH2O) is licensed CC BY-NC 4.0 and is not redistributed in this repository.
