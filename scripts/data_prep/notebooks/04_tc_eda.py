"""Code cells of the notebook 04_tc_eda.ipynb, exported without outputs.

Provenance only: the code is kept as it was run (absolute paths and all).
Source notebook SHA-256: 258f980803ef8fcfb80709d8c971362f8e00ee0d72063aa4acc7ddce78e91f9c
"""

# %% cell 1
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

# Paths
DATA_DIR = Path('../data')
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

print("Setup OK!")

# %% cell 2
# Coeficientes de Manning por classe LULC
MANNING_COEFS = {
    'Floresta': 0.150,
    'Vegetação Natural': 0.100,
    'Agropecuária': 0.035,
    'Solo Exposto': 0.025,
    'Água': 0.030,
    'Urbano': 0.015
}

N_REF = 0.035  # Referência (agropecuária)

# Criar tabela de fatores de ajuste
df_manning = pd.DataFrame([
    {'Classe': k, 'Manning_n': v, 'Fator_Tc': v / N_REF, 'Fator_Velocidade': N_REF / v}
    for k, v in MANNING_COEFS.items()
])

print("Coeficientes de Manning e Fatores de Ajuste:")
print("=" * 70)
print(df_manning.to_string(index=False))

# %% cell 3
# Visualização dos coeficientes de Manning
cores_lulc = {
    'Floresta': '#1f8d49',
    'Vegetação Natural': '#7dc975',
    'Agropecuária': '#E8D639',
    'Solo Exposto': '#8B4513',
    'Água': '#0066CC',
    'Urbano': '#d4271e'
}

fig = go.Figure()

fig.add_trace(go.Bar(
    x=df_manning['Classe'],
    y=df_manning['Manning_n'],
    marker_color=[cores_lulc[c] for c in df_manning['Classe']],
    text=[f'n={v:.3f}' for v in df_manning['Manning_n']],
    textposition='outside'
))

fig.add_hline(y=N_REF, line_dash='dash', line_color='gray',
              annotation_text=f'Referência (n={N_REF})')

fig.update_layout(
    title='Coeficientes de Manning por Classe de Uso do Solo',
    xaxis_title='Classe LULC',
    yaxis_title='Coeficiente de Manning (n)',
    height=450,
    showlegend=False
)

fig.show()

# %% cell 4
# Carregar ottobacias CN 2022
otto = gpd.read_file(RAW_DIR / 'ottobacias' / 'ottobacias_cn_2022.gpkg')
otto['area_km2'] = otto.geometry.area / 1e6

# Carregar limite da bacia
bacia = gpd.read_file(RAW_DIR / 'bacia' / 'bacia_manuel_duarte.shp')

print(f"Ottobacias carregadas: {len(otto)}")
print(f"Área total: {otto['area_km2'].sum():,.1f} km²")

# %% cell 5
# Carregar estatísticas de LULC
with open(PROCESSED_DIR / 'lulc' / 'estatisticas_lulc_2022.json') as f:
    lulc_stats = json.load(f)

# Criar DataFrame de uso do solo
lulc_data = []
for code, stats in lulc_stats['estatisticas'].items():
    if stats['percent'] > 0:
        lulc_data.append({
            'Codigo': int(code),
            'Classe': lulc_stats['classes'][code],
            'Area_km2': stats['area_km2'],
            'Percentual': stats['percent'],
            'Manning_n': lulc_stats['manning'][code]
        })

df_lulc = pd.DataFrame(lulc_data).sort_values('Percentual', ascending=False)

print("Uso do Solo na Bacia Manuel Duarte (MapBiomas 2022):")
print("=" * 60)
print(df_lulc.to_string(index=False))
print(f"\nÁrea total: {df_lulc['Area_km2'].sum():,.1f} km²")

# %% cell 6
# Carregar rasters
tc_base_file = PROCESSED_DIR / 'tc' / 'tc_base.tif'
tc_manning_file = PROCESSED_DIR / 'tc' / 'tc_manning_2022.tif'
lulc_file = PROCESSED_DIR / 'lulc' / 'lulc_2022.tif'

with rasterio.open(tc_base_file) as src:
    tc_base_data = src.read(1)
    tc_profile = src.profile
    tc_transform = src.transform
    tc_bounds = src.bounds

with rasterio.open(tc_manning_file) as src:
    tc_manning_data = src.read(1)

with rasterio.open(lulc_file) as src:
    lulc_data_raster = src.read(1)

print(f"Tc base shape: {tc_base_data.shape}")
print(f"Tc Manning shape: {tc_manning_data.shape}")
print(f"LULC shape: {lulc_data_raster.shape}")

# %% cell 7
# Carregar estatísticas de Tc
with open(PROCESSED_DIR / 'tc' / 'estatisticas_tc_manning_2022.json') as f:
    tc_stats = json.load(f)

print("Estatísticas do Tc (horas):")
print("=" * 50)
print(f"Tc Base:    Média = {tc_stats['tc_base']['mean_h']:.1f}h, Máx = {tc_stats['tc_base']['max_h']:.1f}h")
print(f"Tc Manning: Média = {tc_stats['tc_manning']['mean_h']:.1f}h, Máx = {tc_stats['tc_manning']['max_h']:.1f}h")
print(f"Aumento:    {tc_stats['adjustment']['diff_pct']:.1f}%")

# %% cell 8
# Preparar dados para visualização
# Converter para WGS84 para mapas
otto_wgs = otto.to_crs(epsg=4326)
bacia_wgs = bacia.to_crs(epsg=4326)

# Calcular centróide
bacia_centroid = bacia_wgs.geometry.unary_union.centroid
center_lat = bacia_centroid.y
center_lon = bacia_centroid.x

print(f"Centro da bacia: {center_lat:.4f}, {center_lon:.4f}")

# %% cell 9
# Extrair Tc e LULC dominante por ottobacia
def extract_raster_stats(gdf, raster_file, stat='mean'):
    """Extrai estatísticas de raster para cada polígono."""
    results = []
    with rasterio.open(raster_file) as src:
        for idx, row in gdf.iterrows():
            try:
                out_image, _ = mask(src, [row.geometry], crop=True, nodata=0)
                values = out_image[0]
                valid = values[values > 0]
                if len(valid) == 0:
                    results.append(np.nan)
                elif stat == 'mean':
                    results.append(valid.mean())
                elif stat == 'mode':
                    unique, counts = np.unique(valid, return_counts=True)
                    results.append(unique[counts.argmax()])
                else:
                    results.append(valid.max())
            except:
                results.append(np.nan)
    return results

print("Extraindo Tc base por ottobacia...")
otto['tc_base_s'] = extract_raster_stats(otto, tc_base_file, 'mean')
otto['tc_base_h'] = otto['tc_base_s'] / 3600

print("Extraindo Tc Manning por ottobacia...")
otto['tc_manning_s'] = extract_raster_stats(otto, tc_manning_file, 'mean')
otto['tc_manning_h'] = otto['tc_manning_s'] / 3600

print("Extraindo LULC dominante por ottobacia...")
otto['lulc_code'] = extract_raster_stats(otto, lulc_file, 'mode')

# Mapear código para nome da classe
lulc_names = {int(k): v for k, v in lulc_stats['classes'].items()}
otto['lulc_classe'] = otto['lulc_code'].map(lulc_names)

# Calcular fator de ajuste
otto['tc_factor'] = otto['tc_manning_h'] / otto['tc_base_h']

print("\nExtração concluída!")

# %% cell 10
# Atualizar otto_wgs com os novos dados
otto_wgs = otto.to_crs(epsg=4326)

# Estatísticas ponderadas por área
valid_mask = otto['tc_base_h'].notna() & otto['tc_manning_h'].notna()
tc_base_ponderado = np.average(otto.loc[valid_mask, 'tc_base_h'], 
                               weights=otto.loc[valid_mask, 'area_km2'])
tc_manning_ponderado = np.average(otto.loc[valid_mask, 'tc_manning_h'], 
                                  weights=otto.loc[valid_mask, 'area_km2'])

print("=" * 60)
print("ESTATÍSTICAS DE Tc POR OTTOBACIA (PONDERADO POR ÁREA)")
print("=" * 60)
print(f"\nTc Base (Maidment):    {tc_base_ponderado:.2f} horas")
print(f"Tc Manning (LULC 2022): {tc_manning_ponderado:.2f} horas")
print(f"Aumento:                {(tc_manning_ponderado / tc_base_ponderado - 1) * 100:+.1f}%")

# %% cell 11
# Criar subplots: Tc Base | LULC | Tc Manning
from plotly.subplots import make_subplots

# Preparar GeoJSON para cada mapa
geojson = otto_wgs.geometry.__geo_interface__

# Cores para LULC
lulc_color_map = {
    'Floresta': '#1f8d49',
    'Vegetacao Natural': '#7dc975',
    'Agropecuaria': '#E8D639',
    'Nao Vegetada': '#8B4513',
    'Agua': '#0066CC',
    'Urbano': '#d4271e'
}

# Criar figura com 3 subplots
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        f'Tc Base (Maidment)<br>Média: {tc_base_ponderado:.1f}h',
        'Uso do Solo (LULC 2022)',
        f'Tc Manning<br>Média: {tc_manning_ponderado:.1f}h'
    ),
    specs=[[{'type': 'choroplethmapbox'}, {'type': 'choroplethmapbox'}, {'type': 'choroplethmapbox'}]],
    horizontal_spacing=0.02
)

# Mapa 1: Tc Base
fig.add_trace(
    go.Choroplethmapbox(
        geojson=geojson,
        locations=otto_wgs.index,
        z=otto_wgs['tc_base_h'],
        colorscale='Viridis',
        zmin=0,
        zmax=80,
        marker_opacity=0.8,
        marker_line_width=0.5,
        colorbar=dict(title='Tc (h)', x=0.28, len=0.8)
    ),
    row=1, col=1
)

# Mapa 2: LULC (usando código numérico para colorscale)
fig.add_trace(
    go.Choroplethmapbox(
        geojson=geojson,
        locations=otto_wgs.index,
        z=otto_wgs['lulc_code'],
        colorscale=[
            [0.0, '#1f8d49'],   # Floresta (1)
            [0.2, '#7dc975'],   # Vegetação Natural (2)
            [0.4, '#E8D639'],   # Agropecuária (3)
            [0.6, '#8B4513'],   # Não Vegetada (4)
            [0.8, '#0066CC'],   # Água (5)
            [1.0, '#d4271e']    # Urbano (7)
        ],
        zmin=1,
        zmax=7,
        marker_opacity=0.8,
        marker_line_width=0.5,
        showscale=False,
        text=otto_wgs['lulc_classe'],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=1, col=2
)

# Mapa 3: Tc Manning
fig.add_trace(
    go.Choroplethmapbox(
        geojson=geojson,
        locations=otto_wgs.index,
        z=otto_wgs['tc_manning_h'],
        colorscale='Viridis',
        zmin=0,
        zmax=80,
        marker_opacity=0.8,
        marker_line_width=0.5,
        colorbar=dict(title='Tc (h)', x=1.02, len=0.8)
    ),
    row=1, col=3
)

# Configurar layout
fig.update_layout(
    title=dict(
        text='Fluxo Metodológico: Tc Base × (n_LULC / n_ref) = Tc Manning',
        x=0.5
    ),
    height=500,
    margin=dict(l=0, r=0, t=80, b=0),
    mapbox=dict(
        style='white-bg',
        center=dict(lat=center_lat, lon=center_lon),
        zoom=7.5
    ),
    mapbox2=dict(
        style='white-bg',
        center=dict(lat=center_lat, lon=center_lon),
        zoom=7.5
    ),
    mapbox3=dict(
        style='white-bg',
        center=dict(lat=center_lat, lon=center_lon),
        zoom=7.5
    )
)

fig.show()

# %% cell 12
# Gráfico de pizza do uso do solo
cores_pizza = ['#1f8d49', '#E8D639', '#d4271e', '#8B4513', '#0066CC']

fig = go.Figure(data=[go.Pie(
    labels=df_lulc['Classe'],
    values=df_lulc['Area_km2'],
    marker_colors=cores_pizza[:len(df_lulc)],
    textinfo='label+percent',
    textposition='outside',
    hole=0.3
)])

fig.update_layout(
    title='Distribuição do Uso do Solo - Bacia Manuel Duarte (2022)',
    height=450,
    showlegend=True
)

fig.show()

# %% cell 13
# ==============================================================================
# PLOT LULC - Classe Dominante por Ottobacia (Plotly)
# ==============================================================================

# Cores das classes LULC
lulc_colors_map = {
    1: '#1f8d49',  # Floresta
    3: '#E8D639',  # Agropecuária
    4: '#8B4513',  # Não Vegetada
    5: '#0066CC',  # Água
    7: '#d4271e',  # Urbano
}

# Criar colorscale discreta para LULC
fig = go.Figure(go.Choroplethmap(
    geojson=geojson,
    locations=otto_wgs.index,
    z=otto_wgs['lulc_code'],
    colorscale=[
        [0.0, '#1f8d49'],   # Floresta (1)
        [0.33, '#E8D639'],  # Agropecuária (3)
        [0.5, '#8B4513'],   # Não Vegetada (4)
        [0.67, '#0066CC'],  # Água (5)
        [1.0, '#d4271e']    # Urbano (7)
    ],
    zmin=1,
    zmax=7,
    marker_opacity=0.8,
    marker_line_width=0.5,
    marker_line_color='gray',
    showscale=False,
    text=otto_wgs['lulc_classe'],
    hovertemplate='<b>%{text}</b><extra></extra>'
))

fig.update_layout(
    title=dict(
        text='Uso e Cobertura do Solo - Classe Dominante por Ottobacia (LULC 2022)',
        x=0.5,
        font=dict(size=16)
    ),
    map=dict(
        style='white-bg',
        center=dict(lat=center_lat, lon=center_lon),
        zoom=8
    ),
    height=550,
    margin=dict(l=10, r=10, t=50, b=10),
    # Legenda manual
    annotations=[
        dict(x=0.02, y=0.98, xref='paper', yref='paper', showarrow=False,
             text='<b>Legenda:</b>', font=dict(size=12)),
        dict(x=0.02, y=0.93, xref='paper', yref='paper', showarrow=False,
             text='<span style="color:#1f8d49">■</span> Floresta (38.5%)', font=dict(size=11)),
        dict(x=0.02, y=0.88, xref='paper', yref='paper', showarrow=False,
             text='<span style="color:#E8D639">■</span> Agropecuária (60.5%)', font=dict(size=11)),
        dict(x=0.02, y=0.83, xref='paper', yref='paper', showarrow=False,
             text='<span style="color:#d4271e">■</span> Urbano (0.4%)', font=dict(size=11)),
    ]
)

fig.show()

# Estatísticas
print('\nDistribuição das classes dominantes nas ottobacias:')
print(otto['lulc_classe'].value_counts())

# %% cell 14
# ==============================================================================
# PLOT Tc Base vs Tc Manning - Mesma escala (0-80h)
# ==============================================================================

# Estatísticas
tc_base_mean = tc_base_ponderado
tc_manning_mean = tc_manning_ponderado
tc_base_max = otto['tc_base_h'].max()
tc_manning_max = otto['tc_manning_h'].max()

# Criar figura com 2 subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f'(a) Tc Base (Maidment 1996)<br>Média: {tc_base_mean:.1f}h | Máx: {tc_base_max:.1f}h',
        f'(b) Tc Manning (LULC 2022)<br>Média: {tc_manning_mean:.1f}h | Máx: {tc_manning_max:.1f}h'
    ),
    specs=[[{'type': 'choroplethmap'}, {'type': 'choroplethmap'}]],
    horizontal_spacing=0.02
)

# Mapa 1: Tc Base
fig.add_trace(
    go.Choroplethmap(
        geojson=geojson,
        locations=otto_wgs.index,
        z=otto_wgs['tc_base_h'],
        colorscale='Viridis',
        zmin=0,
        zmax=80,
        marker_opacity=0.8,
        marker_line_width=0.5,
        marker_line_color='gray',
        showscale=False
    ),
    row=1, col=1
)

# Mapa 2: Tc Manning
fig.add_trace(
    go.Choroplethmap(
        geojson=geojson,
        locations=otto_wgs.index,
        z=otto_wgs['tc_manning_h'],
        colorscale='Viridis',
        zmin=0,
        zmax=80,
        marker_opacity=0.8,
        marker_line_width=0.5,
        marker_line_color='gray',
        colorbar=dict(
            title=dict(text='Tc (h)', font=dict(size=14)),
            x=1.02,
            len=0.9,
            thickness=15
        )
    ),
    row=1, col=2
)

# Configurar layout
fig.update_layout(
    title=dict(
        text='Tempo de Concentração: Base (Topográfico) vs Manning (LULC)<br>Escala uniforme: 0-80h',
        x=0.5,
        font=dict(size=16)
    ),
    height=550,
    margin=dict(l=0, r=0, t=100, b=0),
    map=dict(
        style='white-bg',
        center=dict(lat=center_lat, lon=center_lon),
        zoom=8
    ),
    map2=dict(
        style='white-bg',
        center=dict(lat=center_lat, lon=center_lon),
        zoom=8
    )
)

fig.show()

# Estatísticas comparativas
print('=' * 60)
print('COMPARAÇÃO Tc BASE vs Tc MANNING')
print('=' * 60)
print(f'\n{"Métrica":<20} {"Tc Base":>12} {"Tc Manning":>12} {"Aumento":>12}')
print('-' * 56)
print(f'{"Média (h)":<20} {tc_base_mean:>12.1f} {tc_manning_mean:>12.1f} {(tc_manning_mean/tc_base_mean-1)*100:>+11.0f}%')
print(f'{"Máximo (h)":<20} {tc_base_max:>12.1f} {tc_manning_max:>12.1f} {(tc_manning_max/tc_base_max-1)*100:>+11.0f}%')
print('\nInterpretação:')
print('  → Tc Manning ~2.4x maior que Tc Base')
print('  → Floresta (38.5% da área) retarda significativamente o escoamento')
print('  → Efeito mais pronunciado nas cabeceiras (maior cobertura florestal)')

# %% cell 15
# Estatísticas dos rasters (apenas pixels válidos)
tc_base_valid = tc_base_data[(tc_base_data > 0) & ~np.isnan(tc_base_data)]
tc_manning_valid = tc_manning_data[(tc_manning_data > 0) & ~np.isnan(tc_manning_data)]

# Converter para horas
tc_base_h = tc_base_valid / 3600
tc_manning_h = tc_manning_valid / 3600

print("=" * 65)
print("ESTATÍSTICAS GLOBAIS DOS RASTERS DE Tc")
print("=" * 65)
print()
print(f"{'Métrica':<20} {'Tc Base (h)':>15} {'Tc Manning (h)':>15} {'Diferença':>15}")
print("-" * 65)
print(f"{'Mínimo':<20} {tc_base_h.min():>15.2f} {tc_manning_h.min():>15.2f} {tc_manning_h.min() - tc_base_h.min():>+15.2f}")
print(f"{'Média':<20} {tc_base_h.mean():>15.2f} {tc_manning_h.mean():>15.2f} {tc_manning_h.mean() - tc_base_h.mean():>+15.2f}")
print(f"{'Mediana':<20} {np.median(tc_base_h):>15.2f} {np.median(tc_manning_h):>15.2f} {np.median(tc_manning_h) - np.median(tc_base_h):>+15.2f}")
print(f"{'Máximo':<20} {tc_base_h.max():>15.2f} {tc_manning_h.max():>15.2f} {tc_manning_h.max() - tc_base_h.max():>+15.2f}")
print()
print(f"Aumento médio: {(tc_manning_h.mean() / tc_base_h.mean() - 1) * 100:+.1f}%")

# %% cell 16
# Histogramas comparativos
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=tc_base_h,
    name='Tc Base (Maidment)',
    opacity=0.7,
    nbinsx=50,
    marker_color='#1f77b4'
))

fig.add_trace(go.Histogram(
    x=tc_manning_h,
    name='Tc Manning (LULC 2022)',
    opacity=0.7,
    nbinsx=50,
    marker_color='#ff7f0e'
))

fig.update_layout(
    title='Distribuição do Tempo de Concentração: Base vs Manning',
    xaxis_title='Tempo de Concentração (horas)',
    yaxis_title='Frequência',
    barmode='overlay',
    height=450
)

fig.add_vline(x=tc_base_h.mean(), line_dash='dash', line_color='#1f77b4',
              annotation_text=f'Tc Base: {tc_base_h.mean():.1f}h')
fig.add_vline(x=tc_manning_h.mean(), line_dash='dash', line_color='#ff7f0e',
              annotation_text=f'Tc Manning: {tc_manning_h.mean():.1f}h')

fig.show()

# %% cell 17
# Scatter plot Tc base vs Tc Manning
fig = px.scatter(
    otto,
    x='tc_base_h',
    y='tc_manning_h',
    size='area_km2',
    color='lulc_classe',
    color_discrete_map={
        'Floresta': '#1f8d49',
        'Agropecuaria': '#E8D639',
        'Urbano': '#d4271e',
        'Agua': '#0066CC',
        'Nao Vegetada': '#8B4513'
    },
    hover_data=['cobacia', 'area_km2', 'tc_factor'],
    title='Relação Tc Base vs Tc Manning por Ottobacia',
    labels={
        'tc_base_h': 'Tc Base (horas)',
        'tc_manning_h': 'Tc Manning (horas)',
        'lulc_classe': 'LULC',
        'area_km2': 'Área (km²)'
    }
)

# Linha 1:1
max_tc = max(otto['tc_base_h'].max(), otto['tc_manning_h'].max())
fig.add_trace(go.Scatter(
    x=[0, max_tc],
    y=[0, max_tc],
    mode='lines',
    name='1:1',
    line=dict(dash='dash', color='gray')
))

fig.update_layout(height=500)
fig.show()

# %% cell 18
# ==============================================================================
# CONSTRUÇÃO DO IUH - MÉTODO DE MAIDMENT (1996)
# ==============================================================================

# Parâmetros
PIXEL_AREA_M2 = 900  # 30m x 30m
C_MAIDMENT = 0.5     # Expoente c de Maidment para área

# Intervalo de tempo para o histograma (1 hora)
DT_HOURS = 1
bins_hours = np.arange(0, 121, DT_HOURS)
bins_seconds = bins_hours * 3600

# Tc Manning em segundos (pixels válidos)
tc_valid_s = tc_manning_valid

# Histograma de contagem de pixels por faixa de Tc
counts, edges = np.histogram(tc_valid_s, bins=bins_seconds)

# Aplicação do expoente c de Maidment
area_per_bin = counts * (PIXEL_AREA_M2 ** C_MAIDMENT)
area_normalized = area_per_bin / PIXEL_AREA_M2
iuh_por_mm = area_normalized / 1000

# Centros dos bins em horas
t_hours = (bins_hours[:-1] + bins_hours[1:]) / 2

# Validação com evento real
Q_PICO_REAL = 441.9  # m³/s (Nov/2021)
P_EFETIVA = 50       # mm estimado
q_estimado = P_EFETIVA * iuh_por_mm.max()

print("=" * 70)
print("IUH - MÉTODO DE MAIDMENT COM EXPOENTE c = 0.5")
print("=" * 70)
print("")
print("Parâmetros:")
print(f"  Expoente c (Maidment): {C_MAIDMENT}")
print(f"  Fator de escala: 900^{C_MAIDMENT} = {PIXEL_AREA_M2**C_MAIDMENT:.1f}")
print("")
print(f"Pico do IUH: {iuh_por_mm.max():.2f} m²/s por mm de chuva")
print(f"Tempo do pico: {t_hours[iuh_por_mm.argmax()]:.0f} horas")
print("")
print("VALIDAÇÃO COM EVENTO REAL (Nov/2021):")
print(f"  Q pico observado: {Q_PICO_REAL:.1f} m³/s")
print(f"  Q estimado (P_ef={P_EFETIVA}mm): {q_estimado:.1f} m³/s")
print(f"  Razão: {Q_PICO_REAL/q_estimado:.1f}x")
print("=" * 70)

# %% cell 19
# Visualização do IUH
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t_hours,
    y=iuh_por_mm,
    mode='lines',
    fill='tozeroy',
    name='IUH',
    line=dict(color='#1f77b4', width=2)
))

# Marcar Tc médio
tc_manning_medio_h = tc_manning_h.mean()
fig.add_vline(
    x=tc_manning_medio_h,
    line_dash='dash',
    line_color='red',
    annotation_text=f'Tc médio: {tc_manning_medio_h:.1f}h'
)

fig.update_layout(
    title=f'Hidrograma Unitário Instantâneo (IUH) - Bacia Manuel Duarte<br>Método de Maidment com c = 0.5',
    xaxis_title='Tempo de Viagem (horas)',
    yaxis_title='IUH (m²/s por mm de chuva)',
    height=500,
    showlegend=False
)

# Adicionar anotação com pico
fig.add_annotation(
    x=0.98, y=0.95,
    xref='paper', yref='paper',
    text=f'Pico: {iuh_por_mm.max():.2f} m²/s/mm<br>em t = {t_hours[iuh_por_mm.argmax()]:.0f}h',
    showarrow=False,
    font=dict(size=12),
    bgcolor='white',
    bordercolor='black',
    borderwidth=1
)

fig.show()

# %% cell 20
# Comparação IUH: Base vs Manning (ambos com expoente c = 0.5)

# IUH para Tc Base (aplicando mesmo método)
tc_base_valid_s = tc_base_valid  # em segundos
counts_base, _ = np.histogram(tc_base_valid_s, bins=bins_seconds)

# Aplicar expoente c de Maidment
area_per_bin_base = counts_base * (PIXEL_AREA_M2 ** C_MAIDMENT)
area_normalized_base = area_per_bin_base / PIXEL_AREA_M2
iuh_base_por_mm = area_normalized_base / 1000

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t_hours,
    y=iuh_base_por_mm,
    mode='lines',
    fill='tozeroy',
    name='IUH Base (Maidment)',
    line=dict(color='#1f77b4', width=2),
    opacity=0.7
))

fig.add_trace(go.Scatter(
    x=t_hours,
    y=iuh_por_mm,
    mode='lines',
    fill='tozeroy',
    name='IUH Manning (LULC 2022)',
    line=dict(color='#ff7f0e', width=2),
    opacity=0.7
))

# Marcar Tc médios
tc_base_medio_h = tc_base_h.mean()
fig.add_vline(x=tc_base_medio_h, line_dash='dash', line_color='#1f77b4',
              annotation_text=f'Tc Base: {tc_base_medio_h:.1f}h')
fig.add_vline(x=tc_manning_medio_h, line_dash='dash', line_color='#ff7f0e',
              annotation_text=f'Tc Manning: {tc_manning_medio_h:.1f}h')

fig.update_layout(
    title='Comparação do IUH: Base (Maidment) vs Manning (LULC 2022)<br>Ambos com expoente c = 0.5',
    xaxis_title='Tempo de Viagem (horas)',
    yaxis_title='IUH (m²/s por mm de chuva)',
    height=500,
    legend=dict(x=0.85, y=0.95)
)

fig.show()

print("\nInterpretação:")
print(f"  IUH Base: pico de {iuh_base_por_mm.max():.2f} m²/s/mm em t = {t_hours[iuh_base_por_mm.argmax()]:.0f}h")
print(f"  IUH Manning: pico de {iuh_por_mm.max():.2f} m²/s/mm em t = {t_hours[iuh_por_mm.argmax()]:.0f}h")
print(f"  → Manning produz IUH mais disperso (floresta retarda escoamento)")

# %% cell 21
# IUH suavizado com kernel gaussiano
from scipy.ndimage import gaussian_filter1d

# Diferentes valores de sigma (em horas)
sigmas = [1, 2, 4, 6]

fig = go.Figure()

# IUH original
fig.add_trace(go.Scatter(
    x=t_hours,
    y=iuh_por_mm,
    mode='lines',
    name='IUH Original',
    line=dict(color='gray', width=1, dash='dot')
))

# IUH suavizado para diferentes sigmas
cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, sigma in enumerate(sigmas):
    # sigma em número de bins (cada bin = 1 hora)
    iuh_smooth = gaussian_filter1d(iuh_por_mm, sigma=sigma)
    fig.add_trace(go.Scatter(
        x=t_hours,
        y=iuh_smooth,
        mode='lines',
        name=f'σ = {sigma}h',
        line=dict(color=cores[i], width=2)
    ))

fig.update_layout(
    title='IUH Suavizado com Kernel Gaussiano (Parâmetro σ Aprendível)',
    xaxis_title='Tempo de Viagem (horas)',
    yaxis_title='IUH (m²/s por mm)',
    height=500,
    legend=dict(x=0.75, y=0.95)
)

fig.show()

print("\nInterpretação:")
print("  σ pequeno: mantém estrutura detalhada do IUH")
print("  σ grande: suaviza e dispersa a resposta")
print("  No modelo TTD-SCS-LSTM, σ é otimizado durante o treinamento")

# %% cell 22
# Calcular integral do IUH para validação
integral_iuh = np.sum(iuh_por_mm) * 1000 * PIXEL_AREA_M2  # m²

print("=" * 70)
print("RESUMO - TEMPO DE CONCENTRAÇÃO E HIDROGRAMA UNITÁRIO")
print("=" * 70)
print()
print("METODOLOGIA:")
print("  1. Tc Base: Maidment (1996) - Equação 15, página 838")
print("     V = Vm × (S^b × A^c) / [S^b × A^c]_m")
print("     Vm = velocidade média de referência")
print("     b = c = 0.5 (valores recomendados por Maidment et al., 1996)")
print()
print("  2. Tc Manning: Ajuste por rugosidade LULC 2022")
print("     Tc_Manning = Tc_Base × (n_local / n_ref)")
print("     n_ref = 0.035 (agropecuária)")
print()
print("USO DO SOLO (MapBiomas 2022):")
for _, row in df_lulc.iterrows():
    print(f"  {row['Classe']:<20} {row['Area_km2']:>8.1f} km² ({row['Percentual']:>5.1f}%)  n={row['Manning_n']:.3f}")
print()
print("RESULTADOS:")
print(f"  Tc Base médio:    {tc_base_ponderado:.2f} horas")
print(f"  Tc Manning médio: {tc_manning_ponderado:.2f} horas")
print(f"  Aumento:          {(tc_manning_ponderado / tc_base_ponderado - 1) * 100:+.1f}%")
print()
print("HIDROGRAMA UNITÁRIO INSTANTÂNEO (IUH):")
print(f"  Unidade: m²/s por mm de chuva")
print(f"  Integral: {integral_iuh/1e6:,.0f} km² (≈ área da bacia)")
print(f"  Pico:     {iuh_por_mm.max():.2f} m²/s/mm em t = {t_hours[iuh_por_mm.argmax()]:.0f}h")
print()
print("PARÂMETROS APRENDÍVEIS NO MODELO TTD-SCS-LSTM:")
print("  - tc_scale: fator de escala do Tc")
print("  - sigma: dispersão do kernel gaussiano (suavização do IUH)")
print()
print("REFERÊNCIAS:")
print("  - Maidment et al. (1996). Hydrological Processes, 10(6), 831-844.")
print("  - Chow, V. T. (1959). Open-Channel Hydraulics. McGraw-Hill.")
print("  - Engman, E. T. (1986). J. Irrigation and Drainage Eng., 112(1), 39-53.")
print()
print("=" * 70)
