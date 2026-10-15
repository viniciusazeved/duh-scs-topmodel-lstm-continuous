"""Code cells of the notebook 02_ottobacias_cn_eda.ipynb, exported without outputs.

Provenance only: the code is kept as it was run (absolute paths and all).
Source notebook SHA-256: a985ef0f9994fd30156706f39f83225a813b137a5758de2d1481232b9e952600
"""

# %% cell 1
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
from pathlib import Path
import json

# Configurações
DATA_DIR = Path('../data/raw')
OUTPUT_DIR = Path('../data/processed')
OUTPUT_DIR.mkdir(exist_ok=True)

print('Setup OK!')

# %% cell 2
# Carregar CN 2022
otto = gpd.read_file(DATA_DIR / 'ottobacias' / 'ottobacias_cn_2022.gpkg')
otto['area_km2'] = otto.geometry.area / 1e6

# Carregar limite da bacia
bacia = gpd.read_file(DATA_DIR / 'bacia' / 'bacia_manuel_duarte.shp')

print('Dados carregados!')
print(f'  CN 2022: {len(otto)} ottobacias')
print(f'  Área total: {otto["area_km2"].sum():.1f} km²')

# %% cell 3
# Calcular estatísticas ponderadas por área
cn_pond = np.average(otto['ANA_med'], weights=otto['area_km2'])

# Calcular S (retenção potencial)
otto['S_mm'] = 25400 / otto['ANA_med'] - 254
s_pond = np.average(otto['S_mm'], weights=otto['area_km2'])

print('=' * 60)
print('ESTATÍSTICAS - CN 2022 ANA')
print('=' * 60)
print()
print(f'{"Métrica":<30} {"Valor":>15}')
print('-' * 45)
print(f'{"Ottobacias":<30} {len(otto):>15,}')
print(f'{"Área total (km²)":<30} {otto["area_km2"].sum():>15,.1f}')
print(f'{"Área média (km²)":<30} {otto["area_km2"].mean():>15.2f}')
print(f'{"CN mínimo":<30} {otto["ANA_med"].min():>15.1f}')
print(f'{"CN máximo":<30} {otto["ANA_med"].max():>15.1f}')
print(f'{"CN médio (ponderado)":<30} {cn_pond:>15.1f}')
print(f'{"S médio (mm)":<30} {s_pond:>15.1f}')
print()
print('Interpretação:')
print(f'  → CN = {cn_pond:.0f} indica bacia com permeabilidade moderada')
print(f'  → S = {s_pond:.0f} mm de retenção potencial antes do escoamento')

# %% cell 4
# Histograma do CN
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=otto['ANA_med'],
    name='CN 2022 ANA',
    nbinsx=25,
    marker_color='#2E86AB'
))

# Linha de média
fig.add_vline(x=cn_pond, line_dash='dash', line_color='red',
              annotation_text=f'Média ponderada: {cn_pond:.1f}')

fig.update_layout(
    title='Distribuição do Curve Number (CN 2022 ANA)',
    xaxis_title='Curve Number (CN)',
    yaxis_title='Frequência',
    height=450,
    showlegend=False
)
fig.show()

# %% cell 5
# Box plot
fig = go.Figure()

fig.add_trace(go.Box(
    y=otto['ANA_med'],
    name='CN 2022 ANA',
    marker_color='#2E86AB',
    boxpoints='outliers'
))

fig.update_layout(
    title='Box Plot do Curve Number',
    yaxis_title='Curve Number (CN)',
    height=400
)
fig.show()

# %% cell 6
# Converter para WGS84 para visualização
otto_wgs = otto.to_crs(epsg=4326)

# Centralizar no centroide da bacia
otto_proj = otto.to_crs(epsg=31983)  # UTM 23S para cálculo correto
centroid = otto_proj.dissolve().centroid.iloc[0]
centroid_wgs = gpd.GeoSeries([centroid], crs=31983).to_crs(4326).iloc[0]

# Mapa do CN
fig = go.Figure(go.Choroplethmap(
    geojson=json.loads(otto_wgs.geometry.to_json()),
    locations=otto_wgs.index,
    z=otto_wgs['ANA_med'],
    colorscale='RdYlGn_r',
    marker_opacity=0.8,
    marker_line_width=0.5,
    marker_line_color='gray',
    colorbar=dict(
        title=dict(text='CN', font=dict(size=14)),
        thickness=15
    )
))

fig.update_layout(
    title=dict(
        text=f'Curve Number (CN 2022 ANA) - Média Ponderada: {cn_pond:.1f}',
        x=0.5,
        font=dict(size=16)
    ),
    map=dict(
        style='white-bg',
        center=dict(lat=centroid_wgs.y, lon=centroid_wgs.x),
        zoom=8.5
    ),
    height=600,
    margin=dict(l=10, r=10, t=50, b=10)
)

fig.show()

# %% cell 7
# Função SCS
def calc_pe(P, S, lambda_ia=0.2):
    """Calcular precipitação efetiva usando SCS."""
    Ia = lambda_ia * S
    if P <= Ia:
        return 0
    return (P - Ia)**2 / (P - Ia + S)

# Simular para diferentes precipitações
precipitacoes = np.arange(0, 151, 5)
pe = [calc_pe(p, s_pond) for p in precipitacoes]

# Gráfico
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=precipitacoes, y=pe,
    name=f'CN = {cn_pond:.0f}',
    line=dict(color='#2E86AB', width=3)
))

fig.add_trace(go.Scatter(
    x=precipitacoes, y=precipitacoes,
    name='100% escoamento',
    line=dict(color='gray', dash='dot')
))

fig.update_layout(
    title='Curva de Escoamento SCS (CN 2022 ANA)',
    xaxis_title='Precipitação (mm)',
    yaxis_title='Precipitação Efetiva (mm)',
    height=450,
    legend=dict(x=0.02, y=0.98)
)
fig.show()

# %% cell 8
# Tabela de simulação para chuvas típicas
print('=' * 60)
print('SIMULAÇÃO SCS - CN 2022 ANA')
print('=' * 60)
print()
print(f'Parâmetros: CN = {cn_pond:.1f}, S = {s_pond:.1f} mm')
print()
print(f'{"P (mm)":<10} {"Ia (mm)":>10} {"Pe (mm)":>10} {"Coef.Esc.":>12}')
print('-' * 45)

Ia = 0.2 * s_pond
for P in [10, 20, 30, 50, 75, 100, 150]:
    Pe = calc_pe(P, s_pond)
    coef = Pe/P if P > 0 else 0
    print(f'{P:<10} {Ia:>10.1f} {Pe:>10.1f} {coef:>11.1%}')

print()
print('Interpretação:')
print(f'  → Ia = {Ia:.0f} mm (abstração inicial antes de gerar escoamento)')
print(f'  → Chuvas < {Ia:.0f} mm não geram escoamento direto')
print(f'  → Chuva de 50 mm gera ~{calc_pe(50, s_pond):.0f} mm de escoamento')

# %% cell 9
# Classificar CN
bins = [0, 50, 60, 70, 80, 90, 100]
labels = ['<50', '50-60', '60-70', '70-80', '80-90', '>90']

otto['cn_classe'] = pd.cut(otto['ANA_med'], bins=bins, labels=labels)

# Estatísticas por classe
stats = otto.groupby('cn_classe', observed=True).agg({
    'area_km2': ['count', 'sum']
})
stats.columns = ['n_otto', 'area_km2']
stats['pct_area'] = 100 * stats['area_km2'] / stats['area_km2'].sum()

print('Distribuição por classe de CN:')
print()
print(f'{"Classe":<10} {"N. Otto":>10} {"Área (km²)":>12} {"% Área":>10}')
print('-' * 45)
for classe in labels:
    if classe in stats.index:
        row = stats.loc[classe]
        print(f'{classe:<10} {row["n_otto"]:>10.0f} {row["area_km2"]:>12.1f} {row["pct_area"]:>9.1f}%')
    else:
        print(f'{classe:<10} {0:>10} {0:>12.1f} {0:>9.1f}%')

# %% cell 10
# Gráfico de barras
fig = go.Figure()

pct_values = [stats.loc[c, 'pct_area'] if c in stats.index else 0 for c in labels]

fig.add_trace(go.Bar(
    x=labels,
    y=pct_values,
    marker_color='#2E86AB',
    text=[f'{v:.1f}%' for v in pct_values],
    textposition='outside'
))

fig.update_layout(
    title='Distribuição de Área por Classe de CN',
    xaxis_title='Classe de CN',
    yaxis_title='Percentual da Área (%)',
    height=400,
    yaxis_range=[0, max(pct_values) * 1.15]
)
fig.show()

# %% cell 11
# Exportar CN 2022 processado
otto_export = otto[['cobacia', 'ANA_med', 'area_km2', 'S_mm', 'geometry']].copy()
otto_export = otto_export.rename(columns={
    'ANA_med': 'cn_2022',
    'S_mm': 's_mm'
})
otto_export.to_file(OUTPUT_DIR / 'ottobacias_cn_2022.gpkg', driver='GPKG')
print(f'Exportado: {OUTPUT_DIR / "ottobacias_cn_2022.gpkg"}')
print(f'  - {len(otto_export)} ottobacias')
print(f'  - Colunas: {list(otto_export.columns)}')

# %% cell 12
print('=' * 60)
print('RESUMO - CURVE NUMBER (CN 2022 ANA)')
print('=' * 60)
print()
print('CARACTERÍSTICAS DA BACIA:')
print(f'  - Área total: {otto["area_km2"].sum():.1f} km²')
print(f'  - Ottobacias: {len(otto)}')
print(f'  - Área média por ottobacia: {otto["area_km2"].mean():.1f} km²')
print()
print('PARÂMETROS SCS:')
print(f'  - CN médio (ponderado): {cn_pond:.1f}')
print(f'  - S médio: {s_pond:.1f} mm')
print(f'  - Ia médio: {0.2 * s_pond:.1f} mm')
print()
print('CLASSE DOMINANTE:')
classe_dom = stats['pct_area'].idxmax()
print(f'  - Classe {classe_dom}: {stats.loc[classe_dom, "pct_area"]:.1f}% da área')
print()
print('METODOLOGIA:')
print('  - Fonte: ANA (BHAE_CN-2022)')
print('  - Solos: Classificação Sartori et al. (2005)')
print('  - LULC: MapBiomas 2022')
print()
print('=' * 60)
