"""Code cells of the notebook 01_vazao_eda.ipynb, exported without outputs.

Provenance only: the code is kept as it was run (absolute paths and all).
Source notebook SHA-256: 6f461144358ac4838ba2f42d9dc207df9bd0be3bea13d0486ad0f9b2de316de2
"""

# %% cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Configurações
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.size'] = 11

DATA_DIR = Path('../data/raw')
AREA_KM2 = 3117  # Área de drenagem em km²
PERIODO_INICIO = '2021-01-01'
PERIODO_FIM = '2025-12-31'

print("Setup OK!")
print(f"Período de análise: {PERIODO_INICIO} a {PERIODO_FIM}")

# %% cell 2
# Carregar CSV com encoding correto
df_raw = pd.read_csv(
    DATA_DIR / 'vazao' / '58585000-MANUEL DUARTE.csv',
    sep=';',
    encoding='latin-1',
    decimal=',',
    low_memory=False
)

print(f"Registros brutos: {len(df_raw):,}")
print(f"Colunas: {list(df_raw.columns)}")
df_raw.head()

# %% cell 3
# Renomear e processar colunas
df_raw.columns = ['data', 'hora', 'chuva_mm', 'nivel_cm', 'vazao_m3s', 'extra']
df_raw = df_raw.drop(columns=['extra'])

# Criar datetime
df_raw['datetime'] = pd.to_datetime(df_raw['data'] + ' ' + df_raw['hora'], format='%d/%m/%Y %H:%M:%S')

# Converter valores numéricos
for col in ['chuva_mm', 'nivel_cm', 'vazao_m3s']:
    df_raw[col] = pd.to_numeric(df_raw[col].astype(str).str.strip(), errors='coerce')

# Definir índice e ordenar
df_raw = df_raw.set_index('datetime').sort_index()
df_raw = df_raw[['vazao_m3s', 'nivel_cm', 'chuva_mm']]

print(f"Período total disponível: {df_raw.index.min()} a {df_raw.index.max()}")
print(f"Frequência: 15 minutos")

# %% cell 4
# Calcular disponibilidade por ano (considerando registros esperados)
disp_data = []
for ano in sorted(df_raw.index.year.unique()):
    df_ano = df_raw[df_raw.index.year == ano]
    
    # Período esperado
    if ano == df_raw.index.year.min():
        inicio = df_ano.index.min()
    else:
        inicio = pd.Timestamp(f'{ano}-01-01')
    
    if ano == df_raw.index.year.max():
        fim = df_ano.index.max()
    else:
        fim = pd.Timestamp(f'{ano}-12-31 23:45:00')
    
    # Registros esperados (15min = 4 por hora)
    horas_periodo = (fim - inicio).total_seconds() / 3600
    esperado = int(horas_periodo * 4)
    validos = df_ano['vazao_m3s'].notna().sum()
    disp = 100 * validos / esperado if esperado > 0 else 0
    
    disp_data.append({'ano': ano, 'disponibilidade': disp, 'validos': validos, 'esperado': esperado})

disp_df = pd.DataFrame(disp_data)

# Definir cores: verde se >= 90%, laranja caso contrário
colors = ['#2ecc71' if d >= 90 else '#e67e22' for d in disp_df['disponibilidade']]

# Figura Plotly
fig = go.Figure()

fig.add_trace(go.Bar(
    x=disp_df['ano'],
    y=disp_df['disponibilidade'],
    marker_color=colors,
    text=[f'{d:.0f}%' for d in disp_df['disponibilidade']],
    textposition='outside',
    showlegend=False
))

fig.add_hline(y=90, line_dash='dash', line_color='green', annotation_text='90%')
fig.add_vrect(x0=2020.5, x1=2025.5, fillcolor='green', opacity=0.1, 
              annotation_text='Período selecionado')

fig.update_layout(
    title='Disponibilidade de Dados de Vazão por Ano',
    xaxis_title='Ano',
    yaxis_title='Disponibilidade (%)',
    yaxis=dict(range=[0, 118]),
    xaxis=dict(tickmode='linear', tick0=2013, dtick=1),
    height=400
)

fig.show()

# Tabela resumo
print("Disponibilidade por ano:")
for _, row in disp_df.iterrows():
    status = 'OK' if row['disponibilidade'] >= 90 else 'BAIXA' if row['disponibilidade'] >= 70 else 'RUIM'
    marker = ' <-- selecionado' if 2021 <= row['ano'] <= 2025 else ''
    print(f"  {int(row['ano'])}: {row['disponibilidade']:5.1f}% [{status}]{marker}")

# %% cell 5
# Filtrar período selecionado
df = df_raw[PERIODO_INICIO:PERIODO_FIM].copy()

print(f"Período filtrado: {df.index.min()} a {df.index.max()}")
print(f"Registros (15min): {len(df):,}")
print(f"NaN vazão: {df['vazao_m3s'].isna().sum():,} ({100*df['vazao_m3s'].isna().mean():.1f}%)")

# %% cell 6
# Reamostrar para horário
df_h = df[['vazao_m3s', 'nivel_cm', 'chuva_mm']].resample('h').mean()

# Criar índice completo
idx_completo = pd.date_range(start=PERIODO_INICIO, end=f'{PERIODO_FIM} 23:00:00', freq='h')
df_h = df_h.reindex(idx_completo)

print(f"Registros horários esperados: {len(idx_completo):,}")
print(f"NaN vazão: {df_h['vazao_m3s'].isna().sum():,} ({100*df_h['vazao_m3s'].isna().mean():.1f}%)")

# %% cell 7
# Analisar distribuição de gaps
df_h['is_nan'] = df_h['vazao_m3s'].isna()
df_h['gap_id'] = (df_h['is_nan'] != df_h['is_nan'].shift()).cumsum()

gaps = df_h[df_h['is_nan']].groupby('gap_id').size()

print("=" * 50)
print("ANÁLISE DE GAPS")
print("=" * 50)
print(f"Total de gaps: {len(gaps)}")
print(f"Gap médio: {gaps.mean():.1f} horas")
print(f"Gap máximo: {gaps.max()} horas ({gaps.max()/24:.1f} dias)")
print()

# Categorizar
print("Gaps por duração:")
print(f"  1 hora:      {(gaps == 1).sum():3d} gaps")
print(f"  2-6 horas:   {((gaps >= 2) & (gaps <= 6)).sum():3d} gaps")
print(f"  7-24 horas:  {((gaps > 6) & (gaps <= 24)).sum():3d} gaps")
print(f"  1-7 dias:    {((gaps > 24) & (gaps <= 168)).sum():3d} gaps")
print(f"  >7 dias:     {(gaps > 168).sum():3d} gaps")
print()

# Maiores gaps
print("Maiores gaps:")
for gap_id in gaps.nlargest(5).index:
    gap_data = df_h[df_h['gap_id'] == gap_id]
    inicio = gap_data.index.min()
    fim = gap_data.index.max()
    duracao = len(gap_data)
    print(f"  {inicio.strftime('%Y-%m-%d')} a {fim.strftime('%Y-%m-%d')} ({duracao}h = {duracao/24:.1f}d)")

# Limpar colunas auxiliares
df_h = df_h.drop(columns=['is_nan', 'gap_id'])

# %% cell 8
q = df['vazao_m3s'].dropna()
n = df['nivel_cm'].dropna()

print("=" * 50)
print("ESTATÍSTICAS DESCRITIVAS - VAZÃO (m³/s)")
print("=" * 50)

stats = {
    'Contagem': len(q),
    'Média': q.mean(),
    'Desvio Padrão': q.std(),
    'Mínimo': q.min(),
    'Q10': q.quantile(0.10),
    'Q25': q.quantile(0.25),
    'Mediana': q.median(),
    'Q75': q.quantile(0.75),
    'Q90': q.quantile(0.90),
    'Q95': q.quantile(0.95),
    'Máximo': q.max(),
    'CV (%)': 100 * q.std() / q.mean()
}

for k, v in stats.items():
    if 'Contagem' in k:
        print(f"{k:15s}: {v:,.0f}")
    else:
        print(f"{k:15s}: {v:.2f}")

# Vazão específica
q_esp = q.mean() / AREA_KM2 * 1000
print(f"\nVazão específica: {q_esp:.2f} L/s/km²")
print(f"Área de drenagem: {AREA_KM2:,} km²")

# %% cell 9
# Série temporal com matplotlib (mais leve que plotly para muitos pontos)
fig, ax = plt.subplots(figsize=(14, 5))

# Usar dados horários para plotar (mais leve)
ax.plot(df_h.index, df_h['vazao_m3s'], 'b-', linewidth=0.5, alpha=0.8)
ax.axhline(y=q.mean(), color='r', linestyle='--', label=f'Média = {q.mean():.1f} m³/s')

ax.set_xlabel('Data')
ax.set_ylabel('Vazão (m³/s)')
ax.set_title('Série Temporal de Vazão - Manuel Duarte (58585000) - 2021-2025')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% cell 10
# Série temporal por ano (subplots)
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=False)

for i, ano in enumerate(range(2021, 2026)):
    df_ano = df_h[df_h.index.year == ano]
    axes[i].plot(df_ano.index, df_ano['vazao_m3s'], 'b-', linewidth=0.5)
    axes[i].axhline(y=q.mean(), color='r', linestyle='--', alpha=0.5)
    axes[i].set_ylabel('Q (m³/s)')
    axes[i].set_title(f'{ano}')
    axes[i].set_ylim(0, q.max() * 1.1)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% cell 11
# Curva de permanência
q_valid = q.values
q_sorted = np.sort(q_valid)[::-1]
exceedance = np.arange(1, len(q_sorted) + 1) / len(q_sorted) * 100

# Vazões de referência
q95 = np.percentile(q_valid, 5)   # Q95 = 5% excedência
q50 = np.percentile(q_valid, 50)  # Q50 = 50% excedência

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=exceedance,
    y=q_sorted,
    mode='lines',
    name='Vazão',
    line=dict(color='blue', width=2)
))

fig.add_hline(y=q95, line_dash='dash', line_color='red', 
              annotation_text=f'Q95 = {q95:.1f} m³/s')
fig.add_hline(y=q50, line_dash='dash', line_color='green',
              annotation_text=f'Q50 = {q50:.1f} m³/s')

fig.update_layout(
    title='Curva de Permanência - Manuel Duarte (2021-2025)',
    xaxis_title='Permanência (%)',
    yaxis_title='Vazão (m³/s)',
    yaxis_type='log',
    yaxis=dict(range=[np.log10(10), np.log10(max(600, q.max()*1.1))]),
    height=500
)
fig.show()

print(f"Q95 (vazão mínima de referência): {q95:.1f} m³/s")
print(f"Q50 (vazão mediana): {q50:.1f} m³/s")

# %% cell 12
# Boxplot mensal
df['mes'] = df.index.month
meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
         'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

fig = px.box(
    df.reset_index(),
    x='mes',
    y='vazao_m3s',
    title='Sazonalidade da Vazão - Boxplot Mensal (2021-2025)',
    labels={'mes': 'Mês', 'vazao_m3s': 'Vazão (m³/s)'}
)
fig.update_xaxes(ticktext=meses, tickvals=list(range(1, 13)))
fig.update_layout(height=500)
fig.show()

# %% cell 13
# Média mensal
media_mensal = df.groupby('mes')['vazao_m3s'].mean()

fig = px.bar(
    x=meses,
    y=media_mensal.values,
    title='Vazão Média Mensal (2021-2025)',
    labels={'x': 'Mês', 'y': 'Vazão Média (m³/s)'}
)
fig.add_hline(y=q.mean(), line_dash='dash', line_color='red',
              annotation_text=f'Média = {q.mean():.1f} m³/s                      ')
fig.update_layout(height=400)
fig.show()

# Período chuvoso vs seco
q_chuvoso = media_mensal[[11, 12, 1, 2, 3]].mean()
q_seco = media_mensal[[6, 7, 8, 9]].mean()

print(f"Período chuvoso (Nov-Mar): {q_chuvoso:.1f} m³/s")
print(f"Período seco (Jun-Set): {q_seco:.1f} m³/s")
print(f"Razão chuvoso/seco: {q_chuvoso/q_seco:.1f}x")

# %% cell 14
# Top 10 maiores vazões
top_10 = df.nlargest(10, 'vazao_m3s')[['vazao_m3s', 'nivel_cm']]

print("=" * 50)
print("TOP 10 MAIORES VAZÕES (2021-2025)")
print("=" * 50)
for idx, row in top_10.iterrows():
    print(f"{idx}: {row['vazao_m3s']:.1f} m³/s, nível={row['nivel_cm']:.0f} cm")

# %% cell 15
# Maior evento de cheia
pico = df['vazao_m3s'].idxmax()
inicio = pico - pd.Timedelta('3 days')
fim = pico + pd.Timedelta('5 days')

evento = df.loc[inicio:fim]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=['Vazão', 'Nível'],
                    vertical_spacing=0.1)

fig.add_trace(
    go.Scatter(x=evento.index, y=evento['vazao_m3s'], name='Vazão', line_color='blue'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=evento.index, y=evento['nivel_cm'], name='Nível', line_color='green'),
    row=2, col=1
)

fig.update_layout(
    title=f'Maior Evento de Cheia no Período ({pico.strftime("%Y-%m-%d")})',
    height=600
)
fig.update_yaxes(title_text='Vazão (m³/s)', row=1, col=1)
fig.update_yaxes(title_text='Nível (cm)', row=2, col=1)
fig.show()

print(f"Pico de vazão: {df.loc[pico, 'vazao_m3s']:.1f} m³/s")
print(f"Data/hora: {pico}")

# %% cell 16
# Curva-chave observada
df_clean = df.dropna(subset=['nivel_cm', 'vazao_m3s'])

# Correlação
corr = df_clean['nivel_cm'].corr(df_clean['vazao_m3s'])
print(f"Correlação nível x vazão: {corr:.4f}")

# Amostrar para plotar
sample = df_clean.sample(min(10000, len(df_clean)), random_state=42)

fig = px.scatter(
    sample.reset_index(),
    x='nivel_cm',
    y='vazao_m3s',
    opacity=0.3,
    title=f'Relação Nível x Vazão (Curva-Chave) - r = {corr:.3f}',
    labels={'nivel_cm': 'Nível (cm)', 'vazao_m3s': 'Vazão (m³/s)'}
)
fig.update_layout(height=500)
fig.show()

# %% cell 17
# Salvar série horária para uso no modelo
output_dir = Path('../data/processed')
output_dir.mkdir(exist_ok=True)

df_h_export = df_h[['vazao_m3s', 'nivel_cm']].copy()
df_h_export.to_csv(output_dir / 'vazao_horaria.csv')

print(f"Arquivo salvo: {output_dir / 'vazao_horaria.csv'}")
print(f"Registros: {len(df_h_export):,}")
print(f"Período: {df_h_export.index.min()} a {df_h_export.index.max()}")
print(f"NaN: {df_h_export['vazao_m3s'].isna().sum():,} ({100*df_h_export['vazao_m3s'].isna().mean():.1f}%)")

# %% cell 18
print("="*60)
print("RESUMO - DADOS DE VAZÃO")
print("="*60)
print()
print(f"Estação:              Manuel Duarte (58585000)")
print(f"Área de drenagem:     {AREA_KM2:,} km²")
print()
print(f"Período selecionado:  {PERIODO_INICIO} a {PERIODO_FIM}")
print(f"Justificativa:        Série contínua com melhor disponibilidade")
print()
print(f"Frequência original:  15 minutos")
print(f"Frequência modelo:    1 hora")
print(f"Registros (15min):    {len(df):,}")
print(f"Registros (horário):  {len(df_h_export):,}")
print()
print(f"Vazão média:          {q.mean():.1f} m³/s")
print(f"Vazão mediana:        {q.median():.1f} m³/s")
print(f"Vazão mínima:         {q.min():.1f} m³/s")
print(f"Vazão máxima:         {q.max():.1f} m³/s")
print(f"Q95:                  {q95:.1f} m³/s")
print()
print(f"Vazão específica:     {q_esp:.2f} L/s/km²")
print(f"CV:                   {100*q.std()/q.mean():.1f}%")
print()
print(f"Falhas (horário):     {df_h_export['vazao_m3s'].isna().sum():,} ({100*df_h_export['vazao_m3s'].isna().mean():.1f}%)")
print(f"Preenchimento:        Não (manter integridade)")
print("="*60)
