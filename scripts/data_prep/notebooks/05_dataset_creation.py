"""Code cells of the notebook 05_dataset_creation.ipynb, exported without outputs.

Provenance only: the code is kept as it was run (absolute paths and all).
Source notebook SHA-256: e0825c8b4d1627485a22590368136a51af687b8a825737c7e6f617d6dfb06f33
"""

# %% cell 1
import pandas as pd
import numpy as np
import h5py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# Configurações
DATA_RAW = Path('../data/raw')
DATA_PROC = Path('../data/processed')
DATA_PROC.mkdir(exist_ok=True)

# Período de estudo
PERIODO_INICIO = '2021-01-01'
PERIODO_FIM = '2025-12-31'

# Divisão temporal
TRAIN_END = '2024-06-30'      # 70% ~ 3.5 anos
VAL_END = '2025-03-31'        # 15% ~ 9 meses
# TEST: 2025-04-01 a 2025-12-31 (15% ~ 9 meses)

print("Setup OK!")
print(f"Período: {PERIODO_INICIO} a {PERIODO_FIM}")
print(f"Train: {PERIODO_INICIO} a {TRAIN_END}")
print(f"Val: {TRAIN_END} a {VAL_END}")
print(f"Test: {VAL_END} a {PERIODO_FIM}")

# %% cell 2
# Carregar vazão horária (gerada no notebook 01)
df_vazao = pd.read_csv(
    DATA_PROC / 'vazao_horaria.csv',
    index_col=0,
    parse_dates=True
)

print(f"Vazão carregada: {len(df_vazao):,} registros")
print(f"Período: {df_vazao.index.min()} a {df_vazao.index.max()}")
print(f"NaN: {df_vazao['vazao_m3s'].isna().sum():,} ({100*df_vazao['vazao_m3s'].isna().mean():.1f}%)")
df_vazao.head()

# %% cell 3
# Carregar propriedades das ottobacias
# O arquivo ottobacias_tc.csv contém CN, área, Tc Base e Tc Manning para as 245 ottobacias

df_otto = pd.read_csv(DATA_PROC / 'ottobacias_tc.csv')

print(f"Ottobacias carregadas: {len(df_otto)}")
print(f"Área total: {df_otto['area_km2'].sum():.1f} km²")
print(f"\nColunas disponíveis: {list(df_otto.columns)}")

# Verificar campos necessários
required_cols = ['cobacia', 'area_km2', 'cn_ana', 'tc_base_h', 'tc_manning_h']
missing = [c for c in required_cols if c not in df_otto.columns]
if missing:
    raise ValueError(f"Colunas faltando: {missing}")
    
print("\n✅ Todas as colunas necessárias presentes!")
df_otto.head()

# %% cell 4
# Estatísticas das propriedades
print("="*50)
print("ESTATÍSTICAS DAS OTTOBACIAS")
print("="*50)

print(f"\nÁrea (km²):")
print(f"  Min: {df_otto['area_km2'].min():.2f}")
print(f"  Média: {df_otto['area_km2'].mean():.2f}")
print(f"  Max: {df_otto['area_km2'].max():.2f}")
print(f"  Total: {df_otto['area_km2'].sum():.1f}")

print(f"\nCN 2022 (cn_ana):")
print(f"  Min: {df_otto['cn_ana'].min():.1f}")
print(f"  Média: {df_otto['cn_ana'].mean():.1f}")
print(f"  Max: {df_otto['cn_ana'].max():.1f}")

print(f"\nTc Base (Maidment puro):")
print(f"  Min: {df_otto['tc_base_h'].min():.2f} h")
print(f"  Média: {df_otto['tc_base_h'].mean():.2f} h")
print(f"  Max: {df_otto['tc_base_h'].max():.2f} h")

print(f"\nTc Manning (com rugosidade LULC 2022):")
print(f"  Min: {df_otto['tc_manning_h'].min():.2f} h")
print(f"  Média: {df_otto['tc_manning_h'].mean():.2f} h")
print(f"  Max: {df_otto['tc_manning_h'].max():.2f} h")

# Fator de aumento médio
factor = df_otto['tc_manning_h'].mean() / df_otto['tc_base_h'].mean()
print(f"\nFator médio Tc Manning/Tc Base: {factor:.2f}x (+{100*(factor-1):.0f}%)")

# %% cell 5
# Verificar consistência dos dados
print("Verificação de consistência:")
print(f"  Ottobacias: {len(df_otto)}")
print(f"  NaN em cn_ana: {df_otto['cn_ana'].isna().sum()}")
print(f"  NaN em tc_base_h: {df_otto['tc_base_h'].isna().sum()}")
print(f"  NaN em tc_manning_h: {df_otto['tc_manning_h'].isna().sum()}")
print(f"  NaN em area_km2: {df_otto['area_km2'].isna().sum()}")

# Verificar valores razoáveis
assert df_otto['cn_ana'].between(30, 100).all(), "CN fora do range esperado"
assert df_otto['tc_base_h'].gt(0).all(), "Tc Base deve ser > 0"
assert df_otto['tc_manning_h'].gt(0).all(), "Tc Manning deve ser > 0"
assert df_otto['area_km2'].ge(0).all(), "Área deve ser >= 0"

print("\n✅ Todos os dados válidos!")

# %% cell 6
# Carregar precipitação por ottobacia (gerada no notebook 03)
# Pode estar em HDF5 ou CSV
precip_h5 = DATA_PROC / 'precipitation_distributed.h5'
precip_csv = DATA_PROC / 'precipitation_by_ottobacia.csv'

if precip_h5.exists():
    print("Carregando de HDF5...")
    with h5py.File(precip_h5, 'r') as f:
        precip_data = f['precipitation'][:]
        timestamps_raw = f['timestamps'][:]
        otto_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['otto_ids'][:]]
    
    # Converter timestamps - detectar formato
    # Se valores muito grandes (>1e15), são nanosegundos do pandas
    if timestamps_raw.max() > 1e15:
        print(f"  Timestamps em nanosegundos, convertendo...")
        precip_index = pd.to_datetime(timestamps_raw, unit='ns')
    elif timestamps_raw.max() > 1e12:
        print(f"  Timestamps em milissegundos, convertendo...")
        precip_index = pd.to_datetime(timestamps_raw, unit='ms')
    else:
        print(f"  Timestamps em segundos (Unix), convertendo...")
        precip_index = pd.to_datetime(timestamps_raw, unit='s')
    
    df_precip = pd.DataFrame(precip_data, index=precip_index, columns=otto_ids)

elif precip_csv.exists():
    print("Carregando de CSV...")
    df_precip = pd.read_csv(precip_csv, index_col=0, parse_dates=True)
else:
    raise FileNotFoundError("Nenhum arquivo de precipitação encontrado!")

print(f"Precipitação carregada: {df_precip.shape}")
print(f"  Timesteps: {len(df_precip):,}")
print(f"  Ottobacias: {df_precip.shape[1]}")
print(f"Período: {df_precip.index.min()} a {df_precip.index.max()}")
print(f"NaN: {df_precip.isna().sum().sum():,}")
df_precip.head()

# %% cell 7
# IDs das ottobacias na ordem da precipitação
otto_ids_precip = df_precip.columns.tolist()
n_otto = len(otto_ids_precip)
print(f"Ottobacias na precipitação: {n_otto}")

# Converter IDs para string
df_otto['cobacia'] = df_otto['cobacia'].astype(str)
otto_ids_str = [str(x) for x in otto_ids_precip]

# Verificar match
set_precip = set(otto_ids_str)
set_otto = set(df_otto['cobacia'].tolist())
match = len(set_precip & set_otto)
print(f"Match com propriedades: {match}/{n_otto} ({100*match/n_otto:.1f}%)")

# %% cell 8
# Criar dataframe de propriedades na ordem da precipitação
df_props = pd.DataFrame({'otto_id': otto_ids_str})

# Merge com propriedades (nova estrutura v2)
df_props = df_props.merge(
    df_otto[['cobacia', 'cn_ana', 'area_km2', 'tc_base_h', 'tc_manning_h']].rename(columns={
        'cobacia': 'otto_id',
        'cn_ana': 'cn_2022',  # Renomear para clareza
    }),
    on='otto_id',
    how='left'
)

# Verificar NaN
print("NaN por coluna:")
print(df_props.isna().sum())

# Preencher NaN com valores médios se necessário (não deve acontecer se IDs batem)
for col, default_source in [('cn_2022', 'cn_ana'), ('area_km2', 'area_km2'), 
                             ('tc_base_h', 'tc_base_h'), ('tc_manning_h', 'tc_manning_h')]:
    if df_props[col].isna().any():
        median_val = df_otto[default_source if default_source in df_otto.columns else col].median()
        n_nan = df_props[col].isna().sum()
        df_props[col].fillna(median_val, inplace=True)
        print(f"\n⚠️ {n_nan} {col} NaN preenchidos com mediana: {median_val:.2f}")

print(f"\n✅ Propriedades finais: {len(df_props)} ottobacias")
print(f"   Área total: {df_props['area_km2'].sum():.1f} km²")
print(f"   CN 2022 médio: {df_props['cn_2022'].mean():.1f}")
print(f"   Tc Base médio: {df_props['tc_base_h'].mean():.2f} h")
print(f"   Tc Manning médio: {df_props['tc_manning_h'].mean():.2f} h")
df_props.head()

# %% cell 9
# Extrair arrays (nova estrutura v2)
area_km2 = df_props['area_km2'].values
cn_2022 = df_props['cn_2022'].values
tc_base_h = df_props['tc_base_h'].values
tc_manning_h = df_props['tc_manning_h'].values

print(f"Arrays criados:")
print(f"  area_km2:     {area_km2.shape}, total = {area_km2.sum():.1f} km²")
print(f"  cn_2022:      {cn_2022.shape}, média = {cn_2022.mean():.1f}")
print(f"  tc_base_h:    {tc_base_h.shape}, média = {tc_base_h.mean():.2f} h, max = {tc_base_h.max():.2f} h")
print(f"  tc_manning_h: {tc_manning_h.shape}, média = {tc_manning_h.mean():.2f} h, max = {tc_manning_h.max():.2f} h")

# %% cell 10
# Criar índice temporal comum
idx_comum = df_vazao.index.intersection(df_precip.index)
idx_comum = idx_comum[(idx_comum >= PERIODO_INICIO) & (idx_comum <= PERIODO_FIM)]

print(f"Índice comum: {len(idx_comum):,} timesteps")
print(f"Período: {idx_comum.min()} a {idx_comum.max()}")

# Alinhar dados
vazao = df_vazao.loc[idx_comum, 'vazao_m3s'].values
precip = df_precip.loc[idx_comum].values
timestamps = idx_comum.astype(np.int64) // 10**9  # Unix timestamp

print(f"\nShapes:")
print(f"  Vazão: {vazao.shape}")
print(f"  Precipitação: {precip.shape}")
print(f"  Timestamps: {timestamps.shape}")

# %% cell 11
# Índices de divisão
train_end_ts = pd.Timestamp(TRAIN_END)
val_end_ts = pd.Timestamp(VAL_END)

train_mask = idx_comum <= train_end_ts
val_mask = (idx_comum > train_end_ts) & (idx_comum <= val_end_ts)
test_mask = idx_comum > val_end_ts

print("Divisão temporal:")
print(f"  Train: {train_mask.sum():,} ({100*train_mask.mean():.1f}%)")
print(f"  Val:   {val_mask.sum():,} ({100*val_mask.mean():.1f}%)")
print(f"  Test:  {test_mask.sum():,} ({100*test_mask.mean():.1f}%)")

# Separar dados
splits = {
    'train': {
        'precip': precip[train_mask],
        'vazao': vazao[train_mask],
        'timestamps': timestamps[train_mask]
    },
    'val': {
        'precip': precip[val_mask],
        'vazao': vazao[val_mask],
        'timestamps': timestamps[val_mask]
    },
    'test': {
        'precip': precip[test_mask],
        'vazao': vazao[test_mask],
        'timestamps': timestamps[test_mask]
    }
}

# %% cell 12
# Visualizar divisão
fig = go.Figure()

# Train
fig.add_trace(go.Scatter(
    x=idx_comum[train_mask],
    y=vazao[train_mask],
    mode='lines',
    name='Train',
    line=dict(color='blue', width=0.9)
))

# Val
fig.add_trace(go.Scatter(
    x=idx_comum[val_mask],
    y=vazao[val_mask],
    mode='lines',
    name='Validation',
    line=dict(color='orange', width=0.9)
))

# Test
fig.add_trace(go.Scatter(
    x=idx_comum[test_mask],
    y=vazao[test_mask],
    mode='lines',
    name='Test',
    line=dict(color='green', width=0.9)
))

fig.add_vline(x=train_end_ts, line_dash='dash', line_color='red')
fig.add_vline(x=val_end_ts, line_dash='dash', line_color='red')

fig.update_layout(
    title='Divisão Temporal do Dataset',
    xaxis_title='Data',
    yaxis_title='Vazão (m³/s)',
    height=400
)
fig.show()

# %% cell 13
print("="*70)
print("ESTATÍSTICAS POR SPLIT")
print("="*70)

for split_name, data in splits.items():
    q = data['vazao']
    p = data['precip']
    q_valid = q[~np.isnan(q)]
    
    print(f"\n{split_name.upper()}:")
    print(f"  Timesteps: {len(q):,}")
    print(f"  NaN vazão: {np.isnan(q).sum():,} ({100*np.isnan(q).mean():.1f}%)")
    print(f"  Vazão média: {q_valid.mean():.1f} m³/s")
    print(f"  Vazão max: {q_valid.max():.1f} m³/s")
    print(f"  Precip média: {p.mean():.3f} mm/h")
    print(f"  Precip max: {p.max():.1f} mm/h")

# %% cell 14
# Comparar distribuições
fig = make_subplots(rows=1, cols=2, subplot_titles=['Vazão', 'Precipitação Média'])

for split_name, color in [('train', 'blue'), ('val', 'orange'), ('test', 'green')]:
    q = splits[split_name]['vazao']
    q_valid = q[~np.isnan(q)]
    
    fig.add_trace(
        go.Histogram(x=q_valid, name=split_name, opacity=0.6,
                    marker_color=color, nbinsx=50),
        row=1, col=1
    )
    
    p_mean = splits[split_name]['precip'].mean(axis=1)
    fig.add_trace(
        go.Histogram(x=p_mean[p_mean > 0], name=split_name, opacity=0.6,
                    marker_color=color, nbinsx=50, showlegend=False),
        row=1, col=2
    )

fig.update_layout(height=400, title='Distribuições por Split', barmode='overlay')
fig.update_xaxes(title_text='Vazão (m³/s)', row=1, col=1)
fig.update_xaxes(title_text='Precipitação (mm/h)', row=1, col=2)
fig.show()

# %% cell 15
# Visualizar propriedades (nova estrutura v2)
fig = make_subplots(rows=2, cols=2, subplot_titles=[
    'Área (km²)', 'CN 2022', 'Tc Base (h)', 'Tc Manning (h)'
])

fig.add_trace(go.Histogram(x=area_km2, nbinsx=30, marker_color='blue'), row=1, col=1)
fig.add_trace(go.Histogram(x=cn_2022, nbinsx=30, marker_color='green'), row=1, col=2)
fig.add_trace(go.Histogram(x=tc_base_h, nbinsx=30, marker_color='orange'), row=2, col=1)
fig.add_trace(go.Histogram(x=tc_manning_h, nbinsx=30, marker_color='red'), row=2, col=2)

fig.update_layout(height=500, title='Distribuição das Propriedades das Ottobacias (v2)', showlegend=False)
fig.show()

# %% cell 16
output_path = DATA_PROC / 'dataset_v2.h5'

with h5py.File(output_path, 'w') as f:
    # Metadados
    f.attrs['created'] = datetime.now().isoformat()
    f.attrs['version'] = '2.0'
    f.attrs['periodo_inicio'] = PERIODO_INICIO
    f.attrs['periodo_fim'] = PERIODO_FIM
    f.attrs['n_ottobacias'] = n_otto
    f.attrs['area_total_km2'] = float(area_km2.sum())
    f.attrs['description'] = 'Dataset v2 com Tc Base (Maidment) e Tc Manning (LULC 2022)'
    
    # Propriedades das ottobacias (NOVA ESTRUTURA)
    otto_grp = f.create_group('ottobacia')
    otto_grp.create_dataset('area_km2', data=area_km2.astype(np.float32))
    otto_grp.create_dataset('cn_2022', data=cn_2022.astype(np.float32))  # CN ANA 2022
    otto_grp.create_dataset('tc_base_h', data=tc_base_h.astype(np.float32))  # Tc Maidment puro
    otto_grp.create_dataset('tc_manning_h', data=tc_manning_h.astype(np.float32))  # Tc com Manning
    
    # Atributos das ottobacias
    otto_grp.attrs['cn_source'] = 'BHAE_CN-2022 ANA'
    otto_grp.attrs['tc_base_source'] = 'Maidment (1996) - topografia pura'
    otto_grp.attrs['tc_manning_source'] = 'Maidment (1996) + Manning (LULC MapBiomas 2022)'
    
    # Dados por split
    for split_name, data in splits.items():
        grp = f.create_group(split_name)
        grp.create_dataset('precipitation', data=data['precip'].astype(np.float32),
                          compression='gzip', compression_opts=4)
        grp.create_dataset('streamflow', data=data['vazao'].astype(np.float32),
                          compression='gzip', compression_opts=4)
        grp.create_dataset('timestamps', data=data['timestamps'])
        
        grp.attrs['n_samples'] = len(data['vazao'])
        grp.attrs['start'] = datetime.fromtimestamp(data['timestamps'][0]).isoformat()
        grp.attrs['end'] = datetime.fromtimestamp(data['timestamps'][-1]).isoformat()

print(f"✅ Dataset v2 salvo: {output_path}")
print(f"   Tamanho: {output_path.stat().st_size / 1e6:.1f} MB")

# %% cell 17
# Verificar estrutura do dataset v2
def print_h5_structure(h5_file, indent=0):
    for key in h5_file.keys():
        item = h5_file[key]
        prefix = "  " * indent
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/")
            # Imprimir atributos do grupo
            if item.attrs:
                for ak, av in item.attrs.items():
                    print(f"{prefix}  @{ak}: {av}")
            print_h5_structure(item, indent + 1)
        else:
            print(f"{prefix}{key}: {item.shape} {item.dtype}")

with h5py.File(output_path, 'r') as f:
    print("Estrutura do Dataset v2:")
    print("="*50)
    print_h5_structure(f)
    print(f"\nAtributos globais:")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")

# %% cell 18
# Teste de leitura do dataset v2
with h5py.File(output_path, 'r') as f:
    # Carregar um batch de teste
    p_test = f['test/precipitation'][:100]
    q_test = f['test/streamflow'][:100]
    
    # Carregar propriedades (NOVA ESTRUTURA)
    cn = f['ottobacia/cn_2022'][:]
    tc_base = f['ottobacia/tc_base_h'][:]
    tc_manning = f['ottobacia/tc_manning_h'][:]
    area = f['ottobacia/area_km2'][:]
    
print("✅ Teste de leitura OK!")
print(f"  Precipitação shape: {p_test.shape}")
print(f"  Vazão shape: {q_test.shape}")
print(f"  CN 2022 shape: {cn.shape}, média = {cn.mean():.1f}")
print(f"  Tc Base shape: {tc_base.shape}, média = {tc_base.mean():.2f} h")
print(f"  Tc Manning shape: {tc_manning.shape}, média = {tc_manning.mean():.2f} h")
print(f"  Área shape: {area.shape}, total = {area.sum():.1f} km²")

# %% cell 19
print("="*60)
print("RESUMO - DATASET v2 (TTD-SCS-LSTM)")
print("="*60)
print()
print(f"Arquivo:              {output_path.name}")
print(f"Versão:               2.0")
print(f"Tamanho:              {output_path.stat().st_size / 1e6:.1f} MB")
print()
print(f"Período:              {PERIODO_INICIO} a {PERIODO_FIM}")
print(f"Resolução temporal:   1 hora")
print()
print(f"Ottobacias:           {n_otto}")
print(f"Área total:           {area_km2.sum():.1f} km²")
print()
print("Divisão temporal:")
for split_name, data in splits.items():
    n = len(data['vazao'])
    pct = 100 * n / len(vazao)
    print(f"  {split_name:5s}: {n:,} amostras ({pct:.1f}%)")
print()
print("Propriedades das Ottobacias (NOVA ESTRUTURA v2):")
print(f"  cn_2022:        ({n_otto},) - CN ANA 2022")
print(f"                  Média: {cn_2022.mean():.1f}, Range: [{cn_2022.min():.1f}, {cn_2022.max():.1f}]")
print(f"  tc_base_h:      ({n_otto},) - Tc Maidment puro")
print(f"                  Média: {tc_base_h.mean():.2f}h, Max: {tc_base_h.max():.2f}h")
print(f"  tc_manning_h:   ({n_otto},) - Tc com rugosidade Manning")
print(f"                  Média: {tc_manning_h.mean():.2f}h, Max: {tc_manning_h.max():.2f}h")
print(f"  area_km2:       ({n_otto},) - Área")
print()
print("Séries temporais:")
print(f"  precipitation:  (T, {n_otto}) mm/h")
print(f"  streamflow:     (T,) m³/s")
print(f"  timestamps:     (T,) Unix timestamp")
print()
print("="*60)
print("Modelos que usarão este dataset:")
print("  1. LSTM (baseline)")
print("  2. LSTM + TTD(Tc Base)")
print("  3. LSTM + TTD(Tc Manning)")
print("  4. LSTM + TTD(Tc Base) + SCS")
print("  5. LSTM + TTD(Tc Manning) + SCS")
print("="*60)
