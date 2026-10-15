"""Audita predições existentes GR4H na máscara horária neural, somente em CPU."""
from __future__ import annotations

import ast
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(r'D:\TTD_SCS_LSTM')
SKILL = ROOT / 'ablacao_skill'
LEGACY = ROOT / 'revisao2_paper1'
OUTPUT = Path(__file__).resolve().parents[1] / 'artigo_simulacao/revisao_codex_2026-09-12/gr4h_mascara_comum'
MODELS = ['LSTM', 'LSTM_Lumped', 'LSTM_DUH_Topmodel_PeOnly']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_functions(path, names):
    """Executa somente as funções matemáticas verificadas, sem importar train."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError(f'Função ausente em {path}')
    namespace = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)  # noqa: S102 somente funções locais selecionadas
    return namespace


def dates_summary(timestamps):
    return {'count': len(timestamps), 'first_utc': timestamps[0].isoformat() if len(timestamps) else None,
            'last_utc': timestamps[-1].isoformat() if len(timestamps) else None}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {'created_at': datetime.now().astimezone().isoformat(), 'status': 'running',
              'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                           'pandas': pd.__version__, 'h5py': h5py.__version__},
              'protocol': 'CPU, predições existentes, sem retreino ou recalibração, máscara neural invariável',
              'input_hashes_sha256': {}, 'original_reproduction': {}, 'checks': {}}
    train = SKILL / 'scripts/train.py'
    consolidate = SKILL / 'scripts/consolida_sim.py'
    metrics_function = load_functions(train, ['compute_metrics'])['compute_metrics']
    fdc = load_functions(consolidate, ['_fdc', '_lognse'])

    def metrics(pred, target):
        if not np.isfinite(pred).all() or not np.isfinite(target).all():
            raise RuntimeError('Métrica recebeu valores não finitos')
        out = {key: float(value) for key, value in metrics_function(pred, target).items()}
        out['fhv'], out['flv'] = fdc['_fdc'](pred, target)
        out['lognse'] = fdc['_lognse'](pred, target)
        return out

    def record(path):
        report['input_hashes_sha256'][str(path)] = sha256(path)

    try:
        h5path = SKILL / 'data/dataset_58585000_telem.h5'
        for path in [Path(__file__).resolve(), h5path, train, consolidate, LEGACY / 'run_gr4h_airgr.R', LEGACY / 'run_gr4h_airgr_kge.R', LEGACY / 'serie_horaria_gr4h.csv']:
            record(path)
        frames = {}
        for calibration, suffix in [('NSE', ''), ('KGE', '_kge')]:
            csv = LEGACY / ('gr4h_sim_teste' + suffix + '.csv')
            jsonpath = LEGACY / ('gr4h_metricas_paper' + suffix + '.json')
            record(csv)
            record(jsonpath)
            frame = pd.read_csv(csv)
            frame['timestamp_utc'] = pd.to_datetime(frame['Dates'], utc=True, errors='raise')
            if frame['timestamp_utc'].duplicated().any() or not frame['timestamp_utc'].is_monotonic_increasing:
                raise RuntimeError(f'Timestamps duplicados ou desordenados em {csv}')
            frame = frame.set_index('timestamp_utc')
            valid = np.isfinite(frame.Qsim_mm) & np.isfinite(frame.Qobs_mm)
            before = metrics(frame.Qsim_mm.to_numpy()[valid], frame.Qobs_mm.to_numpy()[valid])
            original = json.loads(jsonpath.read_text(encoding='utf-8'))
            deltas = {key: before[key] - original[key] for key in original if key != 'n_horas_validas'}
            if any(abs(value) > 1e-7 for value in deltas.values()):
                raise RuntimeError(f'Métricas originais não reproduzidas: {calibration}, deltas={deltas}')
            if 'n_horas_validas' in original and int(valid.sum()) != original['n_horas_validas']:
                raise RuntimeError('Contagem original diverge do JSON')
            report['original_reproduction'][calibration] = {'unit': 'mm/h', 'metrics': before,
                  'delta_against_json': deltas, 'tolerance_absolute': 1e-7,
                  'all_csv': dates_summary(frame.index), 'finite_csv': dates_summary(frame.index[valid])}
            frames[calibration] = frame
        with h5py.File(h5path, 'r') as handle:
            flow = handle['test/streamflow'][:]
            precip = handle['test/precipitation'][:]
            timestamps = pd.to_datetime(handle['test/timestamps'][:], unit='s', utc=True)
            area = float(handle.attrs['area_total_km2'])
        if len(set(timestamps)) != len(timestamps):
            raise RuntimeError('Timestamps H5 duplicados')
        # Replica literalmente a regra AblationDataset._create_valid_indices.
        starts = []
        rejected_target, rejected_precip = [], []
        for index in range(len(flow) - 240 - 1 + 1):
            if np.isnan(flow[index + 240:index + 241]).any():
                rejected_target.append(index + 240)
                continue
            if np.isnan(precip[index:index + 240]).any():
                rejected_precip.append(index + 240)
                continue
            starts.append(index)
        indices = np.array(starts, dtype=np.int64) + 240
        target = np.asarray(flow[indices], dtype=np.float32).reshape(-1)
        if not np.isfinite(target).all():
            raise RuntimeError('Regra original NaN admitiu alvo infinito')
        neural_dates = timestamps[indices]
        report['mask'] = {'lookback': 240, 'horizon': 1, 'test_h5': dates_summary(timestamps),
                          'neural': dates_summary(neural_dates), 'excluded_initial_lookback': 240,
                          'rejected_target_nan': len(rejected_target), 'rejected_precip_window_nan': len(rejected_precip),
                          'rule': 'NaN no target rejeita; NaN em qualquer precipitação da janela de 240 horas rejeita'}
        equality = []
        neural_nse_check = None
        for seed in range(42, 52):
            for model in MODELS:
                modelroot = SKILL / 'outputs/grade_sim' / f'seed{seed}' / model
                candidates = list(modelroot.glob('**/predictions.npz'))
                if len(candidates) != 1:
                    raise RuntimeError(f'Esperado um NPZ em {modelroot}, achei {len(candidates)}')
                npzpath = candidates[0]
                resultpath = npzpath.parent / 'results.json'
                record(npzpath)
                record(resultpath)
                result = json.loads(resultpath.read_text(encoding='utf-8'))
                if result['config']['lookback'] != 240 or result['config']['horizon'] != 1:
                    raise RuntimeError(f'Configuração incompatível em {resultpath}')
                with np.load(npzpath) as data:
                    observed = data['target'].reshape(-1)
                    predicted = data['pred'].reshape(-1)
                    if observed.dtype != np.float32 or not np.array_equal(observed, target):
                        raise RuntimeError(f'Alvo NPZ não é exatamente o H5 float32: {npzpath}')
                    if not np.isfinite(predicted).all():
                        raise RuntimeError(f'Predição neural não finita: {npzpath}')
                    if seed == 42 and model == 'LSTM':
                        computed = float(metrics_function(predicted, observed)['nse'])
                        stored = result['test_metrics']['nse']
                        if abs(computed - stored) > 1e-7:
                            raise RuntimeError('NSE neural não reproduzido contra results.json')
                        neural_nse_check = {'computed': computed, 'stored': stored, 'delta': computed - stored}
                equality.append({'seed': seed, 'model': model, 'count': len(observed), 'target_float32_exact': True})
        report['checks']['npz_target_equality'] = equality
        report['checks']['neural_seed42_NSE_reproduction'] = neural_nse_check
        # R documenta 3117 km²; CSV Qobs_mm tem resolução de 1e-6 mm/h.
        # A inversão principal preserva a transformação que gerou Qmm para o R.
        # A área exata H5 aparece somente como sensibilidade explícita à reescala.
        factor = 3117.0 / 3.6
        exact_factor = area / 3.6
        report['units'] = {'area_exact_h5_km2': area, 'area_historical_export_km2': 3117.0,
                           'area_primary_inverse_export_km2': 3117.0,
                           'mm_per_hour_to_m3_per_second_factor': factor,
                           'sensitivity_exact_h5_factor': exact_factor,
                           'FLV_original': 'mm/h, eps=1e-3', 'FLV_common': 'm3/s, eps=1e-3',
                           'target_tolerance': 'abs(delta) <= 0.5e-6*3117/3.6 + 2e-5 m3/s (resolução CSV e float32)',
                           'common_observation': 'H5 test streamflow convertido float32, idêntico aos 30 NPZ',
                           'conversion_decision': 'Consenso principal e parecer independente: 3117 inverte Qmm usado no R, preservando predições existentes; área H5 exata reescala e fica como sensibilidade'}
        common = pd.DataFrame({'timestamp_utc': neural_dates, 'Qobs_m3s': target})
        report['common_metrics'] = {}
        for calibration, frame in frames.items():
            if not neural_dates.isin(frame.index).all():
                raise RuntimeError(f'GR4H {calibration} não cobre toda máscara neural, sem reduzir máscara')
            joined = frame.loc[neural_dates]
            if not np.isfinite(joined[['Qsim_mm', 'Qobs_mm']].to_numpy()).all():
                raise RuntimeError(f'GR4H {calibration} tem valores não finitos na máscara neural')
            converted_target = joined.Qobs_mm.to_numpy() * factor
            tolerance = np.full(len(target), .5e-6 * factor + 2e-5)
            delta = np.abs(converted_target - target)
            if not np.all(delta <= tolerance):
                raise RuntimeError(f'Alvo CSV/H5 fora da tolerância em {calibration}: max={delta.max()}')
            prediction = joined.Qsim_mm.to_numpy() * factor
            report['checks'][calibration + '_join'] = {'timestamps_unique': True, 'full_neural_coverage': True,
                 'finite_all_neural_hours': True, 'target_max_abs_error_m3s': float(delta.max()),
                 'target_max_error_over_tolerance': float(np.max(delta / tolerance))}
            allvalid = np.isfinite(frame.Qsim_mm) & np.isfinite(frame.Qobs_mm)
            report['common_metrics'][calibration] = {'unit': 'm3/s', 'metrics': metrics(prediction, target.astype(np.float64)),
                  'before_all_valid_m3s': metrics(frame.Qsim_mm.to_numpy()[allvalid] * factor, frame.Qobs_mm.to_numpy()[allvalid] * factor),
                  'after': dates_summary(neural_dates),
                  'excluded_valid_csv_hours': dates_summary(frame.index[allvalid & ~frame.index.isin(neural_dates)]),
                  'exact_h5_area_sensitivity': {
                      'area_km2': area, 'interpretation': 'Reescala explícita das predições existentes, não é o resultado principal',
                      'metrics': metrics(joined.Qsim_mm.to_numpy() * exact_factor, target.astype(np.float64))}}
            common['GR4H_' + calibration + '_m3s'] = prediction
            excluded = frame.loc[allvalid & ~frame.index.isin(neural_dates)].copy()
            excluded.to_csv(OUTPUT / ('horas_validas_excluidas_GR4H_' + calibration + '.csv'))
        common.to_csv(OUTPUT / 'predicoes_mascara_comum.csv', index=False)
        report['status'] = 'completed_pending_independent_ratification'
    except Exception as error:
        report['status'] = 'blocked'
        report['error'] = str(error)
        raise
    finally:
        report['finished_at'] = datetime.now().astimezone().isoformat()
        (OUTPUT / 'auditoria_gr4h.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Relatório registrado: ' + str(OUTPUT / 'auditoria_gr4h.json'))


if __name__ == '__main__':
    main()
