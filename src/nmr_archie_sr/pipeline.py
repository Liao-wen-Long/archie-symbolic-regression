"""Train on grouped partitions, serialize models, and evaluate held-out cores."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd

from . import __version__
from .data import split_by_core, validate_data
from .model import ExponentModel, SearchConfig, StageTrainer


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')


def predict(frame, m_model, n_model):
    """Deployment uses phi, Rw, Rt and pore fractions; R0/Sw are optional diagnostics."""
    frame = validate_data(frame, prediction_only=True)
    m, m_clipped = m_model.predict(frame)
    n, n_clipped = n_model.predict(frame)
    log_rt = np.log(frame.Rt.to_numpy())
    log_r0 = np.log(frame.Rw.to_numpy()) - m * np.log(frame.phi.to_numpy())
    baseline_log_r0 = np.log(frame.Rw.to_numpy()) - 2.05 * np.log(frame.phi.to_numpy())
    result = pd.DataFrame({'core_id': frame.core_id, 'm_predicted': m, 'n_predicted': n,
                           'Sw_predicted': np.exp(np.minimum(log_r0 - log_rt, 0) / n),
                           'Sw_baseline': np.exp(np.minimum(baseline_log_r0 - log_rt, 0) / 2.05)})
    if 'R0' in frame:
        measured = np.log(frame.R0.to_numpy()) - log_rt
        result['Sw_n_only'] = np.exp(np.minimum(measured, 0) / n)
        result['Sw_n_only_baseline'] = np.exp(np.minimum(measured, 0) / 2.05)
    if 'Sw' in frame:
        result['Sw_observed'] = frame.Sw.to_numpy()
    return result, {'m_clipped_rows': m_clipped, 'n_clipped_rows': n_clipped}


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = prediction - truth
    sst = float(np.sum((truth - truth.mean()) ** 2))
    return {'rows': len(truth), 'mae_fraction': float(np.mean(np.abs(error))),
            'mae_percentage_points': float(100 * np.mean(np.abs(error))),
            'rmse_fraction': float(np.sqrt(np.mean(error ** 2))),
            'mean_relative_error_percent': float(100 * np.mean(np.abs(error) / truth)),
            'r2': float(1 - np.sum(error ** 2) / sst) if sst > 1e-16 else None}


def train(frame, output, config=None, seed=42, split_seed=42, source_sha256=None):
    frame = validate_data(frame)
    config = config or SearchConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    partitions, manifest = split_by_core(frame, split_seed)
    started = time.perf_counter()
    trainers = {stage: StageTrainer(partitions['train'], partitions['validation'], stage,
                                    config, seed + offset)
                for stage, offset in (('m', 0), ('n', 1))}
    models = {stage: trainer.search() for stage, trainer in trainers.items()}
    payload = {'schema_version': 1, 'software_version': __version__,
               'operator_semantics': {'pdiv': 'a / signed_max_abs(b, 1e-12)',
                                      'pexp': 'exp(clip(a,-20,20))',
                                      'plog': 'log(max(abs(a),1e-12))',
                                      'psqrt': 'sqrt(max(a,0))'},
               'saturation_rule': 'exp(min(log(Rw)-m*log(phi)-log(Rt),0)/n)',
               'm': models['m'].to_dict(), 'n': models['n'].to_dict()}
    write_json(output / 'model.json', payload)
    write_json(output / 'split_manifest.json', manifest)
    report, predictions = {}, []
    for name, subset in partitions.items():
        predicted, clipped = predict(subset, models['m'], models['n'])
        predicted['partition'] = name
        predictions.append(predicted)
        partial = subset.Sw.to_numpy() < .999
        report[name] = {'cores': len(manifest[name]), 'all_rows': len(subset), **clipped,
                        'evaluation_scope': 'Partial-saturation rows only; independent cores in test.',
                        'deployment_learned': metrics(subset.Sw.to_numpy()[partial], predicted.Sw_predicted.to_numpy()[partial]),
                        'deployment_baseline_m_n_2_05': metrics(subset.Sw.to_numpy()[partial], predicted.Sw_baseline.to_numpy()[partial]),
                        'measured_R0_diagnostic_learned': metrics(subset.Sw.to_numpy()[partial], predicted.Sw_n_only.to_numpy()[partial]),
                        'measured_R0_diagnostic_baseline_n_2_05': metrics(subset.Sw.to_numpy()[partial], predicted.Sw_n_only_baseline.to_numpy()[partial])}
    pd.concat(predictions, ignore_index=True).to_csv(output / 'predictions.csv', index=False)
    write_json(output / 'metrics.json', report)
    write_json(output / 'search_history.json', {k: t.history for k, t in trainers.items()})
    origin = ('SYNTHETIC' if 'data_origin' in frame and frame.data_origin.eq('SYNTHETIC').all()
              else 'USER_SUPPLIED_NOT_VERIFIED')
    canonical_hash = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    write_json(output / 'run_info.json', {'software_version': __version__, 'data_origin': origin,
                'config': asdict(config), 'search_seed': seed, 'split_seed': split_seed,
                'source_file_sha256': source_sha256, 'normalized_data_sha256': canonical_hash,
                'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'elapsed_seconds': time.perf_counter() - started,
                'unique_expressions_scored': {k: len(t.cache) for k, t in trainers.items()},
                'warning': 'Synthetic examples demonstrate execution only; they do not reproduce manuscript empirical results.'})
    return report


def load_models(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError("Unsupported model schema.")
    return ExponentModel.from_dict(data['m']), ExponentModel.from_dict(data['n'])
