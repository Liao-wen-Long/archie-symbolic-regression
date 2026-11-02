"""Input validation and one shared, core-grouped train/validation/test split."""
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ("core_id", "phi", "p_small", "p_medium", "Rw", "R0", "Rt", "Sw")
ALIASES = {"孔隙度": "phi", "小孔%": "p_small", "中孔%": "p_medium"}


def validate_data(frame: pd.DataFrame, units="fraction", prediction_only=False) -> pd.DataFrame:
    data = frame.copy()
    if not all(isinstance(name, str) for name in data.columns):
        raise ValueError("All input columns must have string names.")
    data.columns = data.columns.str.strip()
    data = data.rename(columns=ALIASES)
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names after applying aliases.")
    required = tuple(n for n in REQUIRED if not prediction_only or n not in ('Sw', 'R0'))
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}. A real core_id is required for grouped splitting.")
    if data.empty or data.core_id.isna().any():
        raise ValueError("Input must be nonempty and every row must have a core_id.")
    data['core_id'] = data.core_id.astype(str).str.strip()
    if data.core_id.str.strip().eq('').any():
        raise ValueError("Empty core identifiers are not allowed.")
    for name in REQUIRED[1:]:
        if name not in data.columns:
            continue
        data[name] = pd.to_numeric(data[name], errors='raise').astype(float)
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Nonfinite values in {name}.")
    if units not in ("fraction", "percent"):
        raise ValueError("units must be 'fraction' or 'percent'.")
    if units == 'percent':
        for name in ('phi', 'p_small', 'p_medium', 'Sw'):
            if name in data.columns:
                data[name] /= 100.0
    for name in ('phi', 'Sw'):
        if name not in data.columns:
            continue
        if not ((data[name] > 0) & (data[name] <= 1)).all():
            raise ValueError(f"{name} must lie in (0, 1] after unit conversion.")
    if (data.phi >= 1).any():
        raise ValueError("Porosity must be strictly less than one.")
    for name in ('p_small', 'p_medium'):
        if not data[name].between(0, 1).all():
            raise ValueError(f"{name} must lie in [0, 1].")
    if ((data.p_small + data.p_medium) > 1 + 1e-12).any():
        raise ValueError("p_small + p_medium must not exceed one.")
    for name in ('Rw', 'R0', 'Rt'):
        if name not in data.columns:
            continue
        if (data[name] <= 0).any():
            raise ValueError(f"{name} must be positive, in a common resistivity unit.")
    return data.reset_index(drop=True)


def load_data(path, units="fraction", prediction_only=False):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        frame = pd.read_csv(path, dtype={'core_id': str})
    elif path.suffix.lower() in ('.xlsx', '.xlsm'):
        frame = pd.read_excel(path, engine='openpyxl')
    else:
        raise ValueError("Use CSV or XLSX input.")
    return validate_data(frame, units, prediction_only)


def split_by_core(data, seed=42, train_fraction=0.6, validation_fraction=0.2):
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train, validation and test fractions must all be positive.")
    groups = np.array(sorted(data.core_id.unique()))
    if len(groups) < 10:
        raise ValueError("At least 10 distinct cores are required for this three-way split.")
    np.random.default_rng(seed).shuffle(groups)
    n_train = max(2, int(train_fraction * len(groups)))
    n_valid = max(2, int(validation_fraction * len(groups)))
    if n_train + n_valid > len(groups) - 2:
        raise ValueError("Split requires at least two cores per partition.")
    selected = {'train': groups[:n_train], 'validation': groups[n_train:n_train+n_valid],
                'test': groups[n_train+n_valid:]}
    partitions = {name: data[data.core_id.isin(ids)].copy().reset_index(drop=True)
                  for name, ids in selected.items()}
    manifest = {name: list(map(str, ids)) for name, ids in selected.items()}
    for name, subset in partitions.items():
        if (subset.Sw >= 0.999).sum() < 2 or ((subset.Sw < 0.999) & (subset.Sw > 0)).sum() < 2:
            raise ValueError(f"Partition {name} needs both fully and partially saturated observations.")
    return partitions, manifest
