# NMR-constrained Archie-type symbolic regression

A Python implementation of expression-tree search for pore-structure-dependent
cementation and saturation exponents. The search combines Monte Carlo tree search
(MCTS) with optional tabular Double Q action priors. An independent MCTS-only
mode is included.

**The experimental core and well-log data are subject to confidentiality
restrictions. Synthetic example data are provided to support code execution,
testing and reproduction of the computational workflow.**

## Install and run

Python 3.10 or newer is required. The bundled results were produced with Python
3.12.14, NumPy 2.3.5 and pandas 3.0.1. The exact requirements below target that
tested Python environment; `pip install .` also allows compatible older libraries.

From the extracted project folder, on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m nmr_archie_sr train --data data/synthetic_cores.csv --output outputs/demo --iterations 80 --simulations 30 --seed 42 --split-seed 42
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux/macOS replace the first two environment-specific steps with
`python3 -m venv .venv` and use `.venv/bin/python` for the commands.
For Excel input, install the optional dependency with `python -m pip install ".[excel]"`.
CSV input requires no spreadsheet software. There is no Julia or plotting dependency.

To regenerate exactly the example input:

```text
python -m nmr_archie_sr generate --cores 80 --seed 20261102 --output outputs/regenerated.csv
```

To disable all Double Q choices, updates and UCT priors:

```text
python -m nmr_archie_sr train --data data/synthetic_cores.csv --output outputs/mcts_only --iterations 80 --simulations 30 --seed 42 --split-seed 42 --disable-double-q
```

To reload a saved model and predict with no observed saturation or measured R0:

```text
python -m nmr_archie_sr predict --model examples/results/double_q/model.json --data examples/prediction_inputs.csv --output outputs/predictions.csv
```

After environment setup, `python` in these generic commands means the Python
executable inside that environment. The installed command `nmr-archie-sr` is equivalent.

## Inputs and outputs

Training columns are `core_id, phi, p_small, p_medium, Rw, R0, Rt, Sw`.
See [the data dictionary](data/README.md). All proportions must be fractions
unless `--units percent` explicitly selects percent for all four proportion
columns. Resistivities must share a unit. Real core identifiers are mandatory:
row numbers cannot replace core grouping.

Prediction requires only `core_id, phi, p_small, p_medium, Rw, Rt`.
If `R0` is supplied, a separate measured-R0 diagnostic is produced; it is not the
deployment prediction. If `Sw` is supplied, it is copied for comparison and is
never used to compute predictions.

Each training run saves `model.json`, `split_manifest.json`, `search_history.json`,
`metrics.json`, `predictions.csv` and `run_info.json`. The last file records input
hashes, software versions, configuration and random seeds. Models use a validated
expression-tree JSON format without `eval`. See [the method guide](docs/METHOD.md).

## Synthetic demonstration and limits

The data contain 80 artificial cores with five saturation states each. One common
60/20/20 core split gives 48 training, 16 validation and 16 test cores. The 64
partially saturated test rows are evaluated separately from training/validation.

The seed-42 demonstration in `examples/results/double_q` gives a deployment water
saturation MAE of **1.2319 percentage points**. This is an execution example under
the stated generator and is not an accuracy claim about real reservoirs.
The arbitrary fixed `m=n=2.05` comparator is not calibrated to the synthetic data,
so its error must not be interpreted as evidence of superiority over a fitted
Archie baseline.

The paired three-seed check in `examples/comparison` compares the same data,
split, templates and search budget with and without Double Q. Re-run it with:

```text
python scripts/compare_search.py --output outputs/comparison
```

The initial templates can themselves be selected as the final model. See
[the verification report](docs/VERIFICATION.md) for actual outcomes. This small
check is not a statistical validation or a substitute for comparison with
independent symbolic-regression methods.

The experimental data remain confidential. The supplied synthetic examples use
the documented input format so that the computational workflow can be executed
and tested. This repository focuses on symbolic regression of Archie exponents;
NMR-MICP calibration, raw T2 processing, well-log validation and external PySR or
gplearn comparisons are outside its implementation scope. The n exponent is
calibrated independently with measured R0; deployment uses R0 predicted from m.

## Project contents

- `src/nmr_archie_sr/`: implementation, input validation and CLI.
- `data/`: synthetic CSV and generation specification.
- `examples/`: synthetic model exports, predictions and paired checks.
- `tests/`: regression tests for search semantics, group separation and inference.
- `docs/`: algorithm details, changes and verification.
- [GitHub upload instructions in Chinese](GITHUB_UPLOAD_zh.md).

## License and provenance

The code and bundled synthetic example materials are distributed under the
[MIT license](LICENSE). No empirical dataset is licensed or redistributed by this
release. The original notebook-style research script remains outside the public
package. This release has not been assigned a DOI and does not imply journal
acceptance. A verified article citation can be added after the manuscript's title,
author list and publication details are finalized.
