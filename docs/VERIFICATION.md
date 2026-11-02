# Verification record for v0.1.0

## Environment

Windows, Python 3.12.14, NumPy 2.3.5, pandas 3.0.1. All execution used the public
synthetic CSV. No confidential input was accessed. Timestamps and run duration
are not intended to be deterministic. Small floating-point differences across
platforms/library versions are possible; serialized model coefficients retain
full precision.

## Automated checks

`python -m unittest discover -s tests -v`: **21 tests passed**.

The checks cover immutable parent expansion, protected division sign and zero
handling, tree export and invalid tokens, retaining a completed rollout winner,
search size/depth constraints, terminal Double Q updates, legal-action bootstrap masking, complete disabling of
Double Q, disjoint exhaustive core splits, deterministic synthetic generation,
explicit units, invalid inputs, samplewise coefficient constraints, fixed-fit
validation, explicit failure instead of fallback, deployment without R0 or Sw,
inference clipping counts, metric units, and JSON model loading.

A target-isolation regression test changes held-out test Rt and Sw and checks
that the serialized fitted models are unchanged at the same search seed. This
checks the implemented training boundary; it does not establish external validity.

The project also built and installed successfully as a wheel using the local
Python packaging toolchain (no dependency downloads for this packaging check).
The installed module executed CLI training with Double Q disabled and CLI
prediction from the saved model. The prediction input omits both R0 and Sw;
its ten outputs are in `examples/prediction_output.csv`.

## Recorded synthetic run

Data seed 20261102, 80 cores, 400 rows. Split seed 42: 48/16/16 cores.
Search seed 42, 80 outer iterations and 30 simulations per search for each stage.
The final formula for both stages is a generic initial template:

```text
m = clip(1.7893423991268858 * pexp(ps*pm), 0.1, 3.0)
n = clip(1.6270399704004781 * pexp(ps*pm), 1.0, 5.0)
```

The inference clipping counts were zero in all three partitions. Training and
validation mean squared errors refer to log-resistivity relations, while the
table below refers to predicted water saturation on 64 partially saturated test
rows from 16 unseen cores.

| Prediction | Test MAE (percentage points) | Test RMSE (fraction) |
|---|---:|---:|
| Learned deployment, predicted R0 | 1.231939 | 0.017824 |
| Fixed m=n=2.05 deployment | 14.809460 | 0.164494 |
| Learned n, measured R0 diagnostic | 1.027361 | 0.012883 |
| Fixed n=2.05, measured R0 diagnostic | 4.589648 | 0.048921 |

These are generator-specific outputs, not empirical performance estimates.
The fixed reference exponents were not fitted to these data, so the table does
not establish superiority over a calibrated conventional model.

## Paired search check

`python scripts/compare_search.py --output outputs/comparison` runs search seeds
42,43,44 at the same split and nominal search budget with Double Q on and off.
All six runs selected the same two templates above, giving the same test MAE
1.231939 percentage points. Exact intermediate search histories and expression
counts differ, but the final prediction is unchanged. The corresponding complete
outputs are in `examples/comparison/`, alongside `comparison.csv`.

**No Double Q advantage is demonstrated by this check.** Because seed templates
already contain the selected formulas, the task is not a discriminating test of
search improvement. Additional justified tasks, independent baselines and a
broader prespecified evaluation would be needed for a methodological claim.

## Reproduction boundary

The package reproduces its synthetic workflow only. It does not reproduce the
earlier manuscript's empirical tables, measurements, figures, well-log results
or external symbolic-regression comparisons. Confidential data restrictions and
later code changes must be disclosed when describing this repository.
