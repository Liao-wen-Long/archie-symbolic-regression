# Version 0.1.0

This version includes the following implementation and verification improvements.

- Immutable expression trees prevent parent mutation and stale cached states.
- The best completed rollout is returned, including improvements from complete seeds.
- Double Q transitions use explicit terminal handling instead of artificial self-loops,
  and bootstrap action selection is masked to legal successor tokens.
- Validation logging and model selection use the training-fitted coefficient.
- A common core-level three-way split replaces separate row-level two-way splits.
- Metrics separate train/validation/test and separate deployment from measured-R0 diagnostics.
- Physical exponent bounds are enforced pointwise during training/validation.
- Inference clipping is explicit and counted. Signed protected division replaces
  the old absolute-denominator behavior; exports name all protected operators.
- No silent constant fallback, global warning suppression, hard-coded private
  workbook, global random seed mutation or import-time training.
- A command-line interface, safe tree JSON, deterministic synthetic generator,
  public examples, tests and English documentation are included.

For reproducible analyses, retain the configuration, units, core-group split and
software versions recorded with each run. Experimental measurements remain
confidential; the synthetic example specification is documented in `data/README.md`.
