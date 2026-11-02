# Version 0.1.0

This refactor changes behavior relative to the supplied later notebook version.
It cannot be described as a byte-for-byte implementation of the submitted paper.

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

Any use for manuscript results requires a new private-data run and corresponding
updates to methods, figures and numbers. The prior manuscript and earlier code
version were not audited as equivalent to this release.
