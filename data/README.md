# Synthetic input data dictionary

`synthetic_cores.csv` is generated solely by `nmr_archie_sr.synthetic.generate`.
It has no source in real measurements or confidential data. Default seed:
20261102; 80 cores; 400 rows; CSV UTF-8; proportions as fractions.

| Column | Meaning | Domain / unit |
|---|---|---|
| core_id | Shared identifier across saturation states of a core | SYNTHETIC_0000 etc. |
| phi | Porosity | (0,1), fraction |
| p_small | Small-pore fraction feature | [0,1], fraction |
| p_medium | Medium-pore fraction feature | [0,1], fraction; p_small+p_medium <=1 |
| Rw | Pore-fluid resistivity | positive, ohm m in this example |
| R0 | Fully water-saturated rock resistivity | positive, same unit as Rw |
| Rt | Rock resistivity at Sw | positive, same unit as Rw |
| Sw | Water saturation target | (0,1], fraction |
| synthetic_true_m | Generating cementation exponent | diagnostic only; not a training feature |
| synthetic_true_n | Generating saturation exponent | diagnostic only; not a training feature |
| data_origin | Explicit provenance flag | SYNTHETIC |

Generator specification, with independent draws for each synthetic core:

```text
phi ~ Uniform(0.06, 0.18)
ps ~ Uniform(0.05, 0.40)
pm ~ Uniform(0.10, 0.45)
Rw ~ Uniform(0.08, 0.12)
m = 1.65 + 0.55*ps + 0.35*pm + 0.6*ps*pm
n = 1.65 + 0.6*(pm-ps)^2 + 0.25*pm
R0 = Rw * phi^(-m) * exp(Normal(0,0.01))
Sw = 1.0, 0.85, 0.65, 0.45, 0.25
Rt = R0 at Sw=1; otherwise R0 * Sw^(-n) * exp(Normal(0,0.02))
```

The added noise is artificial log-resistivity noise, not a calibrated instrument
uncertainty. Each core's R0 and pore features repeat across saturation states.
The ground-truth columns are ignored by all fitting and prediction functions.
The generator's formula is not inserted into the search seed templates.

The aliases `孔隙度`, `小孔%`, `中孔%` are recognized for compatibility, but a
percent sign in a column name does not trigger automatic unit conversion. Use
the explicit `--units` setting, with all proportion columns in the same units.
Rows with Sw>=0.999 are treated as fully saturated for m calibration; Sw<0.999
rows are used for n calibration. Every partition must contain both stages.
