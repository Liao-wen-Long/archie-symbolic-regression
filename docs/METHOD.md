# Implementation and evaluation

## Forward model

The model estimates `m=a*g_m(ps,pm)` and `n=b*g_n(ps,pm)` with positive scalar
coefficients. For m, training minimizes squared residuals of
`log(R0/Rw) = -m*log(phi)` on fully saturated rows. For n, it minimizes residuals
of `log(R0/Rt) = n*log(Sw)` on partially saturated rows. The stages are independently
calibrated; n uses measured R0. It does not use predicted R0 while fitting.

Each scalar is fitted once on training rows by constrained no-intercept least
squares. The feasible interval is the intersection of every training sample's
exponent bounds: m in [0.1,3] and n in [1,5]. Nonpositive g values or incompatible
intervals are rejected. Validation uses the same coefficient, and expressions
violating the bounds on validation samples are rejected. Both inputs must have
a nonzero effect when replacing that input with its training mean. These checks
are heuristic admissibility rules, not proof of a physical mechanism.

## Search

Expressions are immutable trees. An action fills one hole with an operator or
terminal. Legal actions preserve size<=25 and depth<=6 by default. Rollouts
complete all holes. Fully specified root seeds can mutate a terminal. MCTS uses
UCT (exploration 1.41); complete evaluated candidates are retained globally.
No parent tree changes during expansion. Twelve generic templates are evaluated
first: for each of `ps+pm`, `ps*pm`, `ps-pm`, the plain expression, one plus the
expression, two plus the expression and protected exp of the expression.

The optional Double Q component uses two tabular estimates, alpha=0.15 and
gamma=0.92. One table selects the next action and the other evaluates it.
Bootstrap action selection is masked to the successor tree's legal tokens.
Nonterminal transitions have zero reward; the final transition has the terminal
expression score and no bootstrap. Action choice combines uniform exploration
and a softmax of summed Q values; UCT also adds a Q prior weighted by 0.30.
Epsilon starts at 0.30 and decays by 0.995 per outer iteration to a minimum 0.05.
The aggregate state uses size, depth, number of holes, variable-presence flags,
operator counts and root arity. It is a coarse heuristic state representation,
not a unique tree identifier or an exact Markov state. Q entries are keyed by
operator/terminal, sharing action estimates across hole positions. Thus standard
tabular convergence guarantees cannot simply be assumed for this search.

Let E=0.7*train_MSE+0.3*validation_MSE, r=validation_MSE/(train_MSE+1e-12).
The selection score is:

```text
simplicity = max(0.7, 1 - 0.01*max(0, tree_size-3))
overfit = max(0.5, 1 - 0.15*max(0, r-1.5))
reward = simplicity * overfit / (1 + log(1+E))
```

Invalid expressions score -1. Failure to find a valid expression raises an error.
There is no hidden constant-model fallback. The validation set is used for
expression selection, so its scores are not independent generalization estimates.
The final training coefficient is not refitted on combined training+validation.

## Operator definitions

`ps` and `pm` are the two fractions; constants are 1 and 2. Ordinary operators
are addition, subtraction, multiplication and square. Protected operators are:

- `pdiv(a,b)`: divide by b if abs(b)>=1e-12, otherwise divide by signed 1e-12
  (zero uses positive 1e-12). The sign of a negative denominator is preserved.
- `pexp(a)`: exp(clip(a,-20,20)).
- `plog(a)`: log(max(abs(a),1e-12)).
- `psqrt(a)`: sqrt(max(a,0)).

Protected operators keep explicit names in exported expressions. They must not
be simplified as ordinary division, exponential, logarithm or square root.
Expressions producing nonfinite values are rejected during fitting. Nonfinite
inference values raise errors. Inference clips m and n to their declared bounds
and reports the count of affected rows, since unseen inputs can exceed bounds.

## Evaluation and prediction

One seeded core-level split is shared by both stages: 60% training, 20%
validation, remaining cores test. No core appears in two partitions. Test values
are not passed to either trainer. Grouping by core prevents saturation states
of the same core from crossing partitions; additional grouping is required if
independent experimental units are wells, batches or sites rather than cores.

Deployment estimates `log(R0_hat)=log(Rw)-m_hat*log(phi)` and
`Sw_hat=exp(min(log(R0_hat)-log(Rt),0)/n_hat)`, capping saturation at one.
The separately labeled n-only diagnostic substitutes measured R0 and therefore
has extra information. Each baseline uses the same inputs as its learned
counterpart, with fixed m=n=2.05. This fixed arbitrary comparator is a smoke-test
reference and is not an optimized conventional Archie baseline.

Metrics use partially saturated rows only and report fractions, percentage points
and relative percent distinctly. R2 is null when observed targets are constant.
Resistivity noise and R0 normalization can induce within-core dependence, so row
counts are not independent core sample counts. No confidence intervals or real
data performance conclusions are claimed here.
