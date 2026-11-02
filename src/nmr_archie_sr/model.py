"""Two independently calibrated exponents with fixed-coefficient validation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .expressions import Expression, OPERATORS, initial_candidates, partial
from .search import MCTS


@dataclass
class SearchConfig:
    iterations: int = 100
    simulations: int = 40
    max_size: int = 25
    max_depth: int = 6
    validation_weight: float = 0.3
    use_double_q: bool = True
    q_weight: float = 0.3


def fit_coefficient(g, predictor, target, bounds):
    """Fit a positive scalar by no-intercept LS, with bounds at every training point."""
    g, predictor, target = map(lambda x: np.asarray(x, dtype=float), (g, predictor, target))
    if (g.shape != predictor.shape or g.shape != target.shape or not len(g)
            or not all(np.isfinite(x).all() for x in (g, predictor, target))
            or np.any(g <= 1e-8) or np.any(g > 100)):
        return None
    lower, upper = bounds
    if lower <= 0 or upper <= lower:
        raise ValueError("Exponent bounds must be positive and increasing.")
    coefficient_min = float(np.max(lower / g))
    coefficient_max = float(np.min(upper / g))
    if coefficient_min > coefficient_max:
        return None
    x = g * predictor
    denominator = float(x @ x)
    if denominator <= 1e-16:
        return None
    coefficient = float(np.clip(float(x @ target) / denominator, coefficient_min, coefficient_max))
    prediction = coefficient * x
    return coefficient, float(np.mean((target - prediction) ** 2))


@dataclass
class ExponentModel:
    stage: str
    expression: Expression
    coefficient: float
    bounds: tuple[float, float]
    train_mse: float
    validation_mse: float
    selection_reward: float

    def predict(self, frame):
        g = self.expression.evaluate(frame.p_small.to_numpy(), frame.p_medium.to_numpy())
        raw = self.coefficient * g
        if not np.isfinite(raw).all():
            raise ValueError(f"Nonfinite {self.stage} prediction: input is outside the valid model domain.")
        clipped = np.clip(raw, *self.bounds)
        count = int(np.count_nonzero((raw < self.bounds[0]) | (raw > self.bounds[1])))
        return clipped, count

    def to_dict(self):
        return {'stage': self.stage, 'expression': self.expression.to_dict(),
                'expression_text': str(self.expression), 'coefficient': self.coefficient,
                'bounds': list(self.bounds), 'train_mse': self.train_mse,
                'validation_mse': self.validation_mse, 'selection_reward': self.selection_reward,
                'effective_exponent': f"clip({self.coefficient:.17g} * {self.expression}, {self.bounds[0]}, {self.bounds[1]})"}

    @classmethod
    def from_dict(cls, data):
        return cls(data['stage'], Expression.from_dict(data['expression']),
                   float(data['coefficient']), tuple(data['bounds']),
                   float(data['train_mse']), float(data['validation_mse']),
                   float(data['selection_reward']))


class StageTrainer:
    def __init__(self, train, validation, stage, config, seed=42, bounds=None):
        if stage not in ('m', 'n'):
            raise ValueError("stage must be m or n.")
        if config.iterations < 1 or not 0 <= config.validation_weight <= 1:
            raise ValueError("Invalid search configuration.")
        self.stage, self.config = stage, config
        self.bounds = bounds or ((0.1, 3.0) if stage == 'm' else (1.0, 5.0))
        self.train = train[train.Sw >= 0.999].copy() if stage == 'm' else train[train.Sw < 0.999].copy()
        self.validation = (validation[validation.Sw >= 0.999].copy() if stage == 'm'
                           else validation[validation.Sw < 0.999].copy())
        if min(len(self.train), len(self.validation)) < 2:
            raise ValueError("Each stage needs at least two train and validation rows.")
        self.cache, self.history = {}, []
        self.rng = np.random.default_rng(seed)
        self.engine = MCTS(self.reward, self.rng, max_size=config.max_size,
                           max_depth=config.max_depth, simulations=config.simulations,
                           use_double_q=config.use_double_q, q_weight=config.q_weight)

    def _arrays(self, frame):
        if self.stage == 'm':
            return -np.log(frame.phi.to_numpy()), np.log(frame.R0.to_numpy() / frame.Rw.to_numpy())
        return np.log(frame.Sw.to_numpy()), np.log(frame.R0.to_numpy() / frame.Rt.to_numpy())

    def reward(self, tree):
        if tree in self.cache:
            return self.cache[tree].selection_reward if self.cache[tree] else -1.0
        self.cache[tree] = None
        if not tree.complete or not tree.has_variables():
            return -1.0
        ps, pm = self.train.p_small.to_numpy(), self.train.p_medium.to_numpy()
        g = tree.evaluate(ps, pm)
        if not np.isfinite(g).all() or np.std(g) < 1e-8:
            return -1.0
        for replaced in (tree.evaluate(np.full_like(ps, ps.mean()), pm),
                         tree.evaluate(ps, np.full_like(pm, pm.mean()))):
            if not np.isfinite(replaced).all() or np.max(np.abs(g - replaced)) < 1e-8:
                return -1.0
        predictor, target = self._arrays(self.train)
        fitted = fit_coefficient(g, predictor, target, self.bounds)
        if fitted is None:
            return -1.0
        coefficient, train_mse = fitted
        g_valid = tree.evaluate(self.validation.p_small.to_numpy(), self.validation.p_medium.to_numpy())
        effective = coefficient * g_valid
        if (not np.isfinite(effective).all() or np.any(effective < self.bounds[0] - 1e-10)
                or np.any(effective > self.bounds[1] + 1e-10)):
            return -1.0
        valid_predictor, valid_target = self._arrays(self.validation)
        validation_mse = float(np.mean((valid_target - effective * valid_predictor) ** 2))
        if not math.isfinite(validation_mse):
            return -1.0
        w = self.config.validation_weight
        joint = (1 - w) * train_mse + w * validation_mse
        ratio = validation_mse / (train_mse + 1e-12)
        overfit = max(0.5, 1 - 0.15 * max(0.0, ratio - 1.5))
        simplicity = max(0.7, 1 - 0.01 * max(0, tree.size - 3))
        score = float(simplicity * overfit / (1 + math.log1p(joint)))
        self.cache[tree] = ExponentModel(self.stage, tree, coefficient, self.bounds,
                                         train_mse, validation_mse, score)
        return score

    def search(self):
        for candidate in initial_candidates():
            if candidate.size <= self.config.max_size and candidate.depth <= self.config.max_depth:
                self.engine.consider(candidate)
        last = -math.inf
        for iteration in range(self.config.iterations):
            if iteration % 4 == 0 and self.engine.best_tree is not None:
                initial = self.engine.best_tree
            else:
                initial = partial(OPERATORS[int(self.rng.integers(len(OPERATORS)))])
            result = self.engine.search(initial)
            if result.reward > last and result.tree in self.cache and self.cache[result.tree] is not None:
                model = self.cache[result.tree]
                self.history.append({'iteration': iteration, 'expression': str(result.tree),
                                     'reward': result.reward, 'train_mse': model.train_mse,
                                     'validation_mse': model.validation_mse})
                last = result.reward
        tree = self.engine.best_tree
        if tree is None or self.cache.get(tree) is None:
            raise RuntimeError(f"No valid {self.stage} expression was found; no silent baseline fallback was used.")
        return self.cache[tree]
