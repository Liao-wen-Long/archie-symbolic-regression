import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from nmr_archie_sr.data import split_by_core, validate_data
from nmr_archie_sr.expressions import Expression as E, TOKENS, partial, state_key
from nmr_archie_sr.model import ExponentModel, SearchConfig, StageTrainer, fit_coefficient
from nmr_archie_sr.pipeline import load_models, metrics, predict, train
from nmr_archie_sr.search import DoubleQ, MCTS, Node
from nmr_archie_sr.synthetic import generate


class ExpressionTests(unittest.TestCase):
    def test_expansion_preserves_parent_and_cached_state(self):
        parent = Node(partial('+'))
        before, key = str(parent.tree), state_key(parent.tree)
        child = MCTS(lambda t: 1, np.random.default_rng(2))._expand(parent)
        self.assertEqual(str(parent.tree), before)
        self.assertEqual(state_key(parent.tree), key)
        self.assertNotEqual(str(child.tree), before)

    def test_signed_protected_division_and_roundtrip(self):
        expression = E('pdiv', (E('ps'), E('-', (E('pm'), E('2')))))
        np.testing.assert_allclose(expression.evaluate([.2], [.3]), [.2 / -1.7])
        restored = E.from_dict(json.loads(json.dumps(expression.to_dict())))
        self.assertEqual(expression, restored)
        self.assertIn('pdiv', str(expression))

    def test_zero_division_is_finite(self):
        expression = E('pdiv', (E('ps'), E('-', (E('pm'), E('pm')))))
        self.assertTrue(np.isfinite(expression.evaluate([.2], [.3])).all())

    def test_incomplete_export_and_unknown_token_rejected(self):
        with self.assertRaises(ValueError):
            partial('+').to_dict()
        with self.assertRaises(ValueError):
            E.from_dict({'token': '__import__'})


class SearchTests(unittest.TestCase):
    def test_complete_rollout_winner_is_returned(self):
        engine = MCTS(lambda t: float(t.evaluate([.2], [.3])[0]),
                      np.random.default_rng(3), simulations=60)
        initial = E('+', (E('ps'), E('pm')))
        result = engine.search(initial)
        self.assertGreater(result.reward, .5)
        self.assertEqual(result.reward, engine.reward(result.tree))

    def test_partial_search_obeys_limits(self):
        engine = MCTS(lambda t: float(t.size), np.random.default_rng(7),
                      max_size=9, max_depth=3, simulations=50)
        result = engine.search(partial('+'))
        self.assertTrue(result.tree.complete)
        self.assertLessEqual(result.tree.size, 9)
        self.assertLessEqual(result.tree.depth, 3)

    def test_terminal_double_q_does_not_bootstrap(self):
        q = DoubleQ(np.random.default_rng(4), alpha=1)
        q.values(('next',))
        q.q1[('next',)][:] = 100
        q.q2[('next',)][:] = 100
        q.update(('start',), 'ps', 2, ('next',), terminal=True)
        self.assertEqual(q.values(('start',))[TOKENS.index('ps')], 2)

    def test_disabled_double_q_has_no_tables(self):
        engine = MCTS(lambda t: .5, np.random.default_rng(3), use_double_q=False)
        engine.search(partial('+'))
        self.assertFalse(engine.q.q1)
        self.assertFalse(engine.q.q2)

    def test_bootstrap_masks_illegal_actions(self):
        q = DoubleQ(np.random.default_rng(4), alpha=1, gamma=1)
        q.values(('next',))
        q.q1[('next',)][:] = 100
        q.q2[('next',)][:] = 100
        q.q1[('next',)][TOKENS.index('ps')] = 2
        q.q2[('next',)][TOKENS.index('ps')] = 2
        q.update(('start',), 'pm', 0, ('next',), next_tokens=['ps'])
        self.assertEqual(q.values(('start',))[TOKENS.index('pm')], 2)


class DataAndFittingTests(unittest.TestCase):
    def setUp(self):
        self.data = generate(20, 5)

    def test_core_groups_are_disjoint_and_exhaustive(self):
        partitions, ids = split_by_core(self.data)
        a, b, c = map(set, ids.values())
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(a | b | c, set(self.data.core_id))
        self.assertEqual(sum(map(len, partitions.values())), len(self.data))

    def test_seeded_generation_and_split_repeat(self):
        self.assertTrue(self.data.equals(generate(20, 5)))
        self.assertEqual(split_by_core(self.data)[1], split_by_core(self.data)[1])

    def test_units_are_explicit(self):
        percent = self.data.copy()
        for name in ('phi', 'p_small', 'p_medium', 'Sw'):
            percent[name] *= 100
        with self.assertRaises(ValueError):
            validate_data(percent)
        converted = validate_data(percent, 'percent')
        np.testing.assert_allclose(converted.phi, self.data.phi)

    def test_missing_id_nonfinite_and_invalid_fractions_rejected(self):
        with self.assertRaises(ValueError):
            validate_data(self.data.drop(columns='core_id'))
        for column, value in [('Rw', 0), ('phi', np.nan), ('p_small', 1.1), ('Sw', -1)]:
            broken = self.data.copy()
            broken.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_data(broken)

    def test_pointwise_negative_exponent_rejected(self):
        self.assertIsNone(fit_coefficient([-1, 1, 2], [1, 1, 1], [2, 2, 2], (.1, 3)))

    def test_scalar_is_constrained_at_every_sample(self):
        coefficient, _ = fit_coefficient([1, 2], [1, 1], [10, 20], (.1, 3))
        self.assertEqual(coefficient, 1.5)

    def test_validation_uses_training_coefficient(self):
        partitions, _ = split_by_core(self.data)
        trainer = StageTrainer(partitions['train'], partitions['validation'], 'm', SearchConfig())
        expression = E('pexp', (E('*', (E('ps'), E('pm'))),))
        self.assertGreater(trainer.reward(expression), 0)
        model = trainer.cache[expression]
        x, y = trainer._arrays(trainer.validation)
        g = expression.evaluate(trainer.validation.p_small, trainer.validation.p_medium)
        expected = np.mean((y - model.coefficient * g * x) ** 2)
        self.assertAlmostEqual(model.validation_mse, expected)

    def test_no_valid_model_raises_instead_of_baseline_fallback(self):
        partitions, _ = split_by_core(self.data)
        trainer = StageTrainer(partitions['train'], partitions['validation'], 'm', SearchConfig(iterations=1, simulations=1))
        trainer.reward = lambda tree: -1.0
        trainer.engine.reward = trainer.reward
        with self.assertRaises(RuntimeError):
            trainer.search()


class PipelineTests(unittest.TestCase):
    def test_deployment_needs_neither_observed_sw_nor_measured_r0(self):
        data = generate(10).drop(columns=['Sw', 'R0'])
        model = ExponentModel('m', E('1'), 2.05, (.1, 3), 0, 0, 1)
        result, _ = predict(data, model, model)
        self.assertEqual(len(result), len(data))
        self.assertNotIn('Sw_n_only', result)
        self.assertTrue(result.Sw_predicted.between(0, 1).all())

    def test_clipping_is_counted(self):
        data = generate(10)
        model = ExponentModel('m', E('2'), 3, (.1, 3), 0, 0, 1)
        values, count = model.predict(data)
        self.assertTrue((values == 3).all())
        self.assertEqual(count, len(data))

    def test_metrics_report_units_and_constant_r2(self):
        values = metrics([.5, .5], [.6, .4])
        self.assertAlmostEqual(values['mae_percentage_points'], 10)
        self.assertIsNone(values['r2'])

    def test_test_targets_cannot_change_model_and_json_roundtrip(self):
        data = generate(20, 9)
        _, manifest = split_by_core(data)
        modified = data.copy()
        is_test = modified.core_id.isin(manifest['test'])
        modified.loc[is_test, 'Rt'] *= 1.2
        modified.loc[is_test & (modified.Sw < .999), 'Sw'] *= .9
        config = SearchConfig(iterations=3, simulations=5)
        temp_root = Path(__file__).resolve().parents[1] / 'outputs' / 'test_tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(temp_root.resolve()))
            first, second = Path(tmp) / 'first', Path(tmp) / 'second'
            report = train(data, first, config)
            train(modified, second, config)
            self.assertEqual((first / 'model.json').read_text(), (second / 'model.json').read_text())
            m, n = load_models(first / 'model.json')
            result, _ = predict(data, m, n)
            self.assertTrue(np.isfinite(result.Sw_predicted).all())
            self.assertEqual(report['test']['deployment_learned']['rows'], 16)
            self.assertEqual(report['test']['cores'], 4)


if __name__ == '__main__':
    unittest.main()
