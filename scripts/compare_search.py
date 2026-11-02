"""Small paired computational check; not a scientific benchmark or paper reproduction."""
import argparse
from pathlib import Path
import pandas as pd
from nmr_archie_sr.data import load_data
from nmr_archie_sr.model import SearchConfig
from nmr_archie_sr.pipeline import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/synthetic_cores.csv')
    parser.add_argument('--output', default='outputs/comparison')
    args = parser.parse_args()
    data = load_data(args.data)
    rows = []
    for seed in (42, 43, 44):
        for enabled in (True, False):
            variant = 'double_q' if enabled else 'mcts_only'
            path = Path(args.output) / f'{variant}_seed_{seed}'
            report = train(data, path, SearchConfig(iterations=80, simulations=30,
                                                    use_double_q=enabled), seed=seed, split_seed=42)
            rows.append({'variant': variant, 'search_seed': seed, 'split_seed': 42,
                         'test_mae_percentage_points': report['test']['deployment_learned']['mae_percentage_points'],
                         'test_rmse_fraction': report['test']['deployment_learned']['rmse_fraction']})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(Path(args.output) / 'comparison.csv', index=False)


if __name__ == '__main__':
    main()
