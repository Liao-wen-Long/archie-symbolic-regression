"""Command-line entry points. No training is performed on import."""
import argparse
import hashlib
from pathlib import Path
from .data import load_data
from .model import SearchConfig
from .pipeline import load_models, predict, train
from .synthetic import generate


def main(argv=None):
    parser = argparse.ArgumentParser(description='NMR-constrained Archie-type symbolic regression')
    commands = parser.add_subparsers(dest='command', required=True)
    synthetic = commands.add_parser('generate', help='Generate clearly labeled synthetic data')
    synthetic.add_argument('--output', default='data/synthetic_cores.csv')
    synthetic.add_argument('--cores', type=int, default=80)
    synthetic.add_argument('--seed', type=int, default=20261102)
    fit = commands.add_parser('train', help='Train and evaluate using core-grouped partitions')
    fit.add_argument('--data', required=True)
    fit.add_argument('--output', default='outputs/run')
    fit.add_argument('--units', choices=['fraction', 'percent'], default='fraction')
    fit.add_argument('--iterations', type=int, default=100)
    fit.add_argument('--simulations', type=int, default=40)
    fit.add_argument('--seed', type=int, default=42)
    fit.add_argument('--split-seed', type=int, default=42)
    fit.add_argument('--disable-double-q', action='store_true')
    inference = commands.add_parser('predict', help='Predict without observed Sw or measured R0')
    inference.add_argument('--model', required=True)
    inference.add_argument('--data', required=True)
    inference.add_argument('--output', default='outputs/predictions.csv')
    inference.add_argument('--units', choices=['fraction', 'percent'], default='fraction')
    args = parser.parse_args(argv)
    if args.command == 'generate':
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        generate(args.cores, args.seed).to_csv(output, index=False)
        print(f'Generated SYNTHETIC data: {output}')
    elif args.command == 'train':
        config = SearchConfig(iterations=args.iterations, simulations=args.simulations,
                              use_double_q=not args.disable_double_q)
        report = train(load_data(args.data, args.units), args.output, config,
                       args.seed, args.split_seed,
                       hashlib.sha256(Path(args.data).read_bytes()).hexdigest())
        test = report['test']['deployment_learned']
        print(f"Held-out test: {test['rows']} partial-saturation rows; MAE={test['mae_percentage_points']:.4f} percentage points")
        print(f'Files saved in {args.output}')
    else:
        m, n = load_models(args.model)
        result, clipped = predict(load_data(args.data, args.units, prediction_only=True), m, n)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f'Saved {len(result)} predictions to {output}; clipping: {clipped}')


if __name__ == '__main__':
    main()
