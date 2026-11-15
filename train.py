import argparse
from pathlib import Path
import numpy as np
import torch
from hs_trans.data import DATASETS
from hs_trans.engine import environment, prepare_data, train_run, write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=202401, help='First seed; runs use consecutive seeds')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--shift-mode', choices=['cyclic', 'noncyclic'], default='cyclic')
    parser.add_argument('--components', choices=['igi', 'igi+sigi', 'cgi', 'igi+cgi', 'igi+sigi+cgi'], default='igi+sigi+cgi')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.runs < 1 or args.epochs < 1 or args.batch_size < 1 or (args.threads < 1) or (args.lr <= 0) or (args.weight_decay < 0):
        parser.error('runs, epochs, batch-size, threads, lr must be positive; weight-decay must be nonnegative')
    torch.set_num_threads(args.threads)
    for seed in range(args.seed, args.seed + args.runs):
        path = args.output / f'seed_{seed}'
        if path.exists() and any(path.iterdir()):
            parser.error(f'Run directory is not empty: {path}')
    prepared = prepare_data(args.dataset, args.data_root)
    config = {k: str(v) if isinstance(v, Path) else v for (k, v) in vars(args).items()}
    config['variant'] = 'spectral'
    results = [train_run(config, *prepared, args.output / f'seed_{seed}', seed) for seed in range(args.seed, args.seed + args.runs)]
    summary = dict(runs=results, environment=environment(), aggregate={key: dict(mean=float(np.mean([r[key] for r in results])), std_population=float(np.std([r[key] for r in results]))) for key in ('oa_percent', 'aa_percent', 'kappa')})
    write_json(args.output / 'summary.json', summary)
    print(f'Saved {args.runs} run(s) to {args.output}')
if __name__ == '__main__':
    main()
