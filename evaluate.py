import argparse
from pathlib import Path
import torch
from hs_trans.engine import evaluate_checkpoint

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--full-map', action='store_true', help='Also save 1-based predictions for every scene pixel')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive')
    torch.set_num_threads(args.threads)
    evaluate_checkpoint(args.checkpoint, args.data_root, args.device, args.batch_size, args.output_dir or args.checkpoint.parent / 'evaluation', args.full_map)
if __name__ == '__main__':
    main()
