import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--train_start', type=float, default=0.0)
    parser.add_argument('--train_end', type=float, default=1.0)
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'transformer_ae'])
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--replay_horizon', type=int, default=None)
    parser.add_argument('--model_tag', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anomaly_ratio', type=float, default=4.00)
    parser.add_argument(
        '--cpd_penalty',
        type=int,
        default=20,
        help='Penalty value for ruptures change point detection',
    )
    parser.add_argument('--cpd_top_k', type=int, default=3,
                        help='number of zoomed views for CPD visualization')
    parser.add_argument(
        '--cpd_extra_ranges',
        type=str,
        default='0:4000',
        help='comma-separated start:end pairs for additional CPD zoom views',
    )
    parser.add_argument('--min_cpd_gap', type=int, default=30,
                        help='minimum gap between CPD change points')
    parser.add_argument('--cpd_log_interval', type=int, default=20,
                        help='log metrics every N CPD updates')

    def _parse_ranges(arg):
        if not arg:
            return None
        pairs = []
        for part in arg.split(','):
            if ':' not in part:
                continue
            start, end = part.split(':', 1)
            pairs.append((int(start), int(end)))
        return pairs or None

    config = parser.parse_args()
    config.cpd_extra_ranges = _parse_ranges(config.cpd_extra_ranges)
    if config.model_tag is None:
        config.model_tag = config.dataset
    # create timestamped directory under outputs/<name>/ for results
    config.model_save_path = prepare_experiment_dir(config.dataset)
    setup_logging(os.path.join(config.model_save_path, "log.txt"))

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
