"""Visualize training vs test distribution for benchmark datasets."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

missing = []
for _mod in ["numpy", "sklearn", "matplotlib"]:
    try:
        globals()[_mod] = __import__(_mod)
    except ImportError:
        missing.append(_mod)
if missing:
    raise SystemExit(
        "Missing required packages: "
        + ", ".join(missing)
        + ". Install them with 'pip install -r requirements-demo.txt'"
    )

import numpy as np

from data_factory import data_loader
from utils.analysis_tools import plot_projection_by_segment


LOADER_MAP = {
    "SMD": data_loader.SMDSegLoader,
    "SMAP": data_loader.SMAPSegLoader,
    "MSL": data_loader.MSLSegLoader,
    "PSM": data_loader.PSMSegLoader,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset distribution visualization")
    parser.add_argument("--dataset", type=str, default="SMD", help="dataset name (SMD, SMAP, MSL, PSM)")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="path to dataset directory (defaults to 'dataset/<dataset>')",
    )
    parser.add_argument("--method", type=str, choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--save_path", type=str, default=None, help="file to save the plot")
    args = parser.parse_args()

    if args.dataset not in LOADER_MAP:
        raise SystemExit("Unsupported dataset: " + args.dataset)

    if args.data_path is None:
        args.data_path = os.path.join("dataset", args.dataset)

    loader_cls = LOADER_MAP[args.dataset]
    loader = loader_cls(args.data_path, win_size=1, step=1, mode="train")

    train = loader.train
    test = loader.test
    data = np.concatenate([train, test], axis=0)
    segments = [(0, len(train)), (len(train), len(train) + len(test))]

    if args.save_path is None:
        args.save_path = f"{args.dataset.lower()}_{args.method}_segments.png"

    plot_projection_by_segment(data, segments, method=args.method, feature=None, save_path=args.save_path)


if __name__ == "__main__":
    main()
