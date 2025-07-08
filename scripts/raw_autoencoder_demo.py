"""Train a simple autoencoder directly on raw time-series windows."""

import argparse
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

missing = []
for _mod in ["numpy", "torch", "sklearn", "matplotlib"]:
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
import torch

from data_factory.data_loader import get_loader_segment
from utils.window_autoencoder import (
    WindowDataset,
    BasicWindowAutoencoder,
    train_window_autoencoder,

    collect_autoencoder_details,

)
from utils.analysis_tools import (
    plot_reconstruction_tsne,
    plot_reconstruction_pca,
    plot_autoencoder_vs_series,
    plot_vector_projection,

)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AE directly on windows")
    parser.add_argument("--dataset", type=str, default="SMD", help="dataset name")
    parser.add_argument("--data_path", type=str, default="dataset/SMD")
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("outputs", args.dataset.lower(), f"ws{args.win_size}", timestamp)
    os.makedirs(out_dir, exist_ok=True)

    loader = get_loader_segment(
        args.data_path,
        batch_size=1,
        win_size=args.win_size,
        step=1,
        mode="train",
        dataset=args.dataset,
    )
    ds = loader.dataset
    enc_in = ds.train.shape[1]

    dataset = WindowDataset(ds)
    ae = BasicWindowAutoencoder(enc_in=enc_in, latent_dim=args.latent_dim)
    train_window_autoencoder(ae, dataset, epochs=args.epochs, batch_size=16)

    plot_reconstruction_tsne(ae, dataset, save_path=os.path.join(out_dir, "recon_tsne.png"))
    plot_reconstruction_pca(ae, dataset, save_path=os.path.join(out_dir, "recon_pca.png"))
    series = ds.train[:, 0]
    plot_autoencoder_vs_series(
        ae,
        dataset,
        series,
        start=0,
        end=min(200, len(series)),
        save_path=os.path.join(out_dir, "recon_vs_series.png"),
    )


    latents, hidden, errors = collect_autoencoder_details(ae, dataset)
    np.save(os.path.join(out_dir, "latents.npy"), latents)
    np.save(os.path.join(out_dir, "hidden.npy"), hidden)
    np.save(os.path.join(out_dir, "recon_errors.npy"), errors)

    worst_idx = int(errors.argmax())
    np.save(os.path.join(out_dir, "worst_window.npy"), dataset[worst_idx][0].numpy())
    print(f"Worst reconstruction at index {worst_idx}")

    plot_vector_projection(
        latents,
        method="tsne",
        title="Latent t-SNE",
        save_path=os.path.join(out_dir, "latent_tsne.png"),
    )
    plot_vector_projection(
        latents,
        method="pca",
        title="Latent PCA",
        save_path=os.path.join(out_dir, "latent_pca.png"),
    )
    plot_vector_projection(
        hidden,
        method="tsne",
        title="Hidden t-SNE",
        save_path=os.path.join(out_dir, "hidden_tsne.png"),
    )
    plot_vector_projection(
        hidden,
        method="pca",
        title="Hidden PCA",
        save_path=os.path.join(out_dir, "hidden_pca.png"),
    )


if __name__ == "__main__":
    main()
