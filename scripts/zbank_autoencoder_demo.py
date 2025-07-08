"""Demonstrate training an autoencoder on ``z_bank`` latents from real data."""

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
from torch.utils.data import DataLoader
from data_factory.data_loader import get_loader_segment

from model.transformer_ae import AnomalyTransformerAE
from utils.zbank_autoencoder import ZBankAutoencoder, ZBankDataset, train_autoencoder
from utils.analysis_tools import (
    plot_reconstruction_tsne,
    plot_reconstruction_pca,
    plot_autoencoder_vs_series,
)


def _load_z_bank(path: str):
    return torch.load(path)


def _load_model_weights(model: AnomalyTransformerAE, path: str) -> None:
    if os.path.isfile(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded pretrained model from {path}")
    else:
        raise FileNotFoundError(path)


def _save_z_bank(z_bank, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(z_bank, path)


def _build_z_bank(model: AnomalyTransformerAE, loader: DataLoader):
    with torch.no_grad():
        for batch, _ in loader:
            model(batch)
    return model.z_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AE on z_bank latents")
    parser.add_argument("--dataset", type=str, default="SMD", help="dataset name (SMD, SMAP, MSL, PSM)")
    parser.add_argument("--data_path", type=str, default="dataset/SMD", help="path to dataset directory")
    parser.add_argument("--win_size", type=int, default=100, help="window size")
    parser.add_argument("--latent_dim", type=int, default=4, help="latent dimension")
    parser.add_argument("--z_bank", type=str, default=None, help="optional path to load/save z_bank")
    parser.add_argument("--load_model", type=str, default=None, help="pretrained model checkpoint")
    parser.add_argument("--ae_epochs", type=int, default=10, help="autoencoder training epochs")
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

    model = AnomalyTransformerAE(
        win_size=args.win_size,
        enc_in=enc_in,
        d_model=8,
        n_heads=1,
        e_layers=1,
        d_ff=8,
        latent_dim=args.latent_dim,
        replay_size=200,
    )

    if args.load_model:
        _load_model_weights(model, args.load_model)

    if args.z_bank and os.path.isfile(args.z_bank):
        model.z_bank = _load_z_bank(args.z_bank)
    else:
        _build_z_bank(model, loader)
        if args.z_bank:
            _save_z_bank(model.z_bank, args.z_bank)
        else:
            _save_z_bank(model.z_bank, os.path.join(out_dir, "z_bank.pt"))

    dataset = ZBankDataset(model.z_bank)
    ae = ZBankAutoencoder(latent_dim=args.latent_dim, enc_in=enc_in, win_size=args.win_size)
    train_autoencoder(ae, dataset, epochs=args.ae_epochs, batch_size=16)

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


if __name__ == "__main__":
    main()
