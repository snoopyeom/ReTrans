"""Demonstrate CPD and latent space visualizations on synthetic data."""

# Allow running this script directly from ``scripts/`` by adding the project
# root to ``sys.path`` so that ``utils`` can be imported without a module error.
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

missing = []
for _mod in ["numpy", "torch", "sklearn", "matplotlib", "ruptures"]:
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
from torch.utils.data import DataLoader, TensorDataset

from utils.analysis_tools import (
    visualize_cpd_detection,
    plot_z_bank_tsne,
    plot_z_bank_pca,
)
from model.transformer_ae import AnomalyTransformerAE


def create_synthetic_series(n_steps=400):
    """Return a toy time series with a distribution shift."""
    first = np.random.normal(0.0, 1.0, (n_steps // 2, 1))
    second = np.random.normal(3.0, 1.0, (n_steps - n_steps // 2, 1))
    return np.concatenate([first, second], axis=0)


def main():
    series = create_synthetic_series()
    visualize_cpd_detection(
        series.squeeze(),
        penalty=20,
        save_path="cpd_demo.png",
        top_k=1,
        extra_zoom_ranges=[(0, 4000)],
    )

    # minimal model to generate latent vectors
    model = AnomalyTransformerAE(
        win_size=20,
        enc_in=1,
        d_model=8,
        n_heads=1,
        e_layers=1,
        d_ff=8,
        latent_dim=4,
        replay_size=50,
    )
    tensor_series = torch.tensor(series, dtype=torch.float32)
    windows = [tensor_series[i:i + model.win_size]
               for i in range(len(tensor_series) - model.win_size + 1)]
    data = torch.stack(windows)
    labels = torch.zeros(len(data))  # dummy labels required by analysis utils
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=1)

    with torch.no_grad():
        for batch in loader:
            model(batch[0])

    plot_z_bank_tsne(model, loader, save_path="tsne_demo.png")
    plot_z_bank_pca(model, loader, save_path="pca_demo.png")


if __name__ == "__main__":
    main()
