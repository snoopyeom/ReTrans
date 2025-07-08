from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ZBankDataset(Dataset):
    """Dataset exposing ``(z, x)`` pairs stored in a model's ``z_bank``."""

    def __init__(self, z_bank: Sequence[dict]):
        self.z_bank = list(z_bank)

    def __len__(self) -> int:
        return len(self.z_bank)

    def __getitem__(self, idx: int):
        entry = self.z_bank[idx]
        return entry["z"].float(), entry["x"].float()


class SimpleDecoder(nn.Module):
    """Small MLP that maps latent vectors to reconstructions."""

    def __init__(self, latent_dim: int, enc_in: int, win_size: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, enc_in),
        )
        self.win_size = win_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, L, latent_dim]
        b, l, d = z.size()
        out = self.net(z.view(b * l, d))
        return out.view(b, l, -1)


class ZBankAutoencoder(nn.Module):
    """Autoencoder dedicated to reconstructing windows from ``z_bank`` latents."""

    def __init__(self, latent_dim: int, enc_in: int, win_size: int, hidden: int = 64):
        super().__init__()
        self.decoder = SimpleDecoder(latent_dim, enc_in, win_size, hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple forward
        return self.decoder(z)


def train_autoencoder(
    model: ZBankAutoencoder,
    dataset: ZBankDataset,
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> None:
    """Train ``model`` on ``dataset``."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):  # pragma: no cover - simple loop
        for z, x in loader:
            recon = model(z)
            loss = loss_fn(recon, x)
            optim.zero_grad()
            loss.backward()
            optim.step()


