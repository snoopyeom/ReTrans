import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np


class WindowDataset(Dataset):
    """Wrap a base dataset of windows so that each item returns ``(x, x)``."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, _ = self.base[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)


class BasicWindowAutoencoder(nn.Module):
    """Simple autoencoder operating on windowed time series."""

    def __init__(self, enc_in: int, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, enc_in),
        )

    def forward(self, x: torch.Tensor, *, return_hidden: bool = False):
        """Return reconstruction and latent vectors.

        When ``return_hidden`` is ``True`` also return the decoder's hidden
        representation after the activation layer.
        """

        # x: [B, L, enc_in]
        b, l, c = x.size()
        flat = x.view(b * l, c)
        z = self.encoder(flat)
        h = self.decoder[0](z)
        h_act = self.decoder[1](h)
        recon = self.decoder[2](h_act).view(b, l, c)
        z = z.view(b, l, -1)
        if return_hidden:
            h_act = h_act.view(b, l, -1)
            return recon, z, h_act
        return recon, z


def train_window_autoencoder(
    model: BasicWindowAutoencoder,
    dataset: Dataset,
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
    for _ in range(epochs):
        for x, _ in loader:
            recon, _ = model(x)
            loss = loss_fn(recon, x)
            optim.zero_grad()
            loss.backward()
            optim.step()

def collect_autoencoder_details(model: BasicWindowAutoencoder, dataset: Dataset):
    """Return latent vectors, decoder hidden states, and per-window MSE."""

    loader = DataLoader(dataset, batch_size=1)
    latents = []
    hiddens = []
    errors = []
    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            recon, z, h = model(x, return_hidden=True)
            latents.append(z.squeeze(0).cpu())
            hiddens.append(h.squeeze(0).cpu())
            errors.append(loss_fn(recon, x).item())

    latents = torch.stack(latents).numpy()
    hiddens = torch.stack(hiddens).numpy()
    errors = np.array(errors)
    return latents, hiddens, errors

