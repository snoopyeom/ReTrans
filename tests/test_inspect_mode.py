import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from model.transformer_ae import AnomalyTransformerAE
from utils.analysis_tools import (
    plot_latent_tsne,
    plot_latent_pca,
    plot_error_curve,
    plot_reconstruction_per_sample,
)


def test_forward_return_hidden(tmp_path):
    model = AnomalyTransformerAE(
        win_size=4,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
    )
    batch = torch.zeros(2, 4, 1)
    recon, _, _, z, hidden = model(batch, return_hidden=True)
    assert z.shape == (2, 4, 2)
    assert hidden.shape[:2] == (2, 4)

    latents = z.view(-1, 2).detach().numpy()
    plot_latent_tsne(latents, save_path=str(tmp_path / "tsne.png"))
    plot_latent_pca(latents, save_path=str(tmp_path / "pca.png"))
    errors = ((recon - batch) ** 2).mean(dim=(1, 2)).detach().numpy()
    plot_error_curve(errors, save_path=str(tmp_path / "err.png"))
    plot_reconstruction_per_sample(
        batch.numpy(), recon.detach().numpy(), save_path=str(tmp_path / "rec.png")
    )
    for name in ["tsne.png", "pca.png", "err.png", "rec.png"]:
        p = tmp_path / name
        assert p.exists() and p.stat().st_size > 0
