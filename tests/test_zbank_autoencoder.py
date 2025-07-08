import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from utils.zbank_autoencoder import ZBankAutoencoder, ZBankDataset, train_autoencoder
from model.transformer_ae import AnomalyTransformerAE
from utils.analysis_tools import plot_autoencoder_vs_series


def test_autoencoder_training():
    model = AnomalyTransformerAE(
        win_size=4,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
    )
    dummy = torch.zeros(1, 4, 1)
    for _ in range(3):
        model(dummy)
    dataset = ZBankDataset(model.z_bank)
    ae = ZBankAutoencoder(latent_dim=2, enc_in=1, win_size=4)
    train_autoencoder(ae, dataset, epochs=1, batch_size=1)
    out = ae(dataset[0][0].unsqueeze(0))
    assert out.shape == (1, 4, 1)


def test_plot_autoencoder_vs_series(tmp_path):
    series = np.sin(np.linspace(0, 3.14, 40))
    model = AnomalyTransformerAE(
        win_size=10,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
    )
    tensor_series = torch.tensor(series, dtype=torch.float32).unsqueeze(-1)
    windows = [tensor_series[i : i + model.win_size] for i in range(len(series) - model.win_size + 1)]
    data = torch.stack(windows)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, torch.zeros(len(data))), batch_size=1
    )
    with torch.no_grad():
        for batch, _ in loader:
            model(batch)
    dataset = ZBankDataset(model.z_bank)
    ae = ZBankAutoencoder(latent_dim=2, enc_in=1, win_size=10)
    train_autoencoder(ae, dataset, epochs=1, batch_size=1)
    out = tmp_path / "ae_vs_series.png"
    plot_autoencoder_vs_series(ae, dataset, series, end=20, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0
