import pytest
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from utils.window_autoencoder import (
    BasicWindowAutoencoder,
    WindowDataset,
    train_window_autoencoder,
    collect_autoencoder_details,
)
from utils.analysis_tools import plot_autoencoder_vs_series, plot_vector_projection


def _make_dataset(win_size: int, n_windows: int = 10):
    series = torch.sin(torch.linspace(0, 3.14, win_size + n_windows - 1)).unsqueeze(-1)
    windows = [series[i : i + win_size] for i in range(n_windows)]
    base = torch.utils.data.TensorDataset(torch.stack(windows), torch.zeros(n_windows))
    return WindowDataset(base), series.squeeze().numpy()


def test_window_autoencoder_training():
    dataset, _ = _make_dataset(win_size=5)
    ae = BasicWindowAutoencoder(enc_in=1, latent_dim=2)
    train_window_autoencoder(ae, dataset, epochs=1, batch_size=1)
    recon, z = ae(dataset[0][0].unsqueeze(0))
    assert recon.shape == (1, 5, 1)
    assert z.shape == (1, 5, 2)


def test_plot_autoencoder_vs_series(tmp_path):
    dataset, series = _make_dataset(win_size=8, n_windows=20)
    ae = BasicWindowAutoencoder(enc_in=1, latent_dim=2)
    train_window_autoencoder(ae, dataset, epochs=1, batch_size=2)
    out = tmp_path / "ae_vs_series.png"
    plot_autoencoder_vs_series(ae, dataset, series, end=20, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_collect_details_and_projection(tmp_path):
    dataset, _ = _make_dataset(win_size=6, n_windows=15)
    ae = BasicWindowAutoencoder(enc_in=1, latent_dim=2)
    train_window_autoencoder(ae, dataset, epochs=1, batch_size=2)
    lat, hid, err = collect_autoencoder_details(ae, dataset)
    assert lat.shape[0] == len(dataset)
    assert hid.shape[0] == len(dataset)
    assert err.shape == (len(dataset),)
    out = tmp_path / "vec.png"
    plot_vector_projection(lat, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0
