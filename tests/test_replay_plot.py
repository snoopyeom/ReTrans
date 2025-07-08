import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from utils.analysis_tools import plot_replay_vs_series
from model.transformer_ae import AnomalyTransformerAE


def test_plot_replay_vs_series(tmp_path):
    series = np.sin(np.linspace(0, 3.14, 40))
    model = AnomalyTransformerAE(
        win_size=10,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
        replay_size=100,
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
    out = tmp_path / "replay.png"
    plot_replay_vs_series(model, series, end=20, save_path=str(out), ordered=True)
    assert out.exists() and out.stat().st_size > 0

