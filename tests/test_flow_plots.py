import pytest

np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")

from utils.analysis_tools import (
    plot_latent_error_scatter,
    plot_sample_flows,
)


def test_latent_error_scatter(tmp_path):
    latents = np.random.randn(10, 4, 2)
    errors = np.linspace(0, 1, 10)
    out = tmp_path / "scatter.png"
    plot_latent_error_scatter(latents, errors, method="pca", save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_sample_flows(tmp_path):
    latents = np.random.randn(3, 5, 2)
    hidden = np.random.randn(3, 5, 3)
    outputs = np.random.randn(3, 5, 1)
    errors = np.array([0.1, 0.5, 0.2])
    out_dir = tmp_path / "flows"
    plot_sample_flows(latents, hidden, outputs, errors, indices=[1], save_dir=str(out_dir))
    expected = out_dir / "sample_1.png"
    assert expected.exists() and expected.stat().st_size > 0

