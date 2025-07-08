import pytest

np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")

from utils.analysis_tools import (
    plot_projection_by_segment,
    DEFAULT_RAW_VIZ_DIR,
)


def test_plot_projection_tsne(tmp_path):
    data = np.random.randn(100, 2)
    segments = [(0, 50), (50, 100)]
    out = tmp_path / "tsne.png"
    plot_projection_by_segment(data, segments, method="tsne", save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_projection_all_features(tmp_path):
    data = np.random.randn(50, 3)
    segments = [(0, 20), (20, 50)]
    out = tmp_path / "all_features.png"
    plot_projection_by_segment(data, segments, method="pca", feature=None, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_projection_default_directory(tmp_path, monkeypatch):
    data = np.random.randn(100, 2)
    segments = [(0, 50), (50, 100)]
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / DEFAULT_RAW_VIZ_DIR / "pca_segments.png"
    plot_projection_by_segment(data, segments, method="pca")
    assert expected.exists() and expected.stat().st_size > 0
