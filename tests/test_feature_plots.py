import pytest

np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")

from utils.analysis_tools import (
    plot_feature_distribution_by_segment,
    plot_rolling_stats,
)


def test_plot_feature_distribution(tmp_path):
    data = np.random.randn(100, 2)
    out = tmp_path / "feature.png"
    plot_feature_distribution_by_segment(data, [(0, 50), (50, 100)], 0, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_rolling_stats(tmp_path):
    series = np.random.randn(100)
    out = tmp_path / "rolling.png"
    plot_rolling_stats(series, window=10, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0
