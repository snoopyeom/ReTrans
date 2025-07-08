import pytest

np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")

from utils.analysis_tools import (
    plot_feature_distribution_by_segment,
    plot_rolling_stats,
    DEFAULT_RAW_VIZ_DIR,
)


def test_plot_feature_distribution_default(tmp_path, monkeypatch):
    data = np.random.randn(100, 2)
    segments = [(0, 50), (50, 100)]
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / DEFAULT_RAW_VIZ_DIR / "feature_dist.png"
    plot_feature_distribution_by_segment(data, segments, feature=0)
    assert expected.exists() and expected.stat().st_size > 0


def test_plot_rolling_stats_default(tmp_path, monkeypatch):
    data = np.random.randn(100)
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / DEFAULT_RAW_VIZ_DIR / "rolling_stats.png"
    plot_rolling_stats(data, window=10)
    assert expected.exists() and expected.stat().st_size > 0
