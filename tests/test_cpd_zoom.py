import pytest

np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("ruptures")

from utils.analysis_tools import visualize_cpd_detection


def test_zoom_range(tmp_path):
    series = np.sin(np.linspace(0, 10, 100))
    out = tmp_path / "cpd.png"
    visualize_cpd_detection(series, zoom_range=(20, 40), save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_top_k(tmp_path):
    series = np.concatenate([
        np.random.normal(0.0, 1.0, 50),
        np.random.normal(5.0, 1.0, 50),
    ])
    out = tmp_path / "cpd.png"
    visualize_cpd_detection(series, penalty=10, save_path=str(out), top_k=1)
    zoom = tmp_path / "cpd_top1.png"
    assert out.exists() and zoom.exists()


def test_extra_ranges(tmp_path):
    series = np.sin(np.linspace(0, 10, 100))
    out = tmp_path / "cpd.png"
    visualize_cpd_detection(series, save_path=str(out), extra_zoom_ranges=[(0, 4000)])
    visualize_cpd_detection(series, save_path=str(out), extra_zoom_ranges=[(0, 20)])
    zoom = tmp_path / "cpd_range1.png"
    assert out.exists() and zoom.exists()
