import pytest

np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")

from utils import analysis_tools
from utils.analysis_tools import (
    plot_memory_usage_curve,
    plot_parameter_update_efficiency,
    plot_latency_vs_model_size,
)


def test_plot_memory_usage_curve(tmp_path):
    steps = np.arange(5)
    cont = steps * 0.5
    batch = np.ones_like(steps)
    out = tmp_path / "mem.png"
    plot_memory_usage_curve(steps, cont, batch, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_parameter_update_efficiency(tmp_path):
    params = np.array([1, 2, 3])
    perf = np.array([0.1, 0.5, 0.8])
    out = tmp_path / "param.png"
    plot_parameter_update_efficiency(params, perf, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_latency_vs_model_size(tmp_path):
    sizes = np.array([10, 20, 30])
    lat = np.array([0.1, 0.2, 0.4])
    out = tmp_path / "latency.png"
    plot_latency_vs_model_size(sizes, lat, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize(
    "func,args,expected",
    [
        (plot_memory_usage_curve, (np.arange(2), np.arange(2), np.arange(2)), "memory_usage.png"),
        (plot_parameter_update_efficiency, (np.array([1, 2]), np.array([0.1, 0.2])), "param_efficiency.png"),
        (plot_latency_vs_model_size, (np.array([1, 2]), np.array([0.1, 0.2])), "latency_vs_size.png"),
    ],
)
def test_default_directory(monkeypatch, tmp_path, func, args, expected):
    out_dir = tmp_path / "viz"
    monkeypatch.setattr(analysis_tools, "DEFAULT_EFF_VIZ_DIR", str(out_dir))
    func(*args)
    out_file = out_dir / expected
    assert out_file.exists() and out_file.stat().st_size > 0
