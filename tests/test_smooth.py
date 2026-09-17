"""Smooth covariance against dense ground truth and saved pipeline artifacts."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcuq.data import make_toy_gaussian
from pcuq.denoisers import AnalyticGaussianDenoiser, Denoiser
from pcuq.spectrum import smooth_eigenpairs
from pcuq.utils import load_config


class Scale(Denoiser):
    def __init__(self, scale):
        self.scale = scale

    def denoise(self, y):
        return self.scale * y


def cloud(n=16):
    return torch.randn(n, 3, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(17))


def solve(den, y, **kwargs):
    return smooth_eigenpairs(den, y, y, sigma=0.2, k=3, n_basis=8,
                            n_neighbors=3, c=1e-5, **kwargs)


def test_dense_covariance_and_rigid_removal():
    y = cloud()
    toy = make_toy_gaussian(len(y), 1, torch.float64)
    den = AnalyticGaussianDenoiser(toy, 0.2)
    result = solve(den, y)
    b = result["basis"]
    expected = 0.2**2 * b.T @ den.A @ b
    assert torch.allclose(result["reduced_covariance"], expected, atol=1e-11)
    assert torch.allclose(result["eigvals"], torch.linalg.eigvalsh(expected).flip(0)[:3], atol=1e-11)
    assert torch.allclose(b.T @ b, torch.eye(b.shape[1], dtype=b.dtype), atol=1e-12)
    fields = b.T.reshape(-1, len(y), 3)
    assert fields.sum(1).abs().max() < 1e-12
    centered = y - y.mean(0)
    assert torch.linalg.cross(centered.expand_as(fields), fields).sum(1).abs().max() < 1e-12
    assert max(result["diagnostics"]["projected_residuals"]) < 1e-12
    other = solve(den, y, batch_jvp=7)
    assert torch.allclose(result["reduced_covariance"], other["reduced_covariance"], atol=1e-11)


@pytest.mark.parametrize("scale", [0.5, -0.5])
def test_isotropic_and_negative_covariance(scale):
    result = solve(Scale(scale), cloud())
    assert torch.allclose(result["eigvals"], torch.full((3,), 0.04 * scale, dtype=torch.float64))
    expected_negative = result["basis"].shape[1] if scale < 0 else 0
    assert result["diagnostics"]["negative_eigenvalues"] == expected_negative


def test_disconnected_graph_and_duplicates():
    y = cloud()
    y[:8] *= 0.01
    y[8:] = y[8:] * 0.01 + 10
    y[1] = y[0]
    result = solve(Scale(0.5), y)
    assert result["diagnostics"]["graph_components"] == 2
    assert torch.isfinite(result["basis"]).all()
    assert not (result["graph_edges"][0] == result["graph_edges"][1]).any()


def test_invalid_geometry_and_basis():
    y = cloud()
    with pytest.raises(ValueError, match="non-rigid"):
        smooth_eigenpairs(Scale(1), y, y, 0.2, k=3, n_basis=1, n_neighbors=3)
    with pytest.raises(ValueError, match="n_neighbors"):
        smooth_eigenpairs(Scale(1), y, y, 0.2, k=3, n_neighbors=len(y))
    y[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        solve(Scale(1), y)


def test_saved_pipeline_and_independent_resume(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts/run_experiment.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = load_config(str(ROOT / "configs/local.yaml"))
    cfg.update(device="cpu", double_precision=True, out_dir=str(tmp_path))
    cfg["data"].update(n_points=16, n_shapes=1)
    cfg["spectrum"].update(n_ev=2, iters=3)
    cfg["spectrum"]["smooth"].update(n_basis=8, n_neighbors=3)
    config = tmp_path / "config.json"

    def run():
        config.write_text(json.dumps(cfg))
        monkeypatch.setattr(sys, "argv", ["run_experiment", "--config", str(config)])
        runner.main()

    run()  # baseline first; adding smooth mode must not recompute the baseline
    out = tmp_path / "local/run_experiment"
    baseline = out / "toy0_sigma0.02.pt"
    baseline_time = baseline.stat().st_mtime_ns
    cfg["spectrum"]["smooth"]["enabled"] = True
    run()
    path = out / "toy0_sigma0.02_smooth.pt"
    saved = torch.load(path, weights_only=True)
    assert saved["eigvecs"].shape == (2, 16, 3)
    assert all((out / name).is_file() for name in saved["figures"])
    assert all(value.device.type == "cpu" for value in saved.values() if isinstance(value, torch.Tensor))
    assert "smooth" in json.loads((out / "metrics.json").read_text())["toy0_sigma0.02"]
    smooth_time = path.stat().st_mtime_ns
    run()
    assert path.stat().st_mtime_ns == smooth_time
    cfg["spectrum"]["smooth"]["n_basis"] = 7
    run()
    assert torch.load(path, weights_only=True)["settings"]["n_basis"] == 7
    assert baseline.stat().st_mtime_ns == baseline_time
