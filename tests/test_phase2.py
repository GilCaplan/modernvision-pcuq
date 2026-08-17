"""Phase-2: real-denoiser wrapper checks. Skipped when the checkpoint isn't
downloaded (data/ is gitignored scratch — see external/README.md for the source)."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcuq.data import corrupt, fibonacci_sphere
from pcuq.diagnostics import check_equivariance

CKPT = ROOT / "data/checkpoints/noise2score3d_step4500.pth"

pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not downloaded")


@pytest.fixture(scope="module")
def denoiser():
    from pcuq.denoisers import Noise2Score3DWrapper
    return Noise2Score3DWrapper(str(ROOT / "external/Noise2Score3D"), str(CKPT),
                                sigma=0.02, device=torch.device("cpu"))


def test_wrapper_contract(denoiser):
    y = corrupt(fibonacci_sphere(512, dtype=torch.float32)[None], 0.02, seed=0)
    with torch.no_grad():
        out = denoiser(y)
    assert out.shape == y.shape and out.dtype == y.dtype
    assert torch.isfinite(out).all()
    assert not torch.equal(out, y)


def test_real_model_is_permutation_equivariant(denoiser):
    y = corrupt(fibonacci_sphere(512, dtype=torch.float32), 0.02, seed=0)
    assert check_equivariance(denoiser, y) < 1e-5
