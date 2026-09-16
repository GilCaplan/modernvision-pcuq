"""Phase-2: real-denoiser wrapper checks. Skipped when the checkpoint isn't
downloaded (data/ is gitignored scratch — see external/README.md for the source)."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcuq.data import corrupt, fibonacci_sphere
from pcuq.denoisers import _cuda_calls_redirected
from pcuq.diagnostics import check_equivariance
from pcuq.spectrum import top_eigenpairs

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


def test_graph_topology_frozen_matches_graph_frozen_at_zero_perturbation(denoiser):
    """graph_topology_frozen recomputes coarse-level points as voxel means of the
    (possibly perturbed) input through a FIXED voxel-group assignment, instead of
    freezing those coordinates outright like graph_frozen. At the anchor itself
    (zero perturbation) the two must agree exactly -- this is the same math done a
    different way, not an approximation of it. Guards the pcuq-side reimplementation
    of grid_subsample_pack_mode's voxel hashing (denoisers.py:_voxel_group_ids) —
    origin: external/Noise2Score3D/models/easy_kpconv/ops/grid_subsample.py."""
    y = fibonacci_sphere(256, radius=1.0, dtype=torch.float32)
    lengths = torch.tensor([y.shape[0]], dtype=torch.int64)
    build_args = (y, lengths, denoiser.model.num_stages, denoiser.model.voxel_size,
                 denoiser.model.first_radius, denoiser.model.neighbor_limits)

    with denoiser.graph_frozen(), _cuda_calls_redirected(denoiser._device):
        frozen_points = denoiser._n2s3d.build_grid_and_radius_graph_pyramid(*build_args)["points"]
        frozen_points = [p.clone() for p in frozen_points]

    with denoiser.graph_topology_frozen(), _cuda_calls_redirected(denoiser._device):
        build = denoiser._n2s3d.build_grid_and_radius_graph_pyramid
        build(*build_args)  # first call in the context caches the anchor
        topo_points = build(*build_args)["points"]  # second call recomputes via
                                                     # the frozen voxel groups

    for a, b in zip(frozen_points, topo_points):
        assert torch.equal(a, b)


def test_graph_topology_frozen_gives_a_trustworthy_spectrum(denoiser):
    """Regression guard for the 2026-09-16 graph-freezing audit (docs/LOG.md):
    across 3 real ModelNet shapes, top_eigenpairs rejected graph_frozen's output
    as too asymmetric on 2/3 but never rejected graph_topology_frozen's — this
    just checks that basic claim still holds (converges, passes the symmetry gate)
    on one small anchor, not the full audit."""
    y = corrupt(fibonacci_sphere(256, dtype=torch.float32), 0.02, seed=2)
    with denoiser.graph_topology_frozen():
        with torch.no_grad():
            denoiser(y[None])
        torch.manual_seed(0)
        eigvecs, eigvals, history = top_eigenpairs(
            denoiser, y, sigma=0.02, k=2, iters=10, method="central", c=1e-3)
    assert torch.isfinite(eigvals).all()
    assert (history[-1] > 0.9).all()
