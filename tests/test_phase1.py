"""Phase-1 numerics: JVPs and subspace iteration vs closed-form ground truth.

CPU, float64, tiny N — runs in seconds.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcuq.data import make_toy_gaussian
from pcuq.denoisers import AnalyticGaussianDenoiser, Denoiser
from pcuq.diagnostics import antisym_energy, antisym_energy_fd, check_equivariance
from pcuq.jacobian import jvp, vjp
from pcuq.spectrum import top_eigenpairs

SIGMA = 0.02


def _setup(n_points=16, seed=0):
    toy = make_toy_gaussian(n_points, seed, dtype=torch.float64)
    den = AnalyticGaussianDenoiser(toy, SIGMA)
    y = toy.mu + 0.01 * torch.randn(n_points, 3, generator=torch.Generator().manual_seed(1),
                                    dtype=torch.float64)
    return toy, den, y


def test_jvp_matches_analytic_jacobian():
    _, den, y = _setup()
    v = torch.randn(3, *y.shape, generator=torch.Generator().manual_seed(2),
                    dtype=torch.float64)
    expected = (v.reshape(3, -1) @ den.A.T).reshape(v.shape)
    for method in ("forward", "central", "autograd"):
        got = jvp(den, y, v, method=method, c=1e-6)
        assert torch.allclose(got, expected, rtol=1e-4), method
    assert torch.allclose(vjp(den, y, v), expected, rtol=1e-6)  # A symmetric


def test_subspace_iteration_recovers_posterior_eigenpairs():
    toy, den, y = _setup()
    k = 3
    torch.manual_seed(0)
    eigvecs, eigvals, history = top_eigenpairs(den, y, SIGMA, k=k, iters=60,
                                               method="central", c=1e-5)
    true_vecs, true_vals = toy.posterior_eigenpairs(SIGMA, k)
    assert torch.allclose(eigvals, true_vals, rtol=1e-3)
    overlap = (eigvecs.reshape(k, -1) * true_vecs.reshape(k, -1)).sum(dim=1).abs()
    assert (overlap > 0.999).all()
    assert (history[-1] > 0.9999).all(), "iteration should have converged"


def test_symmetrized_matches_plain_for_symmetric_jacobian():
    toy, den, y = _setup()
    torch.manual_seed(0)
    _, vals_plain, _ = top_eigenpairs(den, y, SIGMA, k=2, iters=40, symmetrize=False)
    torch.manual_seed(0)
    _, vals_sym, _ = top_eigenpairs(den, y, SIGMA, k=2, iters=40, symmetrize=True)
    assert torch.allclose(vals_plain, vals_sym, rtol=1e-6)
    assert antisym_energy(den, y) < 1e-10


def test_antisym_fd_matches_autograd_probe_for_symmetric_jacobian():
    _, den, y = _setup()
    V = torch.randn(3, *y.shape, generator=torch.Generator().manual_seed(3),
                    dtype=torch.float64)
    V, _ = torch.linalg.qr(V.reshape(3, -1).T)
    V = V.T.reshape(3, *y.shape)
    assert antisym_energy_fd(den, y, V, c=1e-6) < 1e-8  # A symmetric -> ~0


def test_masked_spectrum_matches_dense_restriction():
    toy, den, y = _setup()
    mask = torch.zeros(16, dtype=torch.bool)
    mask[:8] = True
    # Ground truth: eigendecomposition of the dense restricted operator M A M.
    M = mask.to(torch.float64).repeat_interleave(3).diag()
    dense = SIGMA**2 * (M @ den.A @ M)
    true_vals, true_vecs = torch.linalg.eigh((dense + dense.T) / 2)
    true_vals, idx = true_vals.sort(descending=True)
    true_vecs = true_vecs[:, idx]

    torch.manual_seed(0)
    eigvecs, eigvals, _ = top_eigenpairs(den, y, SIGMA, k=3, iters=60,
                                         method="central", c=1e-5, mask=mask)
    assert torch.allclose(eigvals, true_vals[:3], rtol=1e-3)
    assert (eigvecs[:, ~mask].abs() < 1e-12).all(), "eigvecs must vanish off-mask"
    overlap = (eigvecs.reshape(3, -1) * true_vecs[:, :3].T.reshape(3, -1)).sum(1).abs()
    assert (overlap > 0.999).all()


class _PointwiseShrink(Denoiser):
    def denoise(self, y):
        return 0.5 * y


def test_equivariance_check():
    _, den, y = _setup()
    assert check_equivariance(_PointwiseShrink(), y) < 1e-12  # equivariant by design
    assert check_equivariance(den, y) > 1e-3  # full-cov analytic denoiser is not
