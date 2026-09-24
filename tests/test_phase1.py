"""Phase-1 numerics: JVPs and subspace iteration vs closed-form ground truth.

CPU, float64, tiny N — runs in seconds.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcuq.data import make_toy_gaussian, make_toy_gmm
from pcuq.denoisers import (EXACT_MMSE_FIXED_SIGMA, AnalyticGaussianDenoiser,
                            AnalyticGMMDenoiser, Denoiser)
from pcuq.diagnostics import (antisym_energy, antisym_energy_fd, check_equivariance,
                              is_trustworthy)
from pcuq.jacobian import jvp, vjp
from pcuq.spectrum import _rayleigh_ritz, top_eigenpairs

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


def _gmm_setup(seed=7, n_points=6, n_components=3, mean_sep=0.06, amp=1e-5):
    # amp << SIGMA^2 keeps each component's own posterior gain near 0, so a
    # component's estimate m_k(y) stays close to mu_k instead of collapsing
    # toward y -- separation mean_sep > ~2*SIGMA then survives into m_0(y) -
    # m_1(y), which is what makes the between-component term below exceed SIGMA^2.
    gmm = make_toy_gmm(n_points, seed, dtype=torch.float64, n_components=n_components,
                       mean_sep=mean_sep, amp=amp)
    den = AnalyticGMMDenoiser(gmm, SIGMA)
    # The midpoint between two component means is a genuinely ambiguous anchor
    # (neither component is clearly "the" mode) -- exactly where mode-membership
    # uncertainty should push eigenvalues above the single-Gaussian sigma^2 bound.
    y = 0.5 * (gmm.mu[0] + gmm.mu[1])
    return gmm, den, y


def test_gmm_denoiser_matches_analytic_posterior_mean():
    gmm, den, y = _gmm_setup()
    out = den.denoise(y[None])[0]
    mean, _ = gmm.posterior_mean_and_cov(y, SIGMA)
    assert torch.allclose(out, mean, atol=1e-10)


def test_gmm_toy_has_eigenvalues_above_sigma_squared():
    gmm, _, y = _gmm_setup()
    _, true_vals = gmm.posterior_eigenpairs(y, SIGMA, k=2)
    # The single-Gaussian toy (ToyGaussian) always has eigenvalues <= sigma^2 --
    # this mixture toy is the correction: mode-membership ambiguity at this anchor
    # legitimately exceeds that bound.
    assert true_vals[0] > SIGMA**2


def test_gmm_numeric_spectrum_matches_analytic_ground_truth():
    gmm, den, y = _gmm_setup()
    true_vecs, true_vals = gmm.posterior_eigenpairs(y, SIGMA, k=2)

    torch.manual_seed(9)
    eigvecs, eigvals, history = top_eigenpairs(den, y, SIGMA, k=2, iters=80,
                                               method="autograd")
    assert torch.allclose(eigvals, true_vals, rtol=1e-3)
    overlap = (eigvecs.reshape(2, -1) * true_vecs.reshape(2, -1)).sum(dim=1).abs()
    assert (overlap > 0.999).all()
    assert (history[-1] > 0.999).all(), "iteration should have converged"


class _PointwiseShrink(Denoiser):
    def denoise(self, y):
        return 0.5 * y


class _LinearDenoiser(Denoiser):
    """Small dense linear map used to test spectral numerics directly."""

    covariance_kind = EXACT_MMSE_FIXED_SIGMA  # A is exact ground truth by construction

    def __init__(self, A: torch.Tensor):
        self.A = A

    def denoise(self, y):
        flat = y.reshape(y.shape[0], -1)
        return (flat @ self.A.T).reshape_as(y)


def _symmetric_matrix(lams: torch.Tensor, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(len(lams), len(lams), generator=gen,
                                      dtype=torch.float64))
    return (Q * lams) @ Q.T, Q


def test_rayleigh_ritz_rotates_a_converged_subspace_into_eigenvectors():
    lams = torch.tensor([6.0, 4.0, 1.0, 0.5, 0.25, 0.1], dtype=torch.float64)
    A, Q = _symmetric_matrix(lams)
    # A deliberately rotated basis spans the top-2 invariant subspace but is not
    # itself an eigenbasis; this is exactly the case QR iteration can leave us in.
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=torch.float64)
    V = (rotation @ Q[:, :2].T).reshape(2, 2, 3)
    W = (V.reshape(2, -1) @ A.T).reshape_as(V)

    eigvecs, eigvals = _rayleigh_ritz(V, W, sigma=0.2, symmetry_tol=1e-12)
    assert torch.allclose(eigvals, 0.2**2 * lams[:2], rtol=1e-12, atol=1e-12)
    residual = (eigvecs.reshape(2, -1) @ A.T
                - (eigvals / 0.2**2)[:, None] * eigvecs.reshape(2, -1))
    assert residual.norm() < 1e-11


def test_indefinite_symmetric_operator_returns_valid_signed_ritz_pairs():
    lams = torch.tensor([6.0, 3.0, -0.5, -0.25, 0.1, 0.05], dtype=torch.float64)
    A, _ = _symmetric_matrix(lams)
    den = _LinearDenoiser(A)
    y = torch.zeros(2, 3, dtype=torch.float64)
    torch.manual_seed(4)
    eigvecs, eigvals, _ = top_eigenpairs(den, y, sigma=0.3, k=3, iters=100,
                                         method="autograd")
    assert torch.allclose(eigvals, 0.3**2 * lams[:3], rtol=1e-9, atol=1e-10)
    residual = (eigvecs.reshape(3, -1) @ A.T
                - (eigvals / 0.3**2)[:, None] * eigvecs.reshape(3, -1))
    assert residual.norm() < 1e-8


def test_nearly_repeated_eigenvalues_are_resolved_by_rayleigh_ritz():
    lams = torch.tensor([5.0, 4.99, 1.0, 0.5, 0.25, 0.1], dtype=torch.float64)
    A, _ = _symmetric_matrix(lams)
    den = _LinearDenoiser(A)
    y = torch.zeros(2, 3, dtype=torch.float64)
    torch.manual_seed(5)
    eigvecs, eigvals, _ = top_eigenpairs(den, y, sigma=0.1, k=2, iters=80,
                                         method="autograd")
    assert torch.allclose(eigvals, 0.1**2 * lams[:2], rtol=1e-9, atol=1e-10)
    residual = (eigvecs.reshape(2, -1) @ A.T
                - (eigvals / 0.1**2)[:, None] * eigvecs.reshape(2, -1))
    assert residual.norm() < 1e-8


def test_materially_nonsymmetric_operator_is_rejected_as_covariance():
    A = torch.tensor([[4.0, 8.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5]],
                     dtype=torch.float64)
    den = _LinearDenoiser(A)
    y = torch.zeros(1, 3, dtype=torch.float64)
    torch.manual_seed(6)
    with pytest.raises(ValueError, match="not symmetric enough"):
        top_eigenpairs(den, y, sigma=0.2, k=2, iters=30, method="autograd",
                       symmetry_tol=1e-6)


def test_denoiser_without_covariance_kind_is_rejected():
    class _Unclassified(_LinearDenoiser):
        covariance_kind = None

    den = _Unclassified(torch.eye(3, dtype=torch.float64))
    y = torch.zeros(1, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="covariance_kind"):
        top_eigenpairs(den, y, sigma=0.2, k=1, iters=5, method="autograd")


def test_equivariance_check():
    _, den, y = _setup()
    assert check_equivariance(_PointwiseShrink(), y) < 1e-12  # equivariant by design
    assert check_equivariance(den, y) > 1e-3  # full-cov analytic denoiser is not


def _good_run():
    return {"final_iter_overlap": [0.99, 0.98], "ritz_relative_residuals": [0.01, 0.02]}


def test_is_trustworthy_accepts_a_clean_run():
    assert is_trustworthy(_good_run())


def test_is_trustworthy_rejects_a_rejected_run():
    m = _good_run()
    m["top_eigenpairs_rejected"] = "too asymmetric"
    assert not is_trustworthy(m)


def test_is_trustworthy_rejects_poor_convergence():
    m = _good_run()
    m["final_iter_overlap"] = [0.99, 0.5]  # one mode never converged
    assert not is_trustworthy(m)


def test_is_trustworthy_rejects_high_ritz_residual():
    m = _good_run()
    m["ritz_relative_residuals"] = [0.01, 0.5]  # one mode's Ritz pair is unreliable
    assert not is_trustworthy(m)
