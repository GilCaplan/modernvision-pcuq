"""Top-k eigenpairs of the posterior covariance sigma^2 * J via subspace iteration.

Clean re-derivation, for (N, 3) point clouds, of the reference implementation
(cf. external/GaussianDenoisingPosterior/moments_calculations.py:get_eigvecs).
Differences from the reference: central differences by default, optional
symmetrization, and eigenvalues from Rayleigh quotients (sign-aware) instead of
||Jv|| norms.
"""

import torch

from .denoisers import Denoiser
from .jacobian import jvp, sym_jvp


def _orthonormalize(V: torch.Tensor) -> torch.Tensor:
    """QR-orthonormalize k displacement fields. V: (k, N, 3) -> (k, N, 3).

    Runs on CPU: the matrix is only (3N, k), and QR support off-CPU (MPS) is spotty.
    """
    k = V.shape[0]
    flat = V.reshape(k, -1)
    Q, _ = torch.linalg.qr(flat.T.cpu())
    return Q.T.to(V.device).reshape(V.shape)


def top_eigenpairs(denoiser: Denoiser, y: torch.Tensor, sigma: float, k: int,
                   iters: int, method: str = "central", c: float = 1e-4,
                   symmetrize: bool = False):
    """Estimate the top-k eigenpairs of sigma^2 * J at anchor y (shape (N, 3)).

    Returns (eigvecs (k, N, 3), eigvals (k,) descending, history) where history[i]
    is the per-vector overlap |<v_new, v_old>| at iteration i — a convergence signal
    (all -> 1 when the subspace has settled).
    """
    op = (lambda V: sym_jvp(denoiser, y, V, method, c)) if symmetrize \
        else (lambda V: jvp(denoiser, y, V, method, c))

    V = _orthonormalize(torch.randn(k, *y.shape, device=y.device, dtype=y.dtype))
    history = []
    for _ in range(iters):
        V_new = _orthonormalize(op(V))
        overlap = (V_new.reshape(k, -1) * V.reshape(k, -1)).sum(dim=1).abs()
        history.append(overlap.cpu())
        V = V_new

    # Rayleigh quotients on the converged basis: lambda_i = sigma^2 * <v_i, J v_i>.
    W = op(V)
    eigvals = sigma**2 * (V.reshape(k, -1) * W.reshape(k, -1)).sum(dim=1)
    eigvals, order = eigvals.sort(descending=True, stable=True)
    return V[order], eigvals, history
