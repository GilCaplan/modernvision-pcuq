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
                   symmetrize: bool = False, mask: torch.Tensor = None):
    """Estimate the top-k eigenpairs of sigma^2 * J at anchor y (shape (N, 3)).

    mask: optional bool tensor — restrict the operator to that region (M J M), the
    analog of the reference repo's patch masks: "how can THIS region vary?".
    (N,) selects points of an (N, 3) cloud; a mask shaped like y (e.g. (C, H, W)
    for images) selects elements directly. Whole-signal spectra are nearly flat
    (posterior ~ isotropic); structure lives in regions. Returned eigvecs are zero
    outside the mask.

    Returns (eigvecs (k, *y.shape), eigvals (k,) descending, history) where
    history[i] is the per-vector overlap |<v_new, v_old>| at iteration i — a
    convergence signal (all -> 1 when the subspace has settled).
    """
    if mask is None:
        m = None
    else:
        m = mask.to(device=y.device, dtype=y.dtype)
        if m.shape != y.shape:
            m = m.reshape(-1, 1)  # (N,) point mask against (N, 3)

    def op(V):
        W = sym_jvp(denoiser, y, V, method, c) if symmetrize \
            else jvp(denoiser, y, V, method, c)
        return W if m is None else W * m

    V = torch.randn(k, *y.shape, device=y.device, dtype=y.dtype)
    V = _orthonormalize(V if m is None else V * m)
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
