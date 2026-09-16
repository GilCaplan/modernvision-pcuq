"""Top-k eigenpairs of the posterior covariance sigma^2 * J via subspace iteration.

Clean re-derivation, for (N, 3) point clouds, of the reference implementation
(cf. external/GaussianDenoisingPosterior/moments_calculations.py:get_eigvecs).
Differences from the reference: central differences by default, optional
symmetrization, and Rayleigh--Ritz finalization of the converged subspace instead
of treating arbitrary QR basis vectors as eigenvectors.
"""

import torch

from .denoisers import COVARIANCE_KINDS, Denoiser
from .jacobian import jvp, sym_jvp


def _orthonormalize(V: torch.Tensor) -> torch.Tensor:
    """QR-orthonormalize k displacement fields. V: (k, N, 3) -> (k, N, 3).

    Runs on CPU: the matrix is only (3N, k), and QR support off-CPU (MPS) is spotty.
    """
    k = V.shape[0]
    flat = V.reshape(k, -1)
    Q, _ = torch.linalg.qr(flat.T.cpu())
    return Q.T.to(V.device).reshape(V.shape)


def _rayleigh_ritz(V: torch.Tensor, W: torch.Tensor, sigma: float,
                   symmetry_tol: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonalize the operator restricted to the converged subspace.

    QR iteration identifies an invariant subspace, but its individual rows are
    generally not eigenvectors. Rayleigh--Ritz diagonalization of V A V^T
    produces the corresponding Ritz eigenpairs.

    A posterior covariance is symmetric. A small tolerance allows finite-
    difference roundoff; a materially asymmetric projected operator must be
    treated as a sensitivity operator, not as a covariance.
    """
    flat_v, flat_w = V.reshape(V.shape[0], -1), W.reshape(W.shape[0], -1)
    projected = sigma**2 * (flat_v @ flat_w.T)
    symmetric_part = 0.5 * (projected + projected.T)
    rel_asymmetry = (projected - projected.T).norm() / symmetric_part.norm().clamp(min=1e-30)
    if float(rel_asymmetry) > symmetry_tol:
        raise ValueError(
            "projected operator is not symmetric enough for covariance eigenpairs "
            f"(relative asymmetry {float(rel_asymmetry):.3g} > {symmetry_tol:.3g}); "
            "use a symmetrized JVP/VJP operator or report sensitivity modes instead"
        )

    # This is only k x k. CPU matches the QR backend and avoids accelerator
    # compatibility differences in the final eigendecomposition.
    vals, coeffs = torch.linalg.eigh(symmetric_part.cpu())
    order = torch.argsort(vals, descending=True, stable=True)
    vals = vals[order].to(V.device)
    coeffs = coeffs[:, order].T.to(V.device)
    eigvecs = (coeffs @ flat_v).reshape_as(V)
    return eigvecs, vals


def _principal_angles_degrees(V_new: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Principal angles between two orthonormal row-subspaces, in degrees."""
    flat_new, flat = V_new.reshape(V_new.shape[0], -1), V.reshape(V.shape[0], -1)
    cosines = torch.linalg.svdvals((flat_new @ flat.T).cpu()).clamp(min=-1, max=1)
    return torch.rad2deg(torch.acos(cosines))


def _relative_ritz_residuals(eigvecs: torch.Tensor, eigvals: torch.Tensor,
                             W: torch.Tensor, sigma: float) -> torch.Tensor:
    """Per-mode relative residuals for the covariance operator sigma^2 J."""
    covariance_products = sigma**2 * W.reshape(W.shape[0], -1)
    flat_vecs = eigvecs.reshape(eigvecs.shape[0], -1)
    residuals = covariance_products - eigvals[:, None] * flat_vecs
    scale = torch.maximum(covariance_products.norm(dim=1), eigvals.abs()).clamp(min=1e-30)
    return residuals.norm(dim=1) / scale


def top_eigenpairs(denoiser: Denoiser, y: torch.Tensor, sigma: float, k: int,
                   iters: int, method: str = "central", c: float = 1e-4,
                   symmetrize: bool = False, mask: torch.Tensor = None,
                   symmetry_tol: float = 5e-2, return_diagnostics: bool = False):
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
    if denoiser.covariance_kind not in COVARIANCE_KINDS:
        raise ValueError(
            f"{type(denoiser).__name__}.covariance_kind = {denoiser.covariance_kind!r} "
            "is not a recognized classification — set it to one of "
            f"{sorted(COVARIANCE_KINDS)} (see pcuq.denoisers) before extracting a "
            "spectrum, so callers know whether sigma^2 * J here is an MMSE posterior "
            "covariance or only a local sensitivity operator"
        )
    # For an indefinite approximate operator, subspace iteration identifies the
    # largest-magnitude invariant subspace. Signed outputs are diagnostics, not
    # posterior-covariance eigenvalues.
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
    principal_angle_history = []
    for _ in range(iters):
        V_new = _orthonormalize(op(V))
        overlap = (V_new.reshape(k, -1) * V.reshape(k, -1)).sum(dim=1).abs()
        history.append(overlap.cpu())
        principal_angle_history.append(_principal_angles_degrees(V_new, V))
        V = V_new

    # Finalize the converged invariant subspace into individual Ritz eigenpairs.
    W = op(V)
    eigvecs, eigvals = _rayleigh_ritz(V, W, sigma, symmetry_tol)
    ritz_residuals = _relative_ritz_residuals(eigvecs, eigvals, op(eigvecs), sigma)
    if return_diagnostics:
        return eigvecs, eigvals, history, {
            "subspace_principal_angles_degrees": principal_angle_history,
            "ritz_relative_residuals": ritz_residuals.cpu(),
        }
    return eigvecs, eigvals, history
