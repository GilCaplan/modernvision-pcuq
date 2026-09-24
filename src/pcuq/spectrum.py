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


def smooth_eigenpairs(denoiser: Denoiser, y: torch.Tensor, x_hat: torch.Tensor,
                      sigma: float, k: int, n_basis: int = 30,
                      n_neighbors: int = 16, method: str = "central",
                      c: float = 1e-3, batch_jvp: int = 2,
                      symmetry_tol: float = 5e-2) -> dict:
    """Covariance restricted to smooth, non-rigid displacement fields.

    The caller must freeze the denoiser graph at y before calling this function.
    Geometry is built once on x_hat. B is the orthonormal intersection of the
    low-frequency Laplacian space with the complement of rigid motion. Returned
    tensors are on CPU; basis has shape (3N, d), eigvecs has shape (k, N, 3).
    Smoothness is a restriction, not evidence of structural ambiguity. Residuals
    refer to the reduced symmetric operator, not the full denoiser Jacobian.
    Same gates as top_eigenpairs: the denoiser must declare a covariance_kind, and a
    projected operator more asymmetric than symmetry_tol (same measure as
    _rayleigh_ritz) is rejected with ValueError instead of silently symmetrized.
    """
    if denoiser.covariance_kind not in COVARIANCE_KINDS:
        raise ValueError(
            f"{type(denoiser).__name__}.covariance_kind = {denoiser.covariance_kind!r} "
            f"is not a recognized classification — set it to one of {sorted(COVARIANCE_KINDS)}")
    if y.ndim != 2 or y.shape[1] != 3 or x_hat.shape != y.shape:
        raise ValueError("y and x_hat must have shape (N, 3)")
    if not torch.isfinite(y).all() or not torch.isfinite(x_hat).all():
        raise ValueError("point clouds must be finite")
    n = len(y)
    if not 1 <= n_neighbors < n or not 1 <= n_basis <= n:
        raise ValueError("require 1 <= n_neighbors < N and 1 <= n_basis <= N")
    if k < 1 or batch_jvp < 1 or not 0 < c < float("inf") or not 0 <= sigma < float("inf"):
        raise ValueError("require positive k, batch_jvp, c and finite nonnegative sigma")
    if method not in ("forward", "central", "autograd"):
        raise ValueError("smooth modes require a JVP method: forward, central, autograd")

    points = x_hat.detach().cpu().double()
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    neighbors = distances.argsort(dim=1, stable=True)[:, :n_neighbors]
    adjacency = torch.zeros(n, n, dtype=torch.float64)
    adjacency.scatter_(1, neighbors, 1)
    adjacency = torch.maximum(adjacency, adjacency.T)
    laplacian = adjacency.sum(1).diag() - adjacency
    frequencies, phi = torch.linalg.eigh(laplacian)
    edges = torch.nonzero(torch.triu(adjacency, diagonal=1), as_tuple=False).T
    # kron(phi_k, I3) written out: torch.kron fails on single-column slices.
    raw_basis = torch.einsum("ij,cd->icjd", phi[:, :n_basis],
                             torch.eye(3, dtype=torch.float64)).reshape(3 * n, 3 * n_basis)

    # Normalize the independent rigid fields before computing their overlap so
    # the nullspace tolerance is independent of object size (including flat or
    # collinear clouds, whose rigid-field rank can be less than six).
    centered = points - points.mean(0)
    axes = torch.eye(3, dtype=torch.float64)
    rigid = torch.stack([axis.expand_as(points) for axis in axes] +
                        [torch.linalg.cross(axis.expand_as(points), centered)
                         for axis in axes], dim=-1).reshape(3 * n, 6)
    u, s, _ = torch.linalg.svd(rigid, full_matrices=False)
    rank = int((s > s.max() * max(rigid.shape) * torch.finfo(s.dtype).eps).sum())
    overlap = u[:, :rank].T @ raw_basis
    _, singular, vh = torch.linalg.svd(overlap, full_matrices=True)
    tol = max(overlap.shape) * torch.finfo(overlap.dtype).eps
    removed = int((singular > tol).sum())
    basis = raw_basis @ vh[removed:].T
    if basis.shape[1] < k:
        raise ValueError(f"only {basis.shape[1]} non-rigid basis directions remain; "
                         "increase n_basis or reduce k")
    basis, _ = torch.linalg.qr(basis)

    columns = []
    for start in range(0, basis.shape[1], batch_jvp):
        directions = basis[:, start:start + batch_jvp].T.reshape(-1, n, 3).to(y)
        products = jvp(denoiser, y, directions, method=method, c=c)
        columns.append(basis.T @ products.detach().cpu().double().reshape(-1, 3 * n).T)
    covariance = sigma**2 * torch.cat(columns, dim=1)
    if not torch.isfinite(covariance).all():
        raise ValueError("nonfinite projected Jacobian products")
    symmetric = (covariance + covariance.T) / 2
    rel_asymmetry = float((covariance - covariance.T).norm() / symmetric.norm().clamp(min=1e-30))
    if rel_asymmetry > symmetry_tol:
        raise ValueError(
            "projected operator is not symmetric enough for covariance eigenpairs "
            f"(relative asymmetry {rel_asymmetry:.3g} > {symmetry_tol:.3g})")
    all_values, coefficients = torch.linalg.eigh(symmetric)
    values = all_values.flip(0)[:k]
    coefficients = coefficients.flip(1)[:, :k]
    vectors = (basis @ coefficients).T.reshape(k, n, 3)
    residual = (symmetric @ coefficients - coefficients * values).norm(dim=0)
    scale = symmetric.norm().clamp_min(torch.finfo(symmetric.dtype).tiny)
    roughness = torch.einsum("kic,ij,kjc->k", vectors, laplacian, vectors)
    diagnostics = {
        "basis_dimension": basis.shape[1], "rigid_rank": rank,
        "removed_dimensions": removed,
        "graph_components": int((frequencies.abs() < 1e-10).sum()),
        "asymmetry_relative": rel_asymmetry,
        "projected_residuals": (residual / scale).tolist(),
        "roughness": roughness.tolist(),
        "negative_eigenvalues": int((all_values < -1e-10 * scale).sum()),
        "minimum_eigenvalue": float(all_values.min()),
    }
    return {"eigvecs": vectors, "eigvals": values, "basis": basis,
            "reduced_covariance": covariance, "laplacian_eigenvalues": frequencies[:n_basis],
            "graph_edges": edges, "diagnostics": diagnostics}
