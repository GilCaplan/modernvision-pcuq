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


def smooth_eigenpairs(denoiser: Denoiser, y: torch.Tensor, x_hat: torch.Tensor,
                      sigma: float, k: int, n_basis: int = 30,
                      n_neighbors: int = 16, method: str = "central",
                      c: float = 1e-3, batch_jvp: int = 2) -> dict:
    """Covariance restricted to smooth, non-rigid displacement fields.

    The caller must freeze the denoiser graph at y before calling this function.
    Geometry is built once on x_hat. B is the orthonormal intersection of the
    low-frequency Laplacian space with the complement of rigid motion. Returned
    tensors are on CPU; basis has shape (3N, d), eigvecs has shape (k, N, 3).
    Smoothness is a restriction, not evidence of structural ambiguity. Residuals
    refer to the reduced symmetric operator, not the full denoiser Jacobian.
    """
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
    raw_basis = torch.kron(phi[:, :n_basis].contiguous(), torch.eye(3, dtype=torch.float64))

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
        "asymmetry_relative": float((covariance - covariance.T).norm() /
                                    covariance.norm().clamp_min(torch.finfo(covariance.dtype).tiny)),
        "projected_residuals": (residual / scale).tolist(),
        "roughness": roughness.tolist(),
        "negative_eigenvalues": int((all_values < -1e-10 * scale).sum()),
        "minimum_eigenvalue": float(all_values.min()),
    }
    return {"eigvecs": vectors, "eigvals": values, "basis": basis,
            "reduced_covariance": covariance, "laplacian_eigenvalues": frequencies[:n_basis],
            "graph_edges": edges, "diagnostics": diagnostics}
