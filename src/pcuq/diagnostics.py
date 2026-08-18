"""The proposal's challenge checks, as runnable diagnostics.

Each returns plain floats/dicts ready for a metrics JSON and docs/LOG.md.
"""

import torch

from .denoisers import Denoiser
from .jacobian import jvp, vjp


def _unit_probes(y: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(n, *y.shape, generator=gen, dtype=y.dtype).to(y.device)
    return v / v.reshape(n, -1).norm(dim=1).reshape(n, 1, 1)


def check_equivariance(denoiser: Denoiser, y: torch.Tensor, seed: int = 0) -> float:
    """Relative error of D(P y) vs P D(y) for a random permutation P of the points.

    ~0 for a permutation-equivariant, ordering-preserving denoiser. Hard gate for
    real models (docs/PLAN.md Phase 2): if this is large, correspondence-based
    finite differences are meaningless.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(y.shape[0], generator=gen).to(y.device)
    with torch.no_grad():
        out = denoiser(y[None])[0]
        out_perm = denoiser(y[perm][None])[0]
    return float((out_perm - out[perm]).norm() / out.norm())


def sweep_step_size(denoiser: Denoiser, y: torch.Tensor, cs: list[float],
                    method: str = "central", seed: int = 0) -> dict[float, float]:
    """Relative JVP error vs step size c, against the exact autograd JVP.

    Returns {c: rel_err}. Pick c on the plateau: too large -> nonlinearity error,
    too small -> floating-point cancellation.
    """
    v = _unit_probes(y, 1, seed)
    ref = jvp(denoiser, y, v, method="autograd")
    ref_norm = float(ref.norm())
    return {float(c): float((jvp(denoiser, y, v, method=method, c=c) - ref).norm()) / ref_norm
            for c in cs}


def sweep_step_size_fd(denoiser: Denoiser, y: torch.Tensor, cs: list[float],
                       method: str = "central", seed: int = 0) -> dict[float, float]:
    """Autograd-free variant for models torch.func can't trace (Noise2Score3D):
    self-consistency error ||J_c v - J_{c/2} v|| / ||J_{c/2} v|| per step size c.
    Small on the plateau where the finite difference is trustworthy."""
    v = _unit_probes(y, 1, seed)
    out = {}
    for c in cs:
        g = jvp(denoiser, y, v, method=method, c=c)
        g_half = jvp(denoiser, y, v, method=method, c=c / 2)
        out[float(c)] = float((g - g_half).norm() / g_half.norm())
    return out


def antisym_energy(denoiser: Denoiser, y: torch.Tensor, n_probes: int = 5,
                   seed: int = 0) -> float:
    """How asymmetric is J? mean ||(J - J^T)v|| / ||(J + J^T)v|| over random probes.

    0 for an exact MMSE denoiser (covariance is symmetric); large values mean the
    implied covariance of the approximate denoiser can't be trusted un-symmetrized.
    Needs J^T v via autograd — use antisym_energy_fd for models torch.func can't trace.
    """
    V = _unit_probes(y, n_probes, seed)
    Jv = jvp(denoiser, y, V, method="autograd")
    Jtv = vjp(denoiser, y, V)
    num = (Jv - Jtv).reshape(n_probes, -1).norm(dim=1)
    den = (Jv + Jtv).reshape(n_probes, -1).norm(dim=1)
    return float((num / den).mean())


def antisym_energy_fd(denoiser: Denoiser, y: torch.Tensor, V: torch.Tensor,
                      method: str = "central", c: float = 1e-4) -> float:
    """Forward-passes-only asymmetry probe (for models autograd can't trace).

    Restricted to the subspace spanned by the orthonormal fields V (k, N, 3) —
    typically the converged eigenvectors: form the bilinear matrix
    A_ij = <v_i, J v_j> and return ||A - A^T||_F / ||A + A^T||_F. Asymmetry of J
    *within the reported uncertainty subspace* is exactly what makes the covariance
    estimate untrustworthy, so this is the part worth monitoring.
    """
    k = V.shape[0]
    W = jvp(denoiser, y, V, method=method, c=c)
    A = V.reshape(k, -1) @ W.reshape(k, -1).T
    return float((A - A.T).norm() / (A + A.T).norm())


def psd_report(eigvals: torch.Tensor) -> dict:
    """Negative-eigenvalue mass of an estimated spectrum + its PSD projection."""
    neg = eigvals[eigvals < 0]
    return {
        "n_negative": int((eigvals < 0).sum()),
        "negative_mass_ratio": float(neg.abs().sum() / eigvals.abs().sum().clamp(min=1e-30)),
        "eigvals_psd_projected": [float(v) for v in eigvals.clamp(min=0)],
    }
