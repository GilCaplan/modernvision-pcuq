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


def antisym_energy(denoiser: Denoiser, y: torch.Tensor, n_probes: int = 5,
                   seed: int = 0) -> float:
    """How asymmetric is J? mean ||(J - J^T)v|| / ||(J + J^T)v|| over random probes.

    0 for an exact MMSE denoiser (covariance is symmetric); large values mean the
    implied covariance of the approximate denoiser can't be trusted un-symmetrized.
    """
    V = _unit_probes(y, n_probes, seed)
    Jv = jvp(denoiser, y, V, method="autograd")
    Jtv = vjp(denoiser, y, V)
    num = (Jv - Jtv).reshape(n_probes, -1).norm(dim=1)
    den = (Jv + Jtv).reshape(n_probes, -1).norm(dim=1)
    return float((num / den).mean())


def psd_report(eigvals: torch.Tensor) -> dict:
    """Negative-eigenvalue mass of an estimated spectrum + its PSD projection."""
    neg = eigvals[eigvals < 0]
    return {
        "n_negative": int((eigvals < 0).sum()),
        "negative_mass_ratio": float(neg.abs().sum() / eigvals.abs().sum().clamp(min=1e-30)),
        "eigvals_psd_projected": [float(v) for v in eigvals.clamp(min=0)],
    }
