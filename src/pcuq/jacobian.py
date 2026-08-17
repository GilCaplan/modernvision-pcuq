"""Jacobian-vector products against a frozen denoiser at anchor y.

J = dD(y)/dy is (3N x 3N) but never materialized: every product is one or two
denoiser forward passes (cf. external/GaussianDenoisingPosterior/
moments_calculations.py:_forward_directional). Shapes: anchor y is (N, 3), a block
of k directions v is (k, N, 3); the denoiser batches over the leading dim.
"""

import torch

from .denoisers import Denoiser


def jvp(denoiser: Denoiser, y: torch.Tensor, v: torch.Tensor,
        method: str = "central", c: float = 1e-4) -> torch.Tensor:
    """J v for each direction in v: (k, N, 3) -> (k, N, 3).

    Directions should be O(unit norm); `c` scales the actual perturbation so the
    linear approximation holds (sweep via diagnostics.sweep_step_size).
    """
    if method == "forward":
        with torch.no_grad():
            return (denoiser(y + c * v) - denoiser(y[None])) / c
    if method == "central":
        with torch.no_grad():
            return (denoiser(y + c * v) - denoiser(y - c * v)) / (2 * c)
    if method == "autograd":
        f = lambda x: denoiser(x[None])[0]
        return torch.stack([torch.func.jvp(f, (y,), (vi,))[1] for vi in v])
    raise ValueError(f"unknown jvp method: {method}")


def vjp(denoiser: Denoiser, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """J^T v for each direction in v (autograd only): (k, N, 3) -> (k, N, 3)."""
    f = lambda x: denoiser(x[None])[0]
    _, pullback = torch.func.vjp(f, y)
    return torch.stack([pullback(vi)[0] for vi in v])


def sym_jvp(denoiser: Denoiser, y: torch.Tensor, v: torch.Tensor,
            method: str = "central", c: float = 1e-4) -> torch.Tensor:
    """(J + J^T)/2 applied to v. The true posterior covariance is symmetric; an
    approximate denoiser's J need not be — this is the operator we diagonalize
    when cfg spectrum.symmetrize is on (J^T v always via autograd)."""
    return 0.5 * (jvp(denoiser, y, v, method, c) + vjp(denoiser, y, v))
