"""Frozen denoisers behind one tiny interface.

Contract: denoise(y) maps (B, N, 3) -> (B, N, 3), model frozen, point ordering
preserved (verified by diagnostics.check_equivariance before trusting a real model).

Phase 2 (TODO): Noise2Score3DWrapper around the vendored pretrained model
(cf. external/GaussianDenoisingPosterior/models_wrappers/models_wrapper_base.py for
the wrapper pattern). Fallback: ScoreDenoise — see docs/SOURCES.md.
"""

import torch

from .data import ToyGaussian


class Denoiser:
    def denoise(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return self.denoise(y)


class AnalyticGaussianDenoiser(Denoiser):
    """Exact MMSE denoiser for a known Gaussian prior X ~ N(mu, C):

        D(y) = mu + C (C + sigma^2 I)^{-1} (y - mu)

    Linear, so its Jacobian A = C(C+sigma^2 I)^{-1} is constant and
    sigma^2 * A is exactly the posterior covariance -> ground truth for Phase 1.
    """

    def __init__(self, toy: ToyGaussian, sigma: float):
        gains = toy.lams / (toy.lams + sigma**2)      # eigenvalues of A, in (0, 1)
        self.A = (toy.U * gains) @ toy.U.T            # (3N, 3N), symmetric
        self.mu_flat = toy.mu.reshape(-1)             # (3N,)
        self.sigma = sigma

    def to(self, device: torch.device) -> "AnalyticGaussianDenoiser":
        self.A = self.A.to(device)
        self.mu_flat = self.mu_flat.to(device)
        return self

    def denoise(self, y: torch.Tensor) -> torch.Tensor:
        yf = y.reshape(y.shape[0], -1)
        return (self.mu_flat + (yf - self.mu_flat) @ self.A.T).reshape(y.shape)
