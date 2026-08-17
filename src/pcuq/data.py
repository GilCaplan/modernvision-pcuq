"""Datasets and corruption.

Phase 1: synthetic Gaussian toy data with closed-form posterior (below).
Phase 2 (TODO): load_modelnet(cfg) — ModelNet40 download/cache, mesh -> n_points
sampling via trimesh, unit-sphere normalization. Variant recorded in docs/SOURCES.md.
"""

import math

import torch


def corrupt(x: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Gaussian corruption Y = X + sigma*Z, retaining exact point indices."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(x.shape, generator=gen, dtype=x.dtype).to(x.device)
    return x + sigma * z


def fibonacci_sphere(n_points: int, radius: float = 1.0,
                     dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Deterministic near-uniform points on a sphere, (N, 3)."""
    i = torch.arange(n_points, dtype=torch.float64)
    phi = torch.acos(1 - 2 * (i + 0.5) / n_points)
    theta = math.pi * (1 + 5**0.5) * i
    p = torch.stack([phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()], dim=1)
    return (radius * p).to(dtype)


class ToyGaussian:
    """Gaussian prior X ~ N(mu, C) over point clouds, C = U diag(lams) U^T known.

    For Y = X + sigma*Z the posterior is Gaussian with
        Cov[X|Y] = sigma^2 C (C + sigma^2 I)^{-1},
    which shares eigenvectors with C and has eigenvalues sigma^2*lam/(lam+sigma^2).
    This gives exact ground truth for everything spectrum.py estimates.
    """

    def __init__(self, mu: torch.Tensor, U: torch.Tensor, lams: torch.Tensor):
        self.mu = mu      # (N, 3) base shape
        self.U = U        # (3N, 3N) orthogonal; columns are prior eigenvectors
        self.lams = lams  # (3N,) prior eigenvalues, descending

    def sample(self, n_shapes: int, seed: int) -> torch.Tensor:
        d = self.lams.numel()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(n_shapes, d, generator=gen, dtype=self.mu.dtype)
        x = (z * self.lams.clamp(min=0).sqrt()) @ self.U.T + self.mu.reshape(-1)
        return x.reshape(n_shapes, *self.mu.shape)

    def posterior_eigenpairs(self, sigma: float, k: int):
        """Top-k eigenpairs of Cov[X|Y] in closed form -> (vecs (k,N,3), vals (k,))."""
        post = sigma**2 * self.lams / (self.lams + sigma**2)
        vals, idx = post.sort(descending=True, stable=True)
        vecs = self.U[:, idx[:k]].T.reshape(k, *self.mu.shape)
        return vecs, vals[:k]


def make_toy_gaussian(n_points: int, seed: int, dtype: torch.dtype = torch.float32,
                      amp: float = 1e-2, decay: float = 0.1) -> ToyGaussian:
    """Toy prior around a unit-ish sphere with a strongly separated spectrum.

    lams[i] = amp * decay^i decays fast so the top eigenpairs are well-gapped and
    subspace iteration converges in few iterations (amp is chosen so the leading
    lams straddle typical sigma^2 values — that's where posterior eigenvalues
    remain separated instead of all saturating at sigma^2).
    """
    mu = fibonacci_sphere(n_points, radius=0.5)

    d = 3 * n_points
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(d, d, generator=gen, dtype=torch.float64)
    U, _ = torch.linalg.qr(G)
    lams = amp * torch.pow(torch.tensor(decay, dtype=torch.float64), torch.arange(d))

    return ToyGaussian(mu.to(dtype), U.to(dtype), lams.to(dtype))
