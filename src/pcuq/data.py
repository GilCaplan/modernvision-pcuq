"""Datasets and corruption.

Planned API (Phase 1 fills the toy parts, Phase 2 the ModelNet parts):

- make_toy_gaussian(cfg) -> (x, prior_mean, prior_cov)
    Synthetic point data drawn from a known Gaussian prior, so the posterior
    covariance (and hence the true eigenpairs) is available in closed form.
- load_modelnet(cfg) -> iterator of (points (N, 3), category)
    Download/cache ModelNet40 under cfg['data']['root'], sample cfg n_points from
    each mesh (trimesh), normalize to unit sphere. Exact variant recorded in
    docs/SOURCES.md.
- corrupt(x, sigma, seed) -> y
    Y = X + sigma * Z with Z ~ N(0, I), same shape, indices retained: point i of y
    corresponds to point i of x. This correspondence is what jacobian.py relies on.
"""

import torch


def corrupt(x: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Gaussian corruption Y = X + sigma*Z, retaining exact point indices."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(x.shape, generator=gen, dtype=x.dtype).to(x.device)
    return x + sigma * z
