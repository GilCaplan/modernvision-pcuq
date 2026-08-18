"""Frozen denoisers behind one tiny interface.

Contract: denoise(y) maps (B, N, 3) -> (B, N, 3), model frozen, point ordering
preserved (verified by diagnostics.check_equivariance before trusting a real model).
"""

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

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


@contextlib.contextmanager
def _cuda_calls_redirected(device: torch.device):
    """The vendored Noise2Score3D code hardcodes .cuda() in layer constructors and
    runtime helpers. Inside this context, Tensor.cuda() means "move to `device`"
    when CUDA is unavailable (Mac: cpu or mps), and is untouched when CUDA exists
    (GPU VM) — the vendored files stay unedited."""
    if torch.cuda.is_available():
        yield
        return
    orig = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self.to(device)
    try:
        yield
    finally:
        torch.Tensor.cuda = orig


def _torch_knn(q_points, s_points, k):
    """Exact pure-torch replacement for their pykeops keops_knn: k smallest
    euclidean distances, ascending — identical semantics at our scales (N <= ~2k),
    used only when pykeops isn't installed (Mac). (*, N, C), (*, M, C) -> (*, N, k).

    When a coarse pyramid level has fewer support points than k, pad with
    inf-distance / index-0 entries — their radius mask discards anything beyond
    the search radius, so padding reads as "no neighbor"."""
    m = s_points.shape[-2]
    dists, idx = torch.cdist(q_points, s_points).topk(min(k, m), dim=-1, largest=False)
    if m < k:
        pad = (*dists.shape[:-1], k - m)
        dists = torch.cat([dists, dists.new_full(pad, torch.inf)], dim=-1)
        idx = torch.cat([idx, idx.new_zeros(pad)], dim=-1)
    return dists, idx


def _shim_pykeops_if_missing() -> bool:
    """Their knn.py imports pykeops at module level. If pykeops isn't installed,
    register an empty stand-in so the import succeeds; the caller must then swap
    keops_knn for _torch_knn. Returns True if the shim is in place."""
    if "pykeops" in sys.modules:  # find_spec would choke on our spec-less stub
        return getattr(sys.modules["pykeops"], "_pcuq_shim", False)
    if importlib.util.find_spec("pykeops") is not None:
        return False
    pk = types.ModuleType("pykeops")
    pk._pcuq_shim = True
    pk_torch = types.ModuleType("pykeops.torch")
    pk_torch.LazyTensor = None  # imported by name, unused once keops_knn is swapped
    pk.torch = pk_torch
    sys.modules["pykeops"] = pk
    sys.modules["pykeops.torch"] = pk_torch
    return True


class Noise2Score3DWrapper(Denoiser):
    """Frozen Noise2Score3D (Wei et al., ICCV 2025) from external/Noise2Score3D.

    The network predicts the score s(y) of the noisy distribution; Tweedie gives the
    posterior mean D(y) = y + sigma^2 * s(y) (their test.py inference, in the
    normalized coordinate frame the model was trained in — pass clouds normalized to
    roughly the unit sphere, and sigma in that same frame).
    """

    def __init__(self, repo_dir: str, checkpoint: str, sigma: float,
                 device: torch.device):
        repo = Path(repo_dir).resolve()
        for p in (str(repo), str(repo / "models")):  # they import both
            if p not in sys.path:                    # `models.easy_kpconv` and
                sys.path.insert(0, p)                # `easy_kpconv` styles
        self._device = device
        with _cuda_calls_redirected(device):
            shimmed = _shim_pykeops_if_missing()
            from models import KPconv_test as n2s3d
            if shimmed:  # radius_search imports keops_knn by name — patch both
                sys.modules["models.easy_kpconv.ops.knn"].keops_knn = _torch_knn
                sys.modules["models.easy_kpconv.ops.radius_search"].keops_knn = _torch_knn
            # Their dataloader() hardcodes .cuda() on a fresh tensor; rebuild the
            # same dict with lengths on the input's device instead.
            n2s3d.dataloader = lambda data: {
                "points": data[0],
                "lengths": torch.tensor([data.shape[1]], dtype=torch.int64,
                                        device=data.device),
                "batch_size": 1,
            }
            model = n2s3d.get_model(n2s3d.Config(), normal_channel=None)
            state = torch.load(checkpoint, map_location="cpu")
            # The HF checkpoint was trained with bias-free KPConv layers; this build
            # adds conv biases, zero-initialized (= identical function). Allow only
            # those keys to be missing, nothing else.
            result = model.load_state_dict(state["model_state_dict"], strict=False)
            assert not result.unexpected_keys, result.unexpected_keys
            assert all(k.endswith("conv.bias") for k in result.missing_keys), \
                result.missing_keys
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.sigma = sigma
        self._n2s3d = n2s3d
        self._graph_cache = None

    @contextlib.contextmanager
    def graph_frozen(self):
        """Freeze the graph pyramid at the anchor for all forwards in this context.

        Their forward rebuilds the voxel/radius graph every call; a perturbation
        y + c*v flips discrete assignments, so finite differences pick up O(1)
        jumps instead of the smooth local derivative (measured: no step-size
        plateau, ~40-80%% self-consistency error). Inside this context the FIRST
        forward builds and caches the pyramid — call denoise(anchor) first — and
        later forwards reuse its discrete indices and coarse-level positions,
        with only the level-0 points varying: the smooth branch of the model.
        """
        orig = self._n2s3d.build_grid_and_radius_graph_pyramid

        def cached_build(points, lengths, *args, **kwargs):
            if self._graph_cache is None:
                self._graph_cache = orig(points, lengths, *args, **kwargs)
                return self._graph_cache
            g = dict(self._graph_cache)
            g["points"] = [points] + list(self._graph_cache["points"][1:])
            return g

        self._n2s3d.build_grid_and_radius_graph_pyramid = cached_build
        try:
            yield
        finally:
            self._n2s3d.build_grid_and_radius_graph_pyramid = orig
            self._graph_cache = None

    def denoise(self, y: torch.Tensor) -> torch.Tensor:
        outs = []
        for cloud in y:  # their pack-mode path handles one cloud at a time
            with _cuda_calls_redirected(self._device):
                score, _, _ = self.model(cloud[None], None)  # (N, 3)
            outs.append(cloud + self.sigma**2 * score)
        return torch.stack(outs)
