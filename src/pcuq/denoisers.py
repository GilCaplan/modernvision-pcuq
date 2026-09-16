"""Frozen denoisers behind one tiny interface.

Contract: denoise(y) maps (B, N, 3) -> (B, N, 3), model frozen, point ordering
preserved (verified by diagnostics.check_equivariance before trusting a real model).
"""

import contextlib
import hashlib
import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

from .data import ToyGaussian, ToyGMM

# What sigma^2 * J(D) at a denoiser D is entitled to mean. sigma^2 * J is the exact
# posterior covariance only when D is verifiably the MMSE denoiser for a known prior
# at exactly that sigma; every subclass must pick one so a saved spectrum can never
# silently be read as "the" posterior covariance when it is really a local
# sensitivity operator (cf. docs/LOG.md 2026-09-16).
EXACT_MMSE_FIXED_SIGMA = "exact_mmse_fixed_sigma"          # closed-form, ground truth
APPROXIMATE_MMSE_FIXED_SIGMA = "approximate_mmse_fixed_sigma"  # trained at/near a
                                                                # fixed, known sigma
FROZEN_PYRAMID_SENSITIVITY = "frozen_pyramid_sensitivity"  # no verified fixed-sigma
                                                             # MMSE target at this sigma
COVARIANCE_KINDS = frozenset({
    EXACT_MMSE_FIXED_SIGMA, APPROXIMATE_MMSE_FIXED_SIGMA, FROZEN_PYRAMID_SENSITIVITY,
})


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA256 of a (possibly large, e.g. 293MB) checkpoint file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


class Denoiser:
    """covariance_kind (set by every subclass, see the constants above) classifies
    what sigma^2 * J at this denoiser means — required by spectrum.top_eigenpairs."""

    covariance_kind: str = None

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

    covariance_kind = EXACT_MMSE_FIXED_SIGMA

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


class AnalyticGMMDenoiser(Denoiser):
    """Exact MMSE denoiser for a Gaussian-mixture prior (ToyGMM):

        D(y) = E[X|Y=y] = sum_k r_k(y) * m_k(y)

    m_k(y) the per-component linear-Gaussian posterior mean, r_k(y) the Bayes
    responsibility (softmax over per-component log-marginals). Non-Gaussian and
    multimodal, but still exactly the posterior mean by construction, so
    ToyGMM.posterior_eigenpairs remains exact ground truth for the numerically
    estimated spectrum — unlike AnalyticGaussianDenoiser, Cov[X|Y] here is not
    constant in y and its eigenvalues are not bounded by sigma^2 (see ToyGMM).
    Fully vectorized (einsum, softmax) so torch.func.jvp can trace through it.
    """

    covariance_kind = EXACT_MMSE_FIXED_SIGMA

    def __init__(self, gmm: ToyGMM, sigma: float):
        self.mu_flat = gmm.mu.reshape(gmm.n_components, -1)  # (K, d)
        self.U = gmm.U                                        # (K, d, d)
        self.var = gmm.lams + sigma**2                        # (K, d)
        self.gains = gmm.lams / self.var                      # (K, d)
        self.log_norm_const = gmm.log_pi - 0.5 * torch.log(2 * math.pi * self.var).sum(-1)
        self.out_shape = gmm.mu.shape[1:]                     # (N, 3)
        self.sigma = sigma

    def to(self, device: torch.device) -> "AnalyticGMMDenoiser":
        for name in ("mu_flat", "U", "var", "gains", "log_norm_const"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def denoise(self, y: torch.Tensor) -> torch.Tensor:
        yf = y.reshape(y.shape[0], -1)                                     # (B, d)
        resid = yf[:, None, :] - self.mu_flat[None, :, :]                  # (B, K, d)
        z = torch.einsum("bkd,kde->bke", resid, self.U)                   # U_k coords
        log_marg = self.log_norm_const[None, :] \
            - 0.5 * (z**2 / self.var[None, :, :]).sum(-1)                 # (B, K)
        r = log_marg.softmax(dim=-1)
        m = self.mu_flat[None, :, :] \
            + torch.einsum("kde,bke->bkd", self.U, self.gains[None, :, :] * z)
        return torch.einsum("bk,bkd->bd", r, m).reshape(y.shape[0], *self.out_shape)


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


def _voxel_group_ids(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Reimplementation of grid_subsample_pack_mode's voxel hashing (single-shape,
    no batch padding — this wrapper always calls the model with batch_size=1) that
    returns the group assignment itself instead of discarding it after averaging.
    Needed so graph_topology_frozen below can hold that assignment FIXED across
    perturbations while still recomputing coarse points as new voxel means.
    Origin: external/Noise2Score3D/models/easy_kpconv/ops/grid_subsample.py
    (ravel_hash_func / grid_subsample_pack_mode)."""
    voxels = torch.floor(points / voxel_size).long()
    voxels = voxels - voxels.amin(0, keepdim=True)
    dims = voxels.amax(0) + 1
    hashed = (voxels[:, 0] * dims[1] + voxels[:, 1]) * dims[2] + voxels[:, 2]
    _, group_ids = torch.unique(hashed, return_inverse=True)
    return group_ids  # (N,), contiguous 0..M-1, same partition/order torch.unique gives them


def _voxel_mean(points: torch.Tensor, group_ids: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Scatter-mean of points into n_groups fixed groups: the coarsening step
    grid_subsample_pack_mode performs, but against a caller-supplied (frozen)
    assignment instead of recomputing one from these (possibly perturbed) points."""
    idx = group_ids[:, None].expand(-1, 3)
    sums = points.new_zeros(n_groups, 3).scatter_add_(0, idx, points)
    counts = points.new_zeros(n_groups).scatter_add_(
        0, group_ids, torch.ones_like(group_ids, dtype=points.dtype))
    return sums / counts[:, None]


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


def _shim_open3d_if_missing() -> bool:
    """Their kernel_points.py imports open3d only to cache/read tiny KPConv kernel-
    point dispositions as binary-little-endian PLY files (<=30 points, x/y/z
    float64, no faces/colors) — every disposition Noise2Score3D ships is already
    cached under models/easy_kpconv/layers/kpconv_utils/dispositions/, so only the
    read path is ever exercised at inference. A full open3d install pulls a heavy
    platform-specific binary dependency for that; this shim implements just the
    three calls kernel_points.py makes, for exactly that one narrow PLY format."""
    if "open3d" in sys.modules:
        return getattr(sys.modules["open3d"], "_pcuq_shim", False)
    if importlib.util.find_spec("open3d") is not None:
        return False

    def _read_point_cloud(path):
        import numpy as np
        with open(path, "rb") as f:
            header = []
            while header[-1:] != ["end_header"]:
                header.append(f.readline().decode("ascii").strip())
            assert any("float64" in h for h in header), "shim only handles float64 PLY"
            n = next(int(h.split()[-1]) for h in header if h.startswith("element vertex"))
            pts = np.frombuffer(f.read(n * 3 * 8), dtype="<f8").reshape(n, 3)
        return types.SimpleNamespace(points=pts)

    def _write_point_cloud(path, pcd):
        import numpy as np
        pts = np.asarray(pcd.points, dtype="<f8")
        header = ("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(pts)}\n"
                  "property float64 x\nproperty float64 y\nproperty float64 z\n"
                  "end_header\n").encode("ascii")
        with open(path, "wb") as f:
            f.write(header)
            f.write(pts.tobytes())
        return True

    o3d = types.ModuleType("open3d")
    o3d._pcuq_shim = True
    o3d.geometry = types.SimpleNamespace(PointCloud=lambda: types.SimpleNamespace(points=None))
    o3d.utility = types.SimpleNamespace(Vector3dVector=lambda arr: arr)
    o3d.io = types.SimpleNamespace(read_point_cloud=_read_point_cloud,
                                   write_point_cloud=_write_point_cloud)
    sys.modules["open3d"] = o3d
    return True


class Noise2Score3DWrapper(Denoiser):
    """Frozen Noise2Score3D (Wei et al., ICCV 2025) from external/Noise2Score3D.

    The network predicts the score s(y) of the noisy distribution; Tweedie gives the
    posterior mean D(y) = y + sigma^2 * s(y) (their test.py inference, in the
    normalized coordinate frame the model was trained in — pass clouds normalized to
    roughly the unit sphere, and sigma in that same frame).

    covariance_kind = frozen_pyramid_sensitivity, not an MMSE label: the network is an
    unconditioned score estimate with sigma manually substituted into Tweedie's
    formula, and its Jacobian is only well-behaved once differentiated through a
    frozen discrete graph pyramid (see graph_frozen below) — sigma^2 * J here is a
    local sensitivity operator around the anchor, not a calibrated posterior
    covariance for a verified fixed-sigma MMSE target.
    """

    covariance_kind = FROZEN_PYRAMID_SENSITIVITY

    def __init__(self, repo_dir: str, checkpoint: str, sigma: float,
                 device: torch.device):
        repo = Path(repo_dir).resolve()
        for p in (str(repo), str(repo / "models")):  # they import both
            if p not in sys.path:                    # `models.easy_kpconv` and
                sys.path.insert(0, p)                # `easy_kpconv` styles
        self._device = device
        with _cuda_calls_redirected(device):
            shimmed = _shim_pykeops_if_missing()
            _shim_open3d_if_missing()
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
        self._voxel_groups = None
        # Which exact weights produced a run -- a re-downloaded or swapped
        # checkpoint at the same path would otherwise be indistinguishable from
        # a report's perspective (docs/PLAN.md reproducibility follow-up).
        self.checkpoint_sha256 = _sha256_file(checkpoint)

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

    @contextlib.contextmanager
    def graph_topology_frozen(self):
        """A weaker freeze than graph_frozen: discrete topology (voxel-group
        membership at every pyramid stage, plus all radius-search neighbor/
        subsampling/upsampling edges) is fixed at the anchor, but coarse-level
        POINT COORDINATES are recomputed as voxel means of the perturbed level-0
        input through that fixed membership, instead of being frozen outright too.
        Tests whether graph_frozen's simpler "coarse points fixed as well"
        approximation throws away real signal (see docs/LOG.md 2026-09-16
        graph-freezing audit for the comparison and which variant won).

        Feasible only because grid_subsample_pack_mode's coarsening is exactly a
        voxel-grid mean (_voxel_group_ids/_voxel_mean above reimplement it) — there
        is no analogous continuous relaxation for the discrete radius-search graph
        itself, so neighbor/subsampling/upsampling edges stay frozen exactly as in
        graph_frozen; only the coarse `points` entries differ from that method.
        """
        orig = self._n2s3d.build_grid_and_radius_graph_pyramid
        voxel_size = self.model.voxel_size

        def cached_build(points, lengths, *args, **kwargs):
            if self._graph_cache is None:
                self._graph_cache = orig(points, lengths, *args, **kwargs)
                anchor_points = self._graph_cache["points"]
                self._voxel_groups = [
                    (_voxel_group_ids(anchor_points[i], voxel_size * 2 ** (i + 1)),
                     int(anchor_points[i + 1].shape[0]))
                    for i in range(len(anchor_points) - 1)
                ]
                return self._graph_cache
            new_points = [points]
            for gid, n_groups in self._voxel_groups:
                new_points.append(_voxel_mean(new_points[-1], gid, n_groups))
            g = dict(self._graph_cache)
            g["points"] = new_points
            return g

        self._n2s3d.build_grid_and_radius_graph_pyramid = cached_build
        try:
            yield
        finally:
            self._n2s3d.build_grid_and_radius_graph_pyramid = orig
            self._graph_cache = None
            self._voxel_groups = None

    def denoise(self, y: torch.Tensor) -> torch.Tensor:
        outs = []
        for cloud in y:  # their pack-mode path handles one cloud at a time
            with _cuda_calls_redirected(self._device):
                score, _, _ = self.model(cloud[None], None)  # (N, 3)
            outs.append(cloud + self.sigma**2 * score)
        return torch.stack(outs)
