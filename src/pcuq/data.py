"""Datasets and corruption.

- Toy: synthetic Gaussian prior with closed-form posterior (ground truth for tests).
- Real: ModelNet40 — official zip cached under data/, OFF meshes parsed and sampled
  in pure torch (no trimesh; ModelNet OFF files are often malformed, see _parse_off),
  normalized to the unit sphere, seeded sampling with retained indices.
"""

import math
import zipfile
from pathlib import Path

import torch

MODELNET40_URL = "https://modelnet.cs.princeton.edu/ModelNet40.zip"


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


def _parse_off(text: str):
    """Parse an OFF mesh -> (verts (V, 3) float64, faces (F, 3) long).

    Tolerates ModelNet40's malformed headers ("OFF490 240 0" glued on one line)
    and fan-triangulates faces with more than 3 vertices.
    """
    lines = [ln for ln in (raw.split("#")[0].strip() for raw in text.splitlines()) if ln]
    if lines[0] == "OFF":
        counts, body = lines[1], lines[2:]
    elif lines[0].startswith("OFF"):
        counts, body = lines[0][3:], lines[1:]
    else:
        raise ValueError("not an OFF file")
    n_verts, n_faces = (int(t) for t in counts.split()[:2])

    verts = torch.tensor([[float(t) for t in ln.split()[:3]] for ln in body[:n_verts]],
                         dtype=torch.float64)
    faces = []
    for ln in body[n_verts:n_verts + n_faces]:
        idx = [int(t) for t in ln.split()]
        for j in range(1, idx[0] - 1):  # idx[0] = vertex count of this face
            faces.append([idx[1], idx[1 + j], idx[2 + j]])
    return verts, torch.tensor(faces, dtype=torch.long)


def sample_mesh(verts: torch.Tensor, faces: torch.Tensor, n_points: int,
                seed: int) -> torch.Tensor:
    """Area-weighted uniform surface sampling -> (n_points, 3), deterministic."""
    a, b, c = (verts[faces[:, i]] for i in range(3))
    areas = torch.linalg.cross(b - a, c - a).norm(dim=1)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pick = torch.multinomial(areas.clamp(min=1e-12), n_points, replacement=True,
                             generator=gen)
    uv = torch.rand(n_points, 2, generator=gen, dtype=verts.dtype)
    flip = uv.sum(dim=1) > 1  # reflect into the triangle
    uv[flip] = 1 - uv[flip]
    return (a[pick] + uv[:, :1] * (b - a)[pick] + uv[:, 1:] * (c - a)[pick])


def normalize_unit_sphere(points: torch.Tensor):
    """Center on the centroid, scale the farthest point to radius 1 — the frame
    Noise2Score3D was trained in. Returns (points, center, scale)."""
    center = points.mean(dim=0, keepdim=True)
    points = points - center
    scale = points.norm(dim=1).max()
    return points / scale, center, scale


def load_modelnet(cfg: dict, dtype: torch.dtype = torch.float32):
    """ModelNet40 test-split shapes -> list of (name, points (n_points, 3)).

    Downloads the official zip (~2GB, one-time) into data.root and reads OFF files
    straight out of it (no extraction). Takes data.n_shapes shapes round-robin
    across data.categories; sampling is seeded per shape.
    """
    d = cfg["data"]
    root = Path(d["root"])
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "ModelNet40.zip"
    if not zip_path.exists():
        print(f"Downloading ModelNet40 (~2GB, one-time) to {zip_path} ...")
        torch.hub.download_url_to_file(MODELNET40_URL, str(zip_path))

    zf = zipfile.ZipFile(zip_path)
    per_cat = {cat: sorted(n for n in zf.namelist()
                           if n.startswith(f"ModelNet40/{cat}/test/") and n.endswith(".off"))
               for cat in d["categories"]}
    for cat, names in per_cat.items():
        if not names:
            raise ValueError(f"no test shapes found for category '{cat}'")
    total = sum(len(v) for v in per_cat.values())
    if d["n_shapes"] > total:
        raise ValueError(f"n_shapes={d['n_shapes']} > {total} available test shapes")

    shapes = []
    rank = 0
    while len(shapes) < d["n_shapes"]:
        for cat in d["categories"]:
            if len(shapes) >= d["n_shapes"]:
                break
            names = per_cat[cat]
            if rank >= len(names):
                continue
            member = names[rank]
            verts, faces = _parse_off(zf.read(member).decode())
            pts = sample_mesh(verts, faces, d["n_points"], seed=cfg["seed"] + len(shapes))
            pts, _, _ = normalize_unit_sphere(pts)
            shapes.append((Path(member).stem, pts.to(dtype)))
        rank += 1
    return shapes


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
