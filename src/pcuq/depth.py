"""Point cloud -> depth-map image, for the depth-map 2D benchmark.

Side quest (see scripts/run_depth2d.py): render each ModelNet40 shape into a
single depth image and run it through the reference paper's OWN 2D denoisers
(external/GaussianDenoisingPosterior, via denoisers2d.py) instead of the 3D
Noise2Score3D denoiser — a cross-domain comparison, not a rewrite of vendored
code (point-cloud rendering has no analog in either external/ repo).
"""

import torch
import torch.nn.functional as F


def render_depth_map(points: torch.Tensor, resolution: int, axis: int = 2,
                     dilate: int = 0) -> torch.Tensor:
    """Single-view orthographic depth map of a point cloud already normalized
    to the unit sphere (see data.normalize_unit_sphere / load_modelnet).

    Projects along `axis` (0=x, 1=y, 2=z); the other two coordinates become
    the image plane. Each pixel takes the depth of its NEAREST point along the
    view axis (a z-buffer, via scatter_reduce's "amax"); pixels no point lands
    in stay 0. Depth is normalized to [0, 1] off the fixed unit-sphere range
    [-1, 1] (not a per-shape min/max), so values stay comparable across
    shapes. -> (resolution, resolution) float tensor in [0, 1].

    Point-cloud depth maps are inherently sparse (gaps between samples, unlike
    a rendered mesh); `dilate` rounds of 3x3 max-pool hole-filling close small
    gaps before handing the image to a CNN/DDPM trained on dense photos.
    """
    others = [a for a in range(3) if a != axis]
    u, v, depth = points[:, others[0]], points[:, others[1]], points[:, axis]

    def to_px(c):
        return ((c.clamp(-1, 1) + 1) / 2 * (resolution - 1)).round().long()

    px, py = to_px(u), resolution - 1 - to_px(v)  # row 0 = top = larger v
    flat_idx = py * resolution + px
    dnorm = (depth.clamp(-1, 1) + 1) / 2  # in [0, 1], matching the 0 background

    img = points.new_zeros(resolution * resolution)
    img.scatter_reduce_(0, flat_idx, dnorm, reduce="amax", include_self=True)
    img = img.reshape(resolution, resolution)

    for _ in range(dilate):
        filled = F.max_pool2d(img[None, None], 3, stride=1, padding=1)[0, 0]
        img = torch.where(img <= 0, filled, img)
    return img
