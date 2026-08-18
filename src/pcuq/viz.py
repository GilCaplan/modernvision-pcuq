"""Visualization of uncertainty modes on point clouds (matplotlib, static PNGs).

Interactive (plotly) versions can come in Phase 4 if the report needs them.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _scatter(ax, pts, color, title):
    p = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=2, cmap="viridis")
    ax.set_title(title, fontsize=9)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    return p


def plot_modes(x_hat: torch.Tensor, eigvecs: torch.Tensor, eigvals: torch.Tensor,
               path: Path, mask: torch.Tensor = None) -> None:
    """One panel per mode: the denoised cloud colored by per-point displacement
    magnitude of that eigenvector. x_hat (N, 3), eigvecs (k, N, 3), eigvals (k,).
    With a mask, out-of-region points are drawn as light-gray context."""
    x_hat = x_hat.detach().cpu()
    k = eigvecs.shape[0]
    fig = plt.figure(figsize=(3.2 * k, 3.4))
    for i in range(k):
        ax = fig.add_subplot(1, k, i + 1, projection="3d")
        mag = eigvecs[i].detach().cpu().norm(dim=1)
        if mask is not None:
            mk = mask.cpu()
            ax.scatter(x_hat[~mk, 0], x_hat[~mk, 1], x_hat[~mk, 2],
                       c="0.85", s=2)
            _scatter(ax, x_hat[mk], mag[mk], f"mode {i}  λ={float(eigvals[i]):.2e}")
        else:
            _scatter(ax, x_hat, mag, f"mode {i}  λ={float(eigvals[i]):.2e}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mode_arrows(x_hat: torch.Tensor, eigvec: torch.Tensor, eigval: torch.Tensor,
                     path: Path, mask: torch.Tensor = None,
                     max_arrows: int = 120) -> None:
    """One mode as a displacement-direction arrow field (subsampled to the
    highest-magnitude points). Direction is the interpretable content of a mode;
    arrows are normalized for visibility (true scale is sqrt(λ))."""
    pts, v = x_hat.detach().cpu(), eigvec.detach().cpu()
    mag = v.norm(dim=1)
    idx = mag.topk(min(max_arrows, int((mag > 0).sum()))).indices
    scale = 0.25 / float(mag[idx].max().clamp(min=1e-30))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    ctx = pts if mask is None else pts[~mask.cpu()]
    ax.scatter(ctx[:, 0], ctx[:, 1], ctx[:, 2], c="0.85", s=2)
    if mask is not None:
        reg = pts[mask.cpu()]
        ax.scatter(reg[:, 0], reg[:, 1], reg[:, 2], c="0.55", s=2)
    ax.quiver(pts[idx, 0], pts[idx, 1], pts[idx, 2],
              v[idx, 0] * scale, v[idx, 1] * scale, v[idx, 2] * scale,
              color="crimson", linewidth=0.8, arrow_length_ratio=0.25)
    ax.set_title(f"mode direction  λ={float(eigval):.2e}", fontsize=9)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _img(t):
    """(C, H, W) tensor in [0,1] or [-1,1] -> HxW(xC) array in [0,1] for imshow."""
    a = t.detach().cpu()
    if a.min() < -0.01:
        a = (a + 1) / 2
    a = a.clamp(0, 1)
    return a[0].numpy() if a.shape[0] == 1 else a.permute(1, 2, 0).numpy()


def plot_sweep_2d(x_hat: torch.Tensor, eigvecs: torch.Tensor, eigvals: torch.Tensor,
                  path: Path, ts=(-3.0, -1.5, 0.0, 1.5, 3.0)) -> None:
    """The reference paper's signature figure, from our pipeline: one row per
    mode, columns are x_hat + t*sqrt(λ)*v. x_hat (C,H,W), eigvecs (k,C,H,W)."""
    k = eigvecs.shape[0]
    fig, axes = plt.subplots(k, len(ts), figsize=(1.6 * len(ts), 1.7 * k),
                             squeeze=False)
    for i in range(k):
        step = float(eigvals[i].clamp(min=0).sqrt())
        for j, t in enumerate(ts):
            ax = axes[i][j]
            ax.imshow(_img(x_hat + t * step * eigvecs[i]), cmap="gray",
                      vmin=0, vmax=1)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"t={t:+.1f}", fontsize=8)
        axes[i][0].set_axis_on()
        axes[i][0].set_xticks([]), axes[i][0].set_yticks([])
        axes[i][0].set_ylabel(f"mode {i}\nλ={float(eigvals[i]):.1e}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_mode_sweep(x_hat: torch.Tensor, eigvec: torch.Tensor, eigval: torch.Tensor,
                    path: Path, ts=(-3.0, 0.0, 3.0),
                    mask: torch.Tensor = None) -> None:
    """x_hat + t*sqrt(λ)*v for several t — the 3D analog of the reference repo's
    image sliders. Colors show that mode's per-point displacement magnitude."""
    x_hat, v = x_hat.detach().cpu(), eigvec.detach().cpu()
    step = float(eigval.clamp(min=0).sqrt())
    mag = v.norm(dim=1)
    fig = plt.figure(figsize=(3.2 * len(ts), 3.4))
    for i, t in enumerate(ts):
        ax = fig.add_subplot(1, len(ts), i + 1, projection="3d")
        pts = x_hat + t * step * v
        if mask is not None:
            mk = mask.cpu()
            ax.scatter(pts[~mk, 0], pts[~mk, 1], pts[~mk, 2], c="0.85", s=2)
            _scatter(ax, pts[mk], mag[mk], f"t = {t:+.1f}·√λ")
        else:
            _scatter(ax, pts, mag, f"t = {t:+.1f}·√λ")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
