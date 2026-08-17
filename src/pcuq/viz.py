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
               path: Path) -> None:
    """One panel per mode: the denoised cloud colored by per-point displacement
    magnitude of that eigenvector. x_hat (N, 3), eigvecs (k, N, 3), eigvals (k,)."""
    x_hat = x_hat.detach().cpu()
    k = eigvecs.shape[0]
    fig = plt.figure(figsize=(3.2 * k, 3.4))
    for i in range(k):
        ax = fig.add_subplot(1, k, i + 1, projection="3d")
        mag = eigvecs[i].detach().cpu().norm(dim=1)
        _scatter(ax, x_hat, mag, f"mode {i}  λ={float(eigvals[i]):.2e}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mode_sweep(x_hat: torch.Tensor, eigvec: torch.Tensor, eigval: torch.Tensor,
                    path: Path, ts=(-3.0, 0.0, 3.0)) -> None:
    """x_hat + t*sqrt(λ)*v for several t — the 3D analog of the reference repo's
    image sliders. Colors show that mode's per-point displacement magnitude."""
    x_hat, v = x_hat.detach().cpu(), eigvec.detach().cpu()
    step = float(eigval.clamp(min=0).sqrt())
    mag = v.norm(dim=1)
    fig = plt.figure(figsize=(3.2 * len(ts), 3.4))
    for i, t in enumerate(ts):
        ax = fig.add_subplot(1, len(ts), i + 1, projection="3d")
        _scatter(ax, x_hat + t * step * v, mag, f"t = {t:+.1f}·√λ")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
