"""2D image denoisers from the reference repo (Manor & Michaeli, ICLR 2024),
behind our Denoiser interface — the original method's domain, run through OUR
jacobian/spectrum machinery for a like-for-like comparison with the 3D results.

Tensors are (B, C, H, W) in [0, 1] (MNIST) / [-1, 1] (FFHQ DDPM); pcuq's
jacobian/spectrum code is shape-agnostic, so nothing else changes.

NOTE: do not import this module and the Noise2Score3D wrapper in one process —
both vendored repos define top-level packages with generic names (`models`,
`model`) that would collide on sys.path.
"""

import sys
import types
from pathlib import Path

import torch

from .denoisers import Denoiser


def _add_paths(repo_dir: str, *subdirs: str) -> Path:
    repo = Path(repo_dir).resolve()
    for p in (str(repo), *(str(repo / s) for s in subdirs)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo


class MNISTDenoiser2D(Denoiser):
    """Their bundled MNIST CNN denoiser (checkpoint ships with the repo).

    Trained at a single noise level, sigma = 140.25/255 — unlike the amortized
    3D model there is no sigma knob here.
    """

    SIGMA = 140.25 / 255

    def __init__(self, repo_dir: str, device: torch.device):
        repo = _add_paths(repo_dir, "MNIST")
        from models_wrappers.mnist_wrapper import MNISTWrapper
        self.wrapper = MNISTWrapper(model_path=str(repo / "MNIST/MNIST_n140.25.pth"),
                                    device=device, double_precision=False)
        self.sigma = self.SIGMA

    def denoise(self, y: torch.Tensor) -> torch.Tensor:
        # no torch.no_grad here: reverse-mode products need the graph; params
        # are frozen and finite-diff callers wrap no_grad themselves.
        return self.wrapper(y)


class FFHQDenoiser2D(Denoiser):
    """Their DDPM FFHQ denoiser (guided_diffusion UNet, one p_sample step from
    timestep `from_t`; sigma is implied by the timestep). Needs ffhq.pt (2.2GB,
    see external/README.md). Images (B, 3, 256, 256) in [-1, 1].
    """

    def __init__(self, repo_dir: str, device: torch.device, from_t: int = 100):
        _add_paths(repo_dir, "DDPM_FFHQ")
        # dist_util imports mpi4py at module level; it's only used for
        # multi-node runs — stub it when absent so single-process works.
        if "mpi4py" not in sys.modules:
            try:
                import mpi4py  # noqa: F401
            except ImportError:
                fake = types.ModuleType("mpi4py")

                class _Comm:
                    def Get_rank(self):
                        return 0

                    def Get_size(self):
                        return 1

                    def bcast(self, obj, root=0):
                        return obj

                fake.MPI = types.SimpleNamespace(COMM_WORLD=_Comm())
                sys.modules["mpi4py"] = fake
        from models_wrappers.ddpm_wrapper import DiffWrapper
        # Their UNet routes every block through CheckpointFunction, whose backward
        # differentiates w.r.t. the (frozen) params and crashes. We never need
        # activation checkpointing at inference scale — bypass it so reverse-mode
        # products w.r.t. the INPUT work. (unet.py imports `checkpoint` by name.)
        from guided_diffusion import nn as gd_nn, unet as gd_unet
        gd_nn.checkpoint = gd_unet.checkpoint = \
            lambda func, inputs, params, flag: func(*inputs)
        repo = Path(repo_dir).resolve()
        self.wrapper = DiffWrapper(from_t=from_t, device=device,
                                   double_precision=False,
                                   model_path=str(repo / "DDPM_FFHQ/ffhq.pt"),
                                   open_ai_logdir=str(repo / "DDPM_FFHQ/logdir"))
        self.sigma = float(self.wrapper.get_noise())

    def denoise(self, y: torch.Tensor) -> torch.Tensor:
        return self.wrapper(y)
