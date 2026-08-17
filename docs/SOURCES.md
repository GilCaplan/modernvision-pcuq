# Sources

*(Every paper, repo, dataset, and checkpoint the project touches. Add as you go; note
license and what we use it for.)*

## Core method

- **Manor & Michaeli, "On the Posterior Distribution in Denoising: Application to
  Uncertainty Quantification", ICLR 2024** — the method we adapt.
  [arXiv:2309.13598](https://arxiv.org/abs/2309.13598) ·
  [project page](https://HilaManor.github.io/GaussianDenoisingPosterior) ·
  [code](https://github.com/HilaManor/GaussianDenoisingPosterior)
  (vendored at `external/GaussianDenoisingPosterior/`, shallow clone 2026-08-17).
  Key file: `moments_calculations.py` (subspace iteration on the denoiser Jacobian).

  ```bibtex
  @inproceedings{manor2024posterior,
    title={On the Posterior Distribution in Denoising: Application to Uncertainty Quantification},
    author={Hila Manor and Tomer Michaeli},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
  }
  ```

## Denoiser (frozen model under study)

- **Noise2Score3D** — Tweedie-formula single-step point-cloud denoiser; our primary
  target model. ⚠️ **TODO (Phase 2 blocker): verify public code + checkpoint
  availability**; vendor into `external/` when obtained, record exact commit/checkpoint
  hash here.
- **Fallback: ScoreDenoise** (Luo & Hu, "Score-Based Point Cloud Denoising", ICCV 2021)
  — score-based, public pretrained checkpoints:
  [github.com/luost26/score-denoise](https://github.com/luost26/score-denoise).
  Score-based → Tweedie-compatible; a solid plan B if Noise2Score3D isn't usable.

## Background / theory

- Efron, "Tweedie's formula and selection bias", JASA 2011 — Tweedie's formula.
- Miyasawa 1961 / Robbins 1956 — classical empirical-Bayes posterior-mean identity.
- Golub & Van Loan, *Matrix Computations* — subspace (block power) iteration, Lanczos.

## Dataset

- **ModelNet40** (Wu et al., CVPR 2015) — CAD meshes, 40 categories.
  [modelnet.cs.princeton.edu](https://modelnet.cs.princeton.edu/). We sample point
  clouds from meshes ourselves (`pcuq/data.py`) to control N and keep indices; record
  the exact download variant here once fixed (raw meshes vs `modelnet40_normal_resampled`).

## Tooling

- PyTorch (`torch.func.jvp` for exact JVPs), trimesh (mesh sampling), plotly/matplotlib
  (3D viz).
