# Plan / Roadmap

*(Living document — check items off as they land, add sub-items freely. Every phase ends
with a local smoke run before anything touches the GPU VM; see [WORKFLOW.md](WORKFLOW.md).)*

## Phase 0 — Scaffolding ✅

- [x] Vendor `GaussianDenoisingPosterior` into `external/` (untouched)
- [x] Docs system (this folder)
- [x] Package skeleton `src/pcuq`, configs, scripts

## Phase 1 — Core machinery on a *toy* problem (Mac-only, no pretrained model)

Goal: prove the whole pipeline end-to-end where the answer is known in closed form.

- [ ] `data.py`: synthetic Gaussian / GMM point data with known posterior covariance
- [ ] `denoisers.py`: `AnalyticGaussianDenoiser` — closed-form MMSE denoiser for a
      Gaussian prior (posterior covariance known exactly → ground truth for eigenpairs)
- [ ] `jacobian.py`: forward-diff, central-diff, and autograd (`torch.func.jvp`) JVPs
      behind one interface; symmetrized product `v ↦ ½(J+Jᵀ)v`
- [ ] `spectrum.py`: block power iteration (port the idea from
      `external/.../moments_calculations.py:get_eigvecs`, rewritten clean) + optional Lanczos
- [ ] `diagnostics.py`: antisymmetric energy, PSD check, step-size (`c`) sweep, JVP
      agreement (finite-diff vs autograd)
- [ ] `tests/`: eigenpairs from power iteration match analytic covariance to tolerance
- [ ] **Gate:** `scripts/sanity_gaussian.py --config configs/local.yaml` passes on the Mac

## Phase 2 — Real denoiser integration

- [ ] Obtain Noise2Score3D code + checkpoint (availability check — see
      [SOURCES.md](SOURCES.md); fallback: ScoreDenoise). Vendor into `external/`
- [ ] `denoisers.py`: wrapper conforming to our `Denoiser` interface
      (`(B, N, 3) → (B, N, 3)`, frozen, `eval()`, no grad unless asked)
- [ ] Ordering-preservation / permutation-equivariance test on the real denoiser
      (`diagnostics.py`) — **hard gate**: if ordering breaks, we need matching or a
      different denoiser
- [ ] `data.py`: ModelNet40 download/loading, mesh → N-point sampling, normalization,
      seeded corruption with retained indices
- [ ] **Gate:** tiny end-to-end run on the Mac (few shapes, N small, 1–2 eigenpairs)

## Phase 3 — Full experiments (GPU VM)

- [ ] Step-size sweep + finite-diff vs autograd validation at full N (`configs/gpu.yaml`)
- [ ] Top 3–5 eigenpairs across shape categories and noise levels σ
- [ ] Antisymmetry / PSD monitoring across all runs (report numbers, not vibes)
- [ ] Convergence: power-iteration correlation curves (as in the reference repo)

## Phase 4 — Analysis, visualization, report

- [ ] `viz.py`: eigenmode displacement fields on point clouds (± amounts along mode,
      like the reference repo's image sliders, but as 3D arrows / animated offsets)
- [ ] Quantitative tables: eigenvalue spectra vs σ; validation-gate results
- [ ] Failure cases + discussion (where the linearization breaks)
- [ ] Write-up / figures

## Open questions (move to LOG.md when resolved)

- Noise2Score3D public code/checkpoint availability — verify early (Phase 2 blocker).
- Eigenvectors of the *full* 3N×3N Jacobian vs restricting to a region mask (the
  reference repo uses patch masks; our analog = subset of points).
- Lanczos worth it over subspace iteration for k ≤ 5? (Probably not — decide by Phase 3.)
