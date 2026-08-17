# Plan / Roadmap

*(Living document — check items off as they land, add sub-items freely. Every phase ends
with a local smoke run before anything touches the GPU VM; see [WORKFLOW.md](WORKFLOW.md).)*

## Phase 0 — Scaffolding ✅

- [x] Vendor `GaussianDenoisingPosterior` into `external/` (untouched)
- [x] Docs system (this folder)
- [x] Package skeleton `src/pcuq`, configs, scripts

## Phase 1 — Core machinery on a *toy* problem (Mac-only, no pretrained model) ✅

Goal: prove the whole pipeline end-to-end where the answer is known in closed form.

- [x] `data.py`: synthetic Gaussian point data with known posterior covariance
      (`ToyGaussian`; GMM variant only if we later want non-Gaussian posteriors)
- [x] `denoisers.py`: `AnalyticGaussianDenoiser` — closed-form MMSE denoiser for a
      Gaussian prior (posterior covariance known exactly → ground truth for eigenpairs)
- [x] `jacobian.py`: forward-diff, central-diff, and autograd (`torch.func.jvp`) JVPs
      behind one interface; symmetrized product `v ↦ ½(J+Jᵀ)v`
- [x] `spectrum.py`: block power (subspace) iteration with QR + Rayleigh-quotient
      eigenvalues (Lanczos deferred to the Phase-3 decision below)
- [x] `diagnostics.py`: antisymmetric energy, PSD report, step-size (`c`) sweep vs
      autograd, permutation-equivariance check
- [x] `tests/`: eigenpairs from subspace iteration match analytic covariance (float64,
      rtol 1e-3); JVP methods agree; symmetrized == plain for symmetric J
- [x] **Gate:** `scripts/sanity_gaussian.py --config configs/local.yaml` passes on the
      Mac (2026-08-17: PASS, rel err ≤0.3%, |cos| ≥0.995 — see LOG.md)

## Phase 2 — Real denoiser integration

- [x] Obtain Noise2Score3D code — verified & vendored at `external/Noise2Score3D/`
      (ICCV 2025, pretrained weights on HF — see [SOURCES.md](SOURCES.md))
- [ ] Download pretrained checkpoint from HF; get their model running on the GPU VM
      (deps: PyTorch3D, pykeops — probably too heavy for the Mac; smoke path can keep
      using the analytic denoiser)
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

- Eigenvectors of the *full* 3N×3N Jacobian vs restricting to a region mask (the
  reference repo uses patch masks; our analog = subset of points).
- Lanczos worth it over subspace iteration for k ≤ 5? (Probably not — decide by Phase 3.)
