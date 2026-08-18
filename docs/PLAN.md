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

## Phase 2 — Real denoiser integration ✅

- [x] Obtain Noise2Score3D code — verified & vendored at `external/Noise2Score3D/`
      (ICCV 2025, pretrained weights on HF — see [SOURCES.md](SOURCES.md))
- [x] Download pretrained checkpoint from HF (`data/checkpoints/`, 293MB) — and it
      runs on the Mac CPU (0.2s forward @ 2048 pts): their pykeops/CUDA deps turned
      out to be shim-able (see LOG.md 2026-08-17 Phase-2 entry)
- [x] `denoisers.py`: `Noise2Score3DWrapper` conforming to our `Denoiser` interface
      (`(B, N, 3) → (B, N, 3)`, frozen, Tweedie step `y + σ²·score(y)`)
- [x] Ordering-preservation / permutation-equivariance test on the real denoiser —
      **hard gate PASSED**: relative error 2e-8 under random permutation
- [x] `data.py`: ModelNet40 loading — official zip auto-download (~2GB, one-time),
      pure-torch OFF parsing + area-weighted sampling (handles ModelNet's malformed
      headers), unit-sphere normalization, seeded. Logic unit-tested without the
      download; first live run happens on the VM
- [x] **Gate:** tiny end-to-end run on the Mac — `check_denoiser.py` PASS @ N=2048
      (denoising improves MSE, spectrum extracted; sphere shape — ModelNet still open)

## Phase 3 — Full experiments ✅ (ran on the Mac; VM turned out unnecessary)

- [x] Step-size sweep at full N — plateau at c=1e-3 (frozen graph), now the default
- [x] Top 5 eigenpairs, 50 shapes × 5 categories × σ∈{0.01,0.02,0.05}
      (`outputs/phase3/`, 150 runs) — headline: **calibrated at the σ² bound
      in-distribution (1.06σ² @ σ=0.01), degrades to breakdown beyond the model's
      training range [0.004, 0.034]** — see LOG.md 2026-08-17 Phase-3 entry
- [x] Antisymmetry / PSD monitoring in every run (antisym ≤0.01 in-distribution;
      191/250 negative eigvals at out-of-range σ=0.05 — a finding, not a bug)
- [x] Convergence tracked per run (overlap history saved in each .pt)
- [ ] Optional follow-ups: σ=0.03 run (breakdown boundary), unfrozen ablation slice
      at full scale for the A/B table

## Phase 3.5 — Masked (region-restricted) uncertainty modes — **the missing half**

Phase-3 finding: whole-shape spectra are nearly flat (top-5 eigvals within ~1–8% —
posterior ≈ isotropic), so whole-shape "top modes" are near-degenerate. The reference
paper hit the same wall in images and solved it with *patch masks*; our analog is a
subset of points. This phase delivers the proposal's actual centerpiece: interpretable
geometric uncertainty modes for shape *regions*.

- [x] `spectrum.top_eigenpairs(..., mask=)`: restrict the operator to `M·J·M`
      (cf. `external/.../moments_calculations.py:get_eigvecs`'s mask handling);
      tested against the dense eigendecomposition of `M·A·M` on the toy
- [x] `data.extremity_patch_masks`: kNN patches seeded at shape extremities
- [x] `viz`: mask-aware modes/sweeps + `plot_mode_arrows` (direction is the
      interpretable content; magnitude coloring alone is too faint)
- [x] `scripts/run_masked_modes.py`: region gallery with resume
- [ ] **Gate (in progress):** first 20 region runs show spreads up to 29% @ σ=0.02
      and ~10–24% typical @ σ=0.03 (vs 1–8% whole-shape) — structure exists; the
      15-shape gallery + σ=0.03 sweep + unfrozen ablation are running for the
      comprehensive tables (see LOG.md when they land)

## Phase 4 — Analysis, visualization, report

- [ ] `viz.py`: eigenmode displacement fields on point clouds (± amounts along mode,
      like the reference repo's image sliders, but as 3D arrows / animated offsets)
- [ ] Quantitative tables: eigenvalue spectra vs σ; validation-gate results
- [ ] Failure cases + discussion (where the linearization breaks)
- [ ] Write-up / figures

## Open questions (move to LOG.md when resolved)

- ~~Autograd unavailable for the real model~~ → resolved: c-halving self-consistency
  sweep (`sweep_step_size_fd`) + subspace asymmetry probe from forward passes only
  (`antisym_energy_fd`).
- ~~Negative / inflated eigenvalues, non-PSD covariance~~ → **root-caused and fixed**
  (2026-08-17 frozen-graph LOG entry): the model rebuilds its voxel/neighbor graph
  every forward, so unfrozen finite differences measure O(1) discrete jumps, not the
  derivative. Freezing the anchor's graph pyramid (`freeze_graph: true`) makes the
  spectra clean: positive, converged, near-symmetric.
- Smooth-branch top eigenvalues sit at ~1.25–1.45σ², consistently but slightly above
  the exact-MMSE bound σ². Amortized/blind training of the score model? Check trend
  across σ and against their training noise range; discuss in the report.
- Eigenvectors of the *full* 3N×3N Jacobian vs restricting to a region mask (the
  reference repo uses patch masks; our analog = subset of points).
- Lanczos worth it over subspace iteration for k ≤ 5? (Probably not — decide by Phase 3.)
