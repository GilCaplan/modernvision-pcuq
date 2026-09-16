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
- [x] Graph-freezing audit (small scale, 2026-09-16, see LOG.md): `graph_frozen`
      (the freezing used for the 1.06σ² headline above) is REJECTED by
      `top_eigenpairs`'s symmetry gate on 2/3 real shapes tested; a new, weaker
      `graph_topology_frozen` variant (topology fixed, coarse coordinates continuous)
      was never rejected and is now the default (`denoiser.graph_freeze_variant`).
- [x] Partial re-run at n=15 shapes/sigma (2026-09-16, see LOG.md): in-distribution
      headline (1.06σ² @ 0.01, 1.39σ² @ 0.02) replicates closely (1.10σ², 1.41σ²)
      under the new default. New finding: **8/15 shapes (53%) get rejected by the
      symmetry gate at >=1 sigma**, 20% at sigma=0.01 alone — invisible under the old
      eigensolver, which never rejected anything. σ=0.05 (out of training range)
      still breaks down, now with much better convergence (0.97 vs 0.41) — the old
      number there was noisy, not more correct.
- [x] Full n=50 confirmation (2026-09-16, see LOG.md): in-distribution headline
      confirmed almost exactly (1.079σ²/1.396σ² vs old 1.06σ²/1.39σ²). Rejection
      rate is real and monotonic with sigma: **10% / 16% / 26%** (0.01/0.02/0.05),
      **44% of shapes (22/50) rejected at >=1 sigma**. sigma=0.05 correction: the
      n=15 run's apparent convergence recovery (0.97) was a small-sample fluke — at
      n=50 it's 0.616, and **65% of shapes that pass the symmetry gate at sigma=0.05
      are still poorly converged (<0.9)**, so the real out-of-range breakdown is
      worse and more pervasive than either the old number or the n=15 partial run
      showed.
- [x] sigma=0.05 reporting decided (2026-09-16, see LOG.md): added a composite
      `trustworthy` flag (`pcuq.diagnostics.is_trustworthy` — not rejected AND
      convergence>=0.9 AND max Ritz residual<=0.15). Headline: **86% / 72% / 16%
      trustworthy at sigma=0.01/0.02/0.05** — quote this, not a bare median, for
      sigma=0.05 in the report.
- [x] Report reframed from "calibration" to "sensitivity vs noise level"
      (`results/README.md` §1, `docs/PROJECT.md`'s math section) — Noise2Score3D has
      no per-sigma proof of being the MMSE denoiser at each queried sigma. See
      LOG.md 2026-09-16.
- [x] Reproducibility artifacts (task 8): checkpoint SHA256 now recorded per run
      (`Noise2Score3DWrapper.checkpoint_sha256`); the n=50 run's raw metrics.json is
      archived to `results/metrics/raw/` (git-tracked), not left to become another
      "LOST" entry.
- [x] Task 9, second half (2026-09-16, see LOG.md): **whole-shape mode DIRECTIONS
      are noise artifacts; magnitude is not.** 15 trustworthy sigma=0.02 shapes,
      re-noised with a new seed: eigenvalue ratio stable (median 0.95) but median
      max subspace angle vs. the original = **89.4 degrees** (essentially
      orthogonal) — a mechanism for the flat-spectrum finding above, not just a
      restatement: near-degenerate eigenvalues (median 16.6% top-to-5th spread) make
      the reported eigenvectors numerically ill-defined. Point resampling is much
      gentler (median eigenvalue shift ~8%, only 1/15 rejected vs. 3/15 for
      reseeding). Whole-shape mode *direction* figures should not be presented as a
      stable per-shape property in the report; magnitude (top eigval/sigma^2) can be.
- [x] Follow-up run (2026-09-16, see LOG.md): does region-restricted mode direction
      survive a new noise seed better than whole-shape? **No — tested and rejected.**
      Median max subspace angle for regions = **89.34 degrees**, statistically the
      same as whole-shape's 89.4 degrees. Region-restriction fixes eigenvalue
      spread/degeneracy (Phase 3.5's finding stands), but does NOT fix eigenvector-
      direction reproducibility across noise seeds — a genuine negative result, not
      the improvement predicted. Every mode-direction figure in the report (whole-
      shape or region) should be captioned as "for one specific noisy observation,"
      not a reproducible per-shape property.
- [x] Region-mode exemplars (results/README.md §4) refreshed under the current
      pipeline and `trustworthy`-filtered (2026-09-16) — the 6 pre-fix exemplars
      were never checked against any of this session's corrections. New set: 5
      exemplars, all sigma=0.02 (diversity across sigma traded for validation;
      restoring it needs another run, not done here).

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

## Side quest — depth-map 2D benchmark (not started)

Different from Phase 4's image-domain comparison (which runs the reference paper's
own denoisers on *their* MNIST/FFHQ images for a like-for-like sanity check). This
takes OUR ModelNet40 shapes, renders each to a single-view depth-map image, and runs
the reference paper's own 2D denoiser on that image — testing the original method
out of its training domain (digits/faces) on a depth-map "photo" of a 3D shape.

- [x] Design + `pcuq.depth.render_depth_map` (single-view orthographic z-buffer,
      tests are pure-synthetic, no download needed) + `scripts/run_depth2d.py` +
      `depth2d:` config block in both profiles (local: MNIST CNN, no extra download;
      gpu: FFHQ DDPM, needs `ffhq.pt`)
- [x] **Gate:** `run_depth2d.py --config configs/local.yaml` scaled to 50 shapes ×
      5 categories, MNIST CNN — 2026-09-11: converged (median overlap 1.000), **~50%
      of shapes have a negative eigenvalue**, mostly LOW antisym (unlike the 3D
      noise-range breakdown, which is high-antisym) — a systematic, different-
      mechanism breakdown signature. Explicit decision: stay training-free/frozen-
      model (asked & answered) rather than fit a depth-map-specific denoiser.
- [x] Robustness check: swept `depth2d.render.axis`/`dilate` — the ~50% negative-
      eigenvalue rate and low antisym are stable across renderings (44-52%); the
      *magnitude* (median eigval/σ², fully-negative %) is NOT — 8.9/16% (baseline)
      vs 86.3/2% (no hole-filling). Report the qualitative pattern as the finding;
      quote severity numbers with their render config attached, see LOG.md.
- [ ] Decide if depth maps need per-shape intensity/contrast normalization before
      the reference denoiser's fixed training sigma is meaningful on them
- [ ] Re-tally the ~50%/44-52% non-PSD rate above under the 2026-09-16 pipeline fixes
      (LOG.md): the old eigensolver never rejected anything, so a subset of those
      shapes were actually antisym~1 (near-orthogonal Jv/J^Tv, essentially garbage,
      3/10 in a small resample) rather than the mild low-antisym breakdown the 44-52%
      figure describes — two tiers, not one; the report should split them
- [ ] Track down `ffhq.pt`'s exact source (not recorded anywhere in the repo — a
      pre-existing gap) if a natural-image comparison point is wanted
- [ ] Qualitative + eigval/σ² comparison against the in-domain 3D result; viewer
      integration is a separate follow-up (results/viewer_data_2d.json's schema is
      currently contested between two existing scripts — see LOG.md)

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
