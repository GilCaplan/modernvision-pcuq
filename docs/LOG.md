# Experiment & Decision Log

*(Append-only, newest entry on top. One entry per: experiment result worth keeping,
design decision, surprise, or dead end. `outputs/` is disposable; this file is not.)*

Entry template:

```markdown
## YYYY-MM-DD — <short title>
**Who:** · **Machine:** mac | gpu-vm · **Config:** configs/<file> (+ overrides)
**What:** one paragraph — what was run/decided and why.
**Result:** numbers, paths to figures under outputs/, or the decision taken.
**Next:** what this implies.
```

---

## 2026-08-17 — Phase-3 sweep complete: calibrated in-distribution, breaks beyond training σ
**Who:** Claude (with Rocky) · **Machine:** mac (CPU) · **Config:** local + overrides
(name=phase3: 50 shapes × σ∈{0.01,0.02,0.05} × 5 eigenpairs, 15 iters, frozen graph)
**What:** Full-scale sweep, 150 runs in ~75 min, `outputs/phase3/run_experiment/`.
**Result (per σ, medians over 50 shapes):**
- **σ=0.01: top eigval 1.06σ² [0.81, 1.59], 0 negative eigvals (of 250), convergence
  0.999, antisym 0.001.** Essentially at the exact-MMSE bound — the method is
  *calibrated* in-distribution.
- **σ=0.02: 1.39σ² [1.12, 2.33], 2 negatives, convergence 0.996, antisym 0.009.**
  Mild inflation, growing with σ.
- **σ=0.05: breakdown** — eigvals scattered [-7σ², +7σ²], 191/250 negative, median
  convergence 0.41. Explanation found in their code: the model was trained with
  σ annealed over **[0.004, 0.034]** (`models/KPconv.py:159`); σ=0.05 is ~50% beyond
  the training range, so the score field (and its Jacobian) is extrapolating.
  Denoising MSE still improves there (1.35×) — the *mean* extrapolates better than
  the *derivative*, a nice report point.
The earlier "~1.3σ² anomaly" is now a clean σ-trend: 1.06 → 1.39 → breakdown as σ
approaches/exceeds the training range. Interpretation: score-Jacobian calibration
degrades near the edge of the amortization range.
**Next:** Report material is essentially complete: calibration-vs-σ table, the
frozen-graph A/B, mode galleries. Optional: rerun σ=0.05 → 0.03 (inside training
range) to show the breakdown boundary; unfrozen ablation slice for the A/B table.

## 2026-08-17 — MPS measured: works, but slower than CPU for the real model
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** —
**What:** Tested whether the real denoiser can use Apple's GPU (MPS) instead of CPU.
Required generalizing the `.cuda()` shim from "no-op" to "redirect to the wrapper's
device" (also fixed a shim bug: second wrapper construction crashed on the spec-less
pykeops stub). No `PYTORCH_ENABLE_MPS_FALLBACK` needed.
**Result (N=2048, 5-forward average):** CPU 244 ms/forward, MPS 344 ms/forward —
MPS is ~1.4× *slower*: this KPConv pipeline is many small irregular kernels
(radius search, gather/scatter, pack-mode reshapes), which MPS launch overhead
dominates; Apple GPUs win on big dense matmuls, not this. Outputs agree to ~3% of
output std (kernel differences). Policy: `device: auto` picks CPU for the real model
(explicit `device: mps` is honored); the analytic/toy path keeps using MPS, where its
dense matmuls do win.
**Next:** —

## 2026-08-17 — First real ModelNet results; graph-rebuild discontinuity found & fixed
**Who:** Claude (with Rocky) · **Machine:** mac (CPU) · **Config:** local + overrides
**What:** Ran the first real experiment locally (ModelNet40 downloaded in ~2 min):
10 shapes (chair/airplane/table/lamp/guitar ×2) × σ∈{0.01,0.02,0.05} × 5 eigenpairs,
~10s per run, `outputs/local/run_experiment/`. Results were bad in an instructive way:
top eigenvalues 4–36× above the σ² MMSE bound, negative eigenvalues, subspace antisym
energy ~0.4–0.5, poor convergence, and **no step-size plateau** (40–80% c-halving
error everywhere). Diagnosis: the model rebuilds its voxel/radius graph every forward,
so perturbed passes flip discrete assignments — finite differences measure O(1) jumps,
not the local derivative.
**Fix:** `Noise2Score3DWrapper.graph_frozen()` — freeze the anchor's graph pyramid
(discrete indices + coarse-level positions), let only input points vary: differentiate
the *smooth branch*. A/B on 5 shapes @ σ=0.02 (`outputs/local-frozen/run_experiment/`):
eigvals all positive, tight 1.29–1.45σ² (was median 7σ², max 32σ²); overlaps ~0.99;
antisym energy 0.001–0.017 (was ~0.4); sweep shows a clean U with plateau at c=1e-3
(now the config default; error 2.2e-3 there). `freeze_graph: true` is the default in
both profiles. Mode figures show localized structure (lamp arm / base as separate
modes). Remaining anomaly: eigvals consistently ~1.3σ², slightly above the exact-MMSE
bound — see PLAN.md open questions.
**Next:** Full gpu.yaml sweep (VM or overnight Mac), quantitative eigval-vs-σ tables
via `scripts/summarize_results.py`, mode-figure gallery for the report.

## 2026-08-17 — GPU-ready: ModelNet40 loader, full run_experiment pipeline, VM runbook
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** configs/local.yaml
**What:** Closed the gaps between "gates pass" and "VM can run the real experiment":
ModelNet40 loader (official zip, pure-torch OFF parse + area-weighted sampling —
dropped the trimesh dep), fully implemented `run_experiment.py` (shapes × σ, spectrum,
per-σ diagnostics incl. new autograd-free step-size sweep, mode figures, incremental
metrics.json), matplotlib viz, VM setup runbook in WORKFLOW.md. requirements.txt alone
now suffices on the VM (no pykeops/pytorch3d).
**Result:** 13 tests pass. Local runs of `run_experiment.py`: analytic+toy PASS (19s,
MPS, eigval overlap 0.9999); real-model smoke PASS mechanically (CPU). Smoke caught and
fixed a real bug: coarse pyramid levels with fewer support points than neighbor_limit
broke the kNN shim (now inf-padded = "no neighbor"). Finding: on the OOD toy blob the
real model's top |eigvals| are *negative* (non-PSD, as the proposal warned) — logged as
an open question to re-check on in-distribution ModelNet shapes.
**Next:** On the VM: WORKFLOW.md runbook top to bottom; first `gpu.yaml` run downloads
ModelNet40 (~2GB). Expect ~10s/(shape·σ) on CPU-scale timing — much less on GPU.

## 2026-08-17 — Phase 2 (nearly) done: real denoiser runs ON THE MAC; equivariance gate passed
**Who:** Claude (with Rocky) · **Machine:** mac (CPU) · **Config:** configs/local.yaml
**What:** Wrote `Noise2Score3DWrapper` + `scripts/check_denoiser.py`. Made the vendored
model run without CUDA/pykeops via runtime shims (no vendored files edited): no-op
`.cuda()` when CUDA is absent; swap their pykeops kNN for exact `cdist`+`topk`; their
`dataloader()`'s hardcoded `.cuda()` replaced device-safely; checkpoint loaded with
`strict=False` guarded to allow only the zero-init conv biases the checkpoint lacks.
**Result (N=2048, σ=0.02, sphere):** forward 0.2s CPU; **equivariance 1.7e-8 — the
proposal's hard gate PASSED**; denoising improves MSE 4.02e-4 → 3.32e-4; 2-eigvec
spectrum in ~6s (finite differences). At N=256 denoising *hurt* MSE — the model needs
training-like density (trained on 10k–50k-pt ModelNet), hence local n_points now 2048.
Two surprises → PLAN.md open questions: `torch.func` autograd cannot trace their graph
ops (finite differences only — sweep needs an autograd-free reference), and estimated
eigenvalues ~1.5σ² exceed the MMSE bound σ². Sanity gate re-verified at N=2048 (PASS,
15s, MPS).
**Next:** ModelNet40 loading (last open Phase-2 item), then Phase-3 sweeps on the VM.

## 2026-08-17 — Noise2Score3D availability confirmed; Phase-2 blocker cleared
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** —
**What:** Verified the primary denoiser exists publicly: official ICCV 2025 code at
github.com/Bobby645/Noise2Score3D with pretrained weights on Hugging Face
(bobby645/Noise2Score3D). Vendored at `external/Noise2Score3D/` (commit `fef67d7`).
**Result:** KPConv-based, trained on ModelNet-40 — exact match to our proposal. Two
caveats: no upstream license file (note in report), and heavy deps (PyTorch3D,
pykeops, CUDA-era pins) → the real denoiser likely runs only on the GPU VM; Mac smoke
runs keep the analytic denoiser. ScoreDenoise fallback no longer needed but stays
documented in SOURCES.md.
**Next:** Download checkpoint, stand up their env on the VM, then the
ordering-preservation gate before trusting any spectra.

## 2026-08-17 — Phase 1 complete: toy pipeline validated, gate passes
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** configs/local.yaml
**What:** Implemented the full Phase-1 machinery: `ToyGaussian` prior with closed-form
posterior, `AnalyticGaussianDenoiser`, JVPs (forward/central/autograd), subspace
iteration with Rayleigh-quotient eigenvalues, and the diagnostics suite. 7 tests pass.
**Result:** `sanity_gaussian.py` at local scale (N=256, k=2, 10 iters, MPS, float32):
PASS in ~3.5s. Eigenvalue rel err 0.2–0.3%, eigenvector |cos| 0.995, antisym energy
1e-7. Step-size sweep on float32 shows error *rising* as c shrinks (1e-2 → 3.6e-4 err;
1e-5 → 0.36) — pure cancellation, since the toy denoiser is linear (no nonlinearity
penalty at large c). Metrics: outputs/local/sanity_gaussian/metrics.json.
**Next:** For *real* (nonlinear) denoisers the sweep will be U-shaped; expect the
float32 cancellation floor to matter and double precision (CPU/CUDA only — MPS has no
float64) to be needed for small c. Phase 2: verify Noise2Score3D availability.

## 2026-08-17 — Project scaffolded
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** —
**What:** Created the docs system, package skeleton, and scale-profile configs. Vendored
`GaussianDenoisingPosterior` (shallow clone) into `external/`.
**Result:** Structure as described in [CODE_STRUCTURE.md](CODE_STRUCTURE.md). Read of
`moments_calculations.py:get_eigvecs` confirms the reference algorithm is subspace
iteration with forward-difference JVPs (`c=1e-6`, eigenvalue `‖Jv‖σ²/c`, QR
re-orthonormalization) — directly portable to `(B,N,3)` tensors.
**Next:** Phase 1 (toy Gaussian pipeline) per [PLAN.md](PLAN.md); verify Noise2Score3D
code availability early (Phase 2 blocker).
