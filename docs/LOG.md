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
