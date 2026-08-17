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
