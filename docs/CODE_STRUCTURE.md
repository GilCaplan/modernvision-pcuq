# Code Structure

*(How the code is organized and the rules that keep it minimal and readable.)*

## Design rules

1. **Ours vs theirs.** Our code lives in `src/pcuq/` only. Third-party code is vendored
   under `external/<repo-name>/` and **never edited**. If we need a modified version of
   their function, we rewrite it cleanly in `pcuq` with a comment pointing at the origin
   (e.g. "cf. external/GaussianDenoisingPosterior/moments_calculations.py:get_eigvecs").
2. **Minimal & readable.** Plain Python + PyTorch (+ numpy, matplotlib/plotly for viz).
   No frameworks, no Lightning, no hydra. Config = one small YAML loaded into a dataclass.
3. **One tensor convention everywhere:** point clouds are `(B, N, 3)` float32 (float64
   only when a config asks for `double_precision`). Corruption keeps indices: point `i`
   of `Y` corrupts point `i` of `X`.
4. **Everything scales by config.** No hard-coded sizes. The same script runs on the Mac
   and on the GPU VM; only the YAML differs ([WORKFLOW.md](WORKFLOW.md)).
5. **Determinism.** Every entry point takes a `seed`; results land in
   `outputs/<experiment-name>/` with the resolved config dumped alongside.

## Module map (`src/pcuq/`)

| Module | Responsibility | Key API (planned) |
|---|---|---|
| `utils.py` | Config dataclass + YAML loading, seeding, device pick (cuda→mps→cpu), output dirs | `load_config(path)`, `set_seed(s)`, `get_device(cfg)` |
| `data.py` | ModelNet40 loading, mesh→points sampling, normalization; synthetic Gaussian/GMM toy data; corruption `Y = X + σZ` with retained indices | `load_modelnet(cfg)`, `make_toy_gaussian(cfg)`, `corrupt(x, sigma, seed)` |
| `denoisers.py` | Uniform frozen-denoiser interface + implementations: `AnalyticGaussianDenoiser` (closed-form ground truth), `Noise2Score3DWrapper` (wraps `external/` model) | `Denoiser.denoise(y) -> x_hat`, all `(B,N,3)→(B,N,3)` |
| `jacobian.py` | JVP backends against a frozen denoiser at anchor `y`: forward-diff, central-diff, autograd (`torch.func.jvp`); symmetrized product | `jvp(denoiser, y, v, method, c)`, `sym_jvp(...) = ½(Jv + Jᵀv)` |
| `spectrum.py` | Top-k eigenpairs of `σ²·J` via block power/subspace iteration (QR re-orthonormalization each step); optional Lanczos | `top_eigenpairs(denoiser, y, sigma, k, iters, cfg) -> (eigvecs (k,N,3), eigvals (k,), history)` |
| `diagnostics.py` | The proposal's three challenge checks: step-size sweep, finite-diff↔autograd agreement, permutation-equivariance / ordering preservation, antisymmetric energy, PSD-ness of restricted covariance | `check_equivariance(denoiser, y)`, `antisym_energy(...)`, `sweep_step_size(...)` |
| `viz.py` | Point clouds colored by per-point uncertainty; eigenmode displacement arrows; `x ± t·√λ·v` sweeps (3D analog of the reference repo's sliders) | `plot_modes(x_hat, eigvecs, eigvals, path)` |

## Entry points (`scripts/`)

Every script: `python scripts/<name>.py --config configs/{local,gpu}.yaml [--override key=val]`

- `sanity_gaussian.py` — Phase 1 gate: analytic denoiser, compares estimated eigenpairs
  to the closed-form posterior covariance. Must pass before any real-model work.
- `run_experiment.py` — main pipeline: data → corrupt → denoise → spectrum →
  diagnostics → viz, all driven by the config.

## Reference implementation crib sheet

What we actually reuse from `external/GaussianDenoisingPosterior/` (read, don't import —
their code is image-shaped and entangled with 2D wrappers):

- `moments_calculations.py:get_eigvecs` — the subspace-iteration loop: perturb by
  `c`-scaled candidate vectors, forward pass, subtract MMSE output, QR-orthonormalize,
  eigenvalue = `‖Jv‖·σ²/c`. Our `spectrum.py` is a clean re-derivation of exactly this.
- `moments_calculations.py:_forward_directional` — the `D(y + a·v)` helper pattern.
- `models_wrappers/models_wrapper_base.py` — the frozen-model wrapper idea → our
  `Denoiser` interface.
- Higher-order moments (`calc_moments`) — **out of scope** for us unless time remains.

## Dependencies

`torch`, `numpy`, `tqdm`, `pyyaml`, `matplotlib` (+ `plotly` for interactive 3D, and
`trimesh` for ModelNet mesh sampling). Whatever the vendored denoiser needs stays listed
in *its* folder, not in our top-level `requirements.txt`, unless unavoidable.
