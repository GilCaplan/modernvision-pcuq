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

| Module | Responsibility | Key API |
|---|---|---|
| `utils.py` | Config YAML loading + CLI overrides, seeding, device pick (cuda→mps→cpu), output dirs | `load_config(path)`, `apply_overrides(cfg, ovs)`, `set_seed(s)`, `get_device(cfg)`, `make_out_dir(cfg, name)` |
| `data.py` | ModelNet40 loading (official zip, OFF parsing, area-weighted sampling, unit-sphere normalization); toy Gaussian and Gaussian-mixture data; corruption `Y = X + σZ` with retained indices; extremity region masks | `load_modelnet(cfg)`, `make_toy_gaussian(...)`, `make_toy_gmm(...)`, `corrupt(x, sigma, seed)`, `extremity_patch_masks(...)` |
| `denoisers.py` | Frozen-denoiser interface + implementations: `AnalyticGaussianDenoiser`, `AnalyticGMMDenoiser` (closed-form ground truth), `Noise2Score3DWrapper` (wraps `external/`); `covariance_kind` label on every denoiser; graph freezing (`graph_frozen`, `graph_topology_frozen`) | `Denoiser.denoise(y)`, `COVARIANCE_KINDS`, `.graph_topology_frozen()` |
| `denoisers2d.py` | The reference paper's own 2D denoisers (MNIST CNN, FFHQ DDPM) behind the same interface | used by `run_images2d.py`, `run_depth2d.py` |
| `depth.py` | Single-view orthographic depth-map rendering of a point cloud (depth-map side quest) | `render_depth_map(points, resolution, axis, dilate)` |
| `jacobian.py` | JVP backends against a frozen denoiser at anchor `y`: forward-diff, central-diff, autograd; symmetrized product | `jvp(denoiser, y, v, method, c)`, `sym_jvp(...)` |
| `spectrum.py` | Top-k eigenpairs of `σ²·J`: subspace iteration + Rayleigh–Ritz with a symmetry gate (optionally region-masked); smooth, non-rigid modes in a low-frequency Laplacian basis with rigid motion removed (same gates) | `top_eigenpairs(den, y, sigma, k, iters, mask=, return_diagnostics=)`, `smooth_eigenpairs(den, y, x_hat, sigma, k, n_basis, n_neighbors)` |
| `diagnostics.py` | Step-size sweeps, permutation equivariance, antisymmetric energy, PSD report, composite `trustworthy` flag | `check_equivariance`, `antisym_energy_fd`, `sweep_step_size_fd`, `psd_report`, `is_trustworthy` |
| `viz.py` | Mode figures: per-point magnitude, displacement arrows, `x̂ ± t·√λ·v` sweeps (3D) and 2D image sweeps | `plot_modes`, `plot_mode_arrows`, `plot_mode_sweep`, `plot_sweep_2d` |

## Entry points (`scripts/`)

Every script: `python scripts/<name>.py --config configs/{local,gpu}.yaml [--override key=val]`
(except the results/viewer builders, which take no config).

| Script | What it does |
|---|---|
| `sanity_gaussian.py` | Phase-1 gate: analytic Gaussian + GMM, estimated vs closed-form eigenpairs |
| `check_denoiser.py` | Phase-2 gate: vets the real frozen denoiser (MSE, equivariance, spectrum) |
| `run_experiment.py` | Main pipeline: data → corrupt → denoise → whole-shape spectrum → diagnostics → figures; plus smooth modes when `spectrum.smooth.enabled` |
| `run_masked_modes.py` | Region-restricted modes (extremity patches) |
| `run_fullspectrum.py`, `run_fullspectrum2d.py` | Rank-k spectra that back the viewer's free-form masks |
| `run_images2d.py` | The reference method on its own domain (MNIST, FFHQ) through our pipeline |
| `run_depth2d.py` | Side quest: ModelNet shapes as depth maps through the reference 2D denoisers |
| `audit_graph_freezing.py` | `full` vs `topology` graph freezing on real shapes |
| `audit_seed_stability.py`, `audit_region_seed_stability.py`, `audit_smooth_seed_stability.py` | Do whole-shape / region / smooth modes survive a new noise seed? |
| `summarize_results.py` | Tables from one `run_experiment` output directory |
| `archive_metrics.py` | Copies `outputs/*/*/metrics.json` into git-tracked `results/metrics/` (never over a newer archive) |
| `build_results.py` | Regenerates `results/README.md` + figures from current-pipeline sources |
| `export_viewer_data.py` + `viewer_template.html` | Data bundle and template for the interactive Uncertainty Mode Explorer |
| `export_viewer_shapes.py` | Explorer data for every `run_experiment` shape (whole-shape + smooth modes): an index embedded as `SHAPES50` plus one `shapes50_<σ>.json` per σ, published beside the page and fetched on demand |

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
