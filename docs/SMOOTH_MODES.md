# Smooth deformation uncertainty

This addition estimates uncertainty within smooth displacement fields, alongside
the existing whole-cloud modes. Neighboring points move together, and global
translations and infinitesimal rotations are excluded so the modes describe changes
in shape. It runs through `scripts/run_experiment.py`.

## Setup and running

Run commands from the project root, using a Python environment with the project
dependencies installed:

```powershell
python -m pip install -r requirements.txt
```

Real-denoiser runs require the Noise2Score3D repository and pretrained checkpoint.
If missing, restore them with these PowerShell commands (skip anything already present;
on macOS/Linux use the `curl` commands in [WORKFLOW.md](WORKFLOW.md)):

```powershell
git clone --depth 1 https://github.com/Bobby645/Noise2Score3D external/Noise2Score3D
New-Item -ItemType Directory -Force data/checkpoints | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/bobby645/Noise2Score3D/resolve/main/model_step_4500.pth" -OutFile "data/checkpoints/noise2score3d_step4500.pth"
```

See [external dependencies](../external/README.md) for provenance. The ModelNet40
loader downloads its dataset on first use if it is absent.

Start with one shape and one noise level:

```powershell
python scripts/run_experiment.py --config configs/gpu.yaml --override spectrum.smooth.enabled=true data.n_shapes=1 "data.sigmas=[0.02]"
```

To force that experiment to run again:

```powershell
python scripts/run_experiment.py --config configs/gpu.yaml --override spectrum.smooth.enabled=true data.n_shapes=1 "data.sigmas=[0.02]" --fresh
```

For the full configured experiment:

```powershell
python scripts/run_experiment.py --config configs/gpu.yaml --override spectrum.smooth.enabled=true
```

The profile name `gpu` determines the output subfolder; the `device` setting
determines where the denoiser runs. Graph eigendecomposition runs on CPU.

## Configuration

| Setting | Meaning |
|---|---|
| `data.sigmas` | Noise standard deviations: each shape is corrupted as `y = x + sigma*z`, with independent standard Gaussian coordinate noise. Values are in normalized cloud units. The same sigma is used for denoising and covariance scaling. |
| `data.n_shapes` | Total number of shapes, selected across the configured categories. |
| `spectrum.smooth.enabled` | Enables additional smooth-mode analysis; baseline analysis still runs or resumes. |
| `spectrum.smooth.n_basis` | Number of scalar low-frequency Laplacian functions; default 30. These give up to 90 coordinate displacement directions before rigid-motion removal. |
| `spectrum.smooth.n_neighbors` | Neighbors per point for the symmetric geometric graph; default 16. |
| `spectrum.n_ev` | Number of modes to return in both analyses. |
| `jacobian.batch_jvp` | Number of smooth basis directions processed together. Smaller batches reduce intermediate memory use. |
| `jacobian.method`, `jacobian.c` | Jacobian-product method and finite-difference step size. Smooth modes accept `forward`, `central`, or `autograd`. |
| `denoiser.freeze_graph` | Freezes the real denoiser's graph at the noisy anchor while computing derivatives. Keep enabled for the intended smooth-branch analysis. |
| `denoiser.graph_freeze_variant` | Which freeze both stages use: `topology` (default since the 2026-09-16 audit) or `full`. The smooth stage follows the same setting as the baseline. |

Require `1 <= n_neighbors < N`, `1 <= n_basis <= N`, and at least `n_ev`
independent directions after rigid-motion removal. `local.yaml` leaves smooth modes
off (its default denoiser is the analytic toy); `gpu.yaml` turns them on, which
roughly doubles run time (about 84 extra Jacobian products per shape and sigma). `spectrum.iters` controls the
baseline iteration only; smooth modes use direct eigendecomposition of a reduced
matrix. The smooth reduced matrix is always symmetrized.

## Inspecting the figures

With the commands above, open **`outputs/gpu/run_experiment/`** in File Explorer
and search for `*smooth*.png`. The general location is
`<out_dir>/<profile name>/run_experiment/`.

Each `<tag>` identifies a shape and noise level, such as `<shape>_sigma0.02`:

| File | What to inspect |
|---|---|
| `<tag>_smooth_modes.png` | One panel per mode, colored by displacement magnitude. Shows which regions move most, but not their direction. |
| `<tag>_smooth_mode0_arrows.png` | Red arrows show mode 0's displacement directions. Arrow lengths are enlarged for visibility; they are not uncertainty intervals. |
| `<tag>_smooth_mode0_sweep.png` | Three views of `x_hat + t*sqrt(lambda)*v`, with `t = -3, 0, +3`. Look for coordinated bending, stretching, or relative movement of parts. |

Other modes have their own numbered arrow and sweep figures. Sweeps are saved
only for positive eigenvalues. These are static images, not an interactive viewer.
The +/-3 scaling illustrates a mode; it does not establish calibrated coverage or
guarantee that the displaced shapes are posterior samples. Eigenvector signs are
arbitrary, so reversing the left/right sweep is equivalent.

## Loading saved results

Each `<tag>_smooth.pt` stores CPU tensors and metadata. This example loads the first
result found; use its exact path to select a particular shape:

```python
from pathlib import Path
import torch

paths = sorted(Path("outputs/gpu/run_experiment").glob("*_smooth.pt"))
if not paths:
    raise FileNotFoundError("Run a smooth-mode experiment first.")
result = torch.load(paths[0], map_location="cpu", weights_only=True)
print(paths[0])
print(result["eigvals"])
print(result["diagnostics"])
modes = result["eigvecs"]  # (number of modes, N, 3)
```

| Key | Contents |
|---|---|
| `x`, `y`, `x_hat` | Clean, noisy, and denoised clouds, each `(N,3)`. |
| `eigvecs`, `eigvals` | Unit displacement fields `(k,N,3)` and descending eigenvalues `(k,)`. |
| `basis` | Orthonormal smooth, non-rigid basis `(3N,d)`, using point-major xyz flattening. |
| `reduced_covariance` | Raw projected matrix `(d,d)`, before symmetrization. |
| `laplacian_eigenvalues`, `graph_edges` | Retained scalar graph frequencies and undirected edges `(2,E)`. |
| `settings`, `figures` | Settings used to check reuse and the current result's figure filenames. |
| `diagnostics`, `metrics` | Numerical diagnostics; metrics additionally contain eigenvalues, runtime, `covariance_kind` and `trustworthy`. |

A rejected run (see the symmetry gate below) stores only `x`, `y`, `x_hat`,
`settings` and `metrics`, with `metrics["smooth_rejected"]` holding the reason and
no figures.

The existing `metrics.json` also receives a `smooth` entry for each completed run,
including shapes whose whole-shape baseline was rejected by the symmetry gate.
Diagnostics include basis dimension, removed rigid dimensions, connected-component
count, covariance asymmetry, mode roughness, and projected eigenpair residuals.
The negative-eigenvalue count covers the entire reduced spectrum, with a numerical
tolerance; returned eigenvalues are not clipped. Residuals measure the reduced
symmetric eigenproblem, not error against the full Jacobian or true posterior.

Baseline and smooth stages resume independently. A completed baseline does not
prevent missing smooth results from being generated. Smooth results are recomputed
when saved relevant settings or clean input change, or listed figures are missing.
`--fresh` recomputes both stages. Changing settings does not change the output folder;
use a different `name` override to preserve multiple experiments side by side.

## Method and limitations

On `x_hat`, the solver builds a symmetric, unweighted k-nearest-neighbor graph and
its combinatorial Laplacian. Low-frequency graph eigenvectors define smoothly
varying displacement fields along all three coordinate axes. The basis is
restricted to fields orthogonal to whole-object translation and infinitesimal
rotation, then orthonormalized to form `B`.

With geometry and the anchor graph fixed, the existing Jacobian products estimate:

```text
C = sigma^2 * B.T @ J_D(y) @ B
C_symmetric = (C + C.T) / 2
C_symmetric @ a = lambda * a
v = B @ a
```

Smoothness is an explicit restriction, not evidence of structural ambiguity.
An isotropic covariance stays isotropic after projection. Nearly equal eigenvalues
can produce unstable individual modes; compare their combined subspaces instead.
Graph edges may connect distinct nearby surfaces, and disconnected components can
move independently. Compare neighborhood and basis sizes before interpreting modes.
Symmetrization does not guarantee positive covariance eigenvalues.

**Same gates as the baseline.** `smooth_eigenpairs` refuses a denoiser without a
`covariance_kind` label, and rejects the run with `ValueError` when the reduced
matrix is too asymmetric to call a covariance: relative asymmetry
`‖C − Cᵀ‖ / ‖(C + Cᵀ)/2‖ > symmetry_tol` (default 0.05, the same measure and
threshold as `top_eigenpairs`). `run_experiment.py` records a rejection instead of
crashing. Smooth modes come from a direct eigendecomposition, so there is no
convergence or Ritz-residual criterion: `trustworthy` is simply "passed the symmetry
gate".

This implementation uses dense CPU graph eigendecomposition: quadratic storage and
cubic solve cost in point count. It targets the existing approximately 2,048-point
experiments; increasing resolution substantially can be expensive.

Tests: `python -m pytest tests/test_smooth.py` covers the dense analytic
comparison, rigid-motion removal, edge cases, the symmetry gate, saving and resume
behavior (all passing as of 2026-09-24).

## Results and stability check

Results for the full 50-shape run are in [results/README.md](../results/README.md)
§5 and [LOG.md](LOG.md) (2026-09-24). Whether a smooth mode survives a new noise draw
of the same points is checked by:

```bash
python scripts/audit_smooth_seed_stability.py --config configs/gpu.yaml --override name=smooth device=cpu
```

It reads the trustworthy σ=0.02 smooth runs of `outputs/smooth/run_experiment/`,
recomputes them with a new noise seed, and compares eigenvalues, per-mode overlap and
subspace principal angles against the angles two random subspaces of the same smooth
basis would give. Archive both runs afterwards with `python scripts/archive_metrics.py`.
