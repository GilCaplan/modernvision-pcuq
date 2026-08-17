# Training-Free Structured Uncertainty Estimation from a Frozen Point-Cloud Denoiser

Estimate structured posterior uncertainty (top covariance eigenpairs) directly from the
Jacobian of a **frozen** point-cloud denoiser — no retraining, no posterior sampling.
We adapt Manor & Michaeli (ICLR 2024), which did this for 2D images, to unstructured
3D point clouds (ModelNet40, Gaussian corruption `Y = X + σZ`).

**Team:** Erel, Galit, Gil, Ido, Ori, Yakov · Modern Vision, Semester 8

## Documentation map

| Doc | What it holds |
|---|---|
| [docs/PROJECT.md](docs/PROJECT.md) | The proposal: goal, method, our delta over prior work, challenges |
| [docs/PLAN.md](docs/PLAN.md) | Phased roadmap with checkboxes — **the living to-do list** |
| [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) | Module map, design rules, what goes where |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | How we work: local-Mac smoke runs vs GPU-VM full runs, scale profiles |
| [docs/SOURCES.md](docs/SOURCES.md) | Papers, repos, datasets, checkpoints |
| [docs/LOG.md](docs/LOG.md) | Append-only experiment & decision log |
| [external/README.md](external/README.md) | What's vendored and what we actually use from it |

## Layout

```
project/
├── docs/                  # all project documentation (see map above)
├── configs/               # scale profiles: local.yaml (Mac smoke) / gpu.yaml (VM full)
├── src/pcuq/              # OUR code — minimal, readable, pure Python + PyTorch
├── scripts/               # entry points; every script takes --config
├── tests/                 # tiny CPU-only sanity tests
├── external/              # vendored third-party code, never edited, never mixed with ours
│   └── GaussianDenoisingPosterior/   # Manor & Michaeli ICLR 2024 (reference implementation)
├── data/                  # datasets (gitignored)
└── outputs/               # experiment results (gitignored)
```

Vendored clones under `external/` are gitignored (they're large); `external/README.md`
records the exact upstream commit and the re-clone command.

## Quickstart

```bash
python -m pip install -r requirements.txt

# Smoke run on the Mac (tiny scale, CPU/MPS):
python scripts/run_experiment.py --config configs/local.yaml

# Full run on the GPU VM (same code, bigger numbers):
python scripts/run_experiment.py --config configs/gpu.yaml
```

The **only** difference between a local smoke run and a full GPU run is the config
file — code paths are identical. See [docs/WORKFLOW.md](docs/WORKFLOW.md).
