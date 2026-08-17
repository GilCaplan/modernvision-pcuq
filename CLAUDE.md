# Claude Code guide for this project

Course project (Modern Vision): training-free structured uncertainty from a frozen
point-cloud denoiser. Read [docs/PROJECT.md](docs/PROJECT.md) for the what/why,
[docs/PLAN.md](docs/PLAN.md) for current status, [docs/WORKFLOW.md](docs/WORKFLOW.md)
for the local-vs-GPU process.

## Hard rules

- **Never edit anything under `external/`.** Vendored third-party code. Needed changes
  are clean rewrites in `src/pcuq/` with a pointer comment to the origin.
- **Our code:** minimal, readable, plain Python + PyTorch. No frameworks (no Lightning,
  no hydra). Point clouds are `(B, N, 3)`.
- **Everything compute-scaled is a config knob** (`configs/local.yaml` vs
  `configs/gpu.yaml`), never hard-coded. New knobs go in *both* profiles.
- **Smoke before scale:** any experiment change must pass `--config configs/local.yaml`
  on the Mac before a GPU-VM run is proposed. Local profile targets ≲10 min end-to-end.
- After a meaningful run or decision, append an entry to [docs/LOG.md](docs/LOG.md)
  (template at top of that file) and tick [docs/PLAN.md](docs/PLAN.md) checkboxes.

## Commands

```bash
python -m pytest tests/                                   # fast CPU tests
python scripts/sanity_gaussian.py --config configs/local.yaml   # Phase-1 gate
python scripts/run_experiment.py --config configs/local.yaml    # smoke pipeline
```

`data/` and `outputs/` are gitignored scratch — never rely on their contents persisting.
