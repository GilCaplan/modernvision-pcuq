# Workflow: local Mac ↔ GPU VM

*(The core loop: develop and smoke-test small on the Mac, then run the identical code at
full scale on the GPU VM. Scale is a config property, never a code property.)*

## The rule

> **No experiment reaches the GPU VM until the same script has passed at `local.yaml`
> scale on the Mac.**

Both machines run the same repo, same scripts, same code paths. The only thing that
changes is which YAML you pass:

```bash
# Mac (smoke): seconds-to-minutes, CPU or MPS, tiny sizes
python scripts/run_experiment.py --config configs/local.yaml

# GPU VM (full): same command, big sizes
python scripts/run_experiment.py --config configs/gpu.yaml
```

## Scale profiles

All knobs that cost compute live in the config. Shared schema, two profiles
(see [../configs/](../configs/)):

| Knob | `local.yaml` (Mac) | `gpu.yaml` (VM) | Why it scales |
|---|---|---|---|
| `device` | `auto` (mps/cpu) | `cuda` (`--override device=cpu` works; all 2026-09 runs were CPU) | — |
| `data.dataset` / `denoiser.kind` | `toy_gaussian` / `analytic_gaussian` | `modelnet40` / `noise2score3d` | real data and model |
| `data.n_shapes` | 2 | 50 | shapes evaluated |
| `data.n_points` | 2048 | 2048 | points per cloud (Jacobian is 3N×3N); the model degrades below ~2k |
| `data.sigmas` | `[0.02]` | `[0.01, 0.02, 0.05]` | noise levels |
| `spectrum.n_ev` | 2 | 5 | eigenpairs |
| `spectrum.iters` | 10 | 15 | power iterations |
| `spectrum.smooth.enabled` | false | true | smooth-mode stage, ~84 extra JVPs per shape and σ |
| `jacobian.batch_jvp` | 2 | 5 | JVPs evaluated per forward batch |
| `double_precision` | false | false | the Noise2Score3D weights are float32 |

To smoke-test the real model locally, override the local profile:
`--override device=cpu data.dataset=modelnet40 denoiser.kind=noise2score3d`
(about 45 s for 2 shapes with smooth modes on).

Guideline: a local profile should finish in **≲ 10 minutes** end-to-end. If a smoke run
can't finish in that budget, shrink the profile, don't skip the smoke run.

Deviating overrides go on the CLI (`--override spectrum.n_ev=3`) so YAMLs stay canonical.

## Where results go

- Every run writes to `outputs/<config-name>/<experiment>/` (gitignored): resolved
  config copy, metrics (JSON/CSV), figures.
- Anything worth remembering (numbers, decisions, surprises, failed approaches) gets a
  dated entry in [LOG.md](LOG.md) — the outputs dir is disposable, the log is not.
- After a real run: `python scripts/archive_metrics.py` copies every
  `outputs/*/*/metrics.json` into git-tracked `results/metrics/raw/` (it never
  replaces an archive with an older run), then `python scripts/build_results.py`
  regenerates `results/README.md` and its figures from current-pipeline sources.
- Use a distinct `--override name=<x>` for a new experiment so its metrics archive as
  `raw/<x>__run_experiment.json` instead of replacing an existing `gpu__` archive.

## VM specifics

- The VM runs locally on the Mac for now; treat it as ephemeral — nothing lives only
  there. Sync code via git (or rsync until the repo has a remote); datasets and
  checkpoints re-download via the commands below.
- Checkpoints and datasets cache under `data/` (gitignored) on both machines.

### Fresh VM setup (copy-paste)

```bash
git clone <repo-url> project && cd project     # or rsync the repo over
python -m pip install -r requirements.txt      # nothing else needed — no pykeops/pytorch3d

# vendored code (gitignored; pinned commits in external/README.md)
git clone --depth 1 https://github.com/HilaManor/GaussianDenoisingPosterior external/GaussianDenoisingPosterior
git clone --depth 1 https://github.com/Bobby645/Noise2Score3D external/Noise2Score3D

# pretrained denoiser checkpoint (293MB)
mkdir -p data/checkpoints
curl -sL -o data/checkpoints/noise2score3d_step4500.pth \
  "https://huggingface.co/bobby645/Noise2Score3D/resolve/main/model_step_4500.pth"

# verify the stack end-to-end, then run
python -m pytest tests/ -q                                  # includes real-model tests
python scripts/sanity_gaussian.py --config configs/gpu.yaml # analytic ground-truth gate
python scripts/check_denoiser.py  --config configs/gpu.yaml # real-model gate
python scripts/run_experiment.py  --config configs/gpu.yaml # full experiment
```

First `run_experiment.py` with `dataset: modelnet40` downloads the official
ModelNet40 zip (~2GB, one-time) into `data/`.

## Per-change checklist

1. Code change in `src/pcuq/` (+ test if it's numerics).
2. `python -m pytest tests/` — CPU, seconds.
3. Relevant script at `--config configs/local.yaml` on the Mac.
4. Only then: same script at `--config configs/gpu.yaml` on the VM.
5. `archive_metrics.py` + `build_results.py` (see "Where results go").
6. Result / surprise / decision → [LOG.md](LOG.md); roadmap tick → [PLAN.md](PLAN.md).
