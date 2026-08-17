"""Main pipeline: data -> corrupt -> denoise -> spectrum -> diagnostics -> viz.

Usage (see docs/WORKFLOW.md — smoke locally before any GPU run):
    python scripts/run_experiment.py --config configs/local.yaml
    python scripts/run_experiment.py --config configs/gpu.yaml --override spectrum.n_ev=3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcuq.utils import apply_overrides, get_device, load_config, make_out_dir, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/local.yaml or configs/gpu.yaml")
    parser.add_argument("--override", nargs="*", default=[], help="e.g. spectrum.n_ev=3")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg["seed"])
    device = get_device(cfg)
    out = make_out_dir(cfg, "run_experiment")
    print(f"profile={cfg['name']} device={device} out={out}")

    # TODO Phase 1+: build dataset (pcuq.data), denoiser (pcuq.denoisers),
    # top_eigenpairs (pcuq.spectrum), diagnostics, viz — per docs/PLAN.md.
    raise SystemExit("Pipeline not implemented yet — see docs/PLAN.md Phase 1.")


if __name__ == "__main__":
    main()
