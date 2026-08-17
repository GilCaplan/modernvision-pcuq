"""Phase-1 gate: analytic Gaussian sanity check.

Uses AnalyticGaussianDenoiser (closed-form MMSE denoiser for a known Gaussian prior),
runs the full jvp -> subspace-iteration pipeline, and compares the estimated eigenpairs
against the exact posterior covariance. Must pass at configs/local.yaml scale on the
Mac before any real-denoiser work (docs/PLAN.md Phase 1).

Usage:
    python scripts/sanity_gaussian.py --config configs/local.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcuq.utils import apply_overrides, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg["seed"])

    # TODO Phase 1: make_toy_gaussian -> corrupt -> AnalyticGaussianDenoiser ->
    # top_eigenpairs -> compare to closed-form posterior covariance eigenpairs.
    raise SystemExit("Not implemented yet — first item of docs/PLAN.md Phase 1.")


if __name__ == "__main__":
    main()
