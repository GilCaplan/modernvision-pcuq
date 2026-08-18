"""Summarize a run_experiment output directory into the tables the report needs.

Prints per-sigma aggregates (eigenvalue spectra vs sigma^2, convergence, asymmetry,
PSD violations, denoising gain) and flags runs that need attention.

Usage:
    python scripts/summarize_results.py outputs/local/run_experiment
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    out = Path(sys.argv[1])
    metrics = json.load(open(out / "metrics.json"))

    by_sigma = defaultdict(list)
    for tag, m in metrics.items():
        sigma = float(tag.rsplit("sigma", 1)[1])
        by_sigma[sigma].append((tag.rsplit("_sigma", 1)[0], m))

    for sigma in sorted(by_sigma):
        runs = by_sigma[sigma]
        s2 = sigma**2
        print(f"\n=== sigma={sigma}  (sigma^2 = {s2:.1e}, upper bound for an exact "
              f"MMSE denoiser) — {len(runs)} shapes ===")
        ev0 = np.array([m["eigvals"][0] for _, m in runs])
        conv = np.array([min(m["final_iter_overlap"]) for _, m in runs])
        asym = np.array([m["antisym_energy_subspace"] for _, m in runs])
        gain = np.array([m["mse_noisy"] / m["mse_denoised"] for _, m in runs])
        neg = sum(m["psd"]["n_negative"] for _, m in runs)
        print(f"  top eigval / sigma^2 : median {np.median(ev0) / s2:.2f}   "
              f"range [{ev0.min() / s2:.2f}, {ev0.max() / s2:.2f}]")
        print(f"  denoising gain (MSE noisy/denoised): median {np.median(gain):.2f}x")
        print(f"  convergence (min overlap, want ~1): median {np.median(conv):.3f}")
        print(f"  antisym energy in subspace: median {np.median(asym):.3f}")
        print(f"  negative eigenvalues: {neg} across all shapes")
        flagged = [(n, m) for n, m in runs if min(m["final_iter_overlap"]) < 0.9]
        for n, m in flagged:
            print(f"    ⚠ not converged: {n} (min overlap "
                  f"{min(m['final_iter_overlap']):.2f})")


if __name__ == "__main__":
    main()
