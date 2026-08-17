"""Phase-1 gate: analytic Gaussian sanity check.

Runs the full jvp -> subspace-iteration pipeline against AnalyticGaussianDenoiser
(closed-form MMSE denoiser for a known Gaussian prior) and compares the estimated
eigenpairs to the exact posterior covariance. Exits nonzero on failure.

Gate thresholds: eigenvalue relative error <= 10%, eigenvector |cos| overlap >= 0.95.

Usage:
    python scripts/sanity_gaussian.py --config configs/local.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pcuq.data import corrupt, make_toy_gaussian
from pcuq.denoisers import AnalyticGaussianDenoiser
from pcuq.diagnostics import antisym_energy, psd_report, sweep_step_size
from pcuq.spectrum import top_eigenpairs
from pcuq.utils import apply_overrides, get_device, load_config, make_out_dir, set_seed

VAL_RTOL = 0.10
VEC_OVERLAP_MIN = 0.95


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg["seed"])
    device = get_device(cfg)
    dtype = torch.float64 if cfg["double_precision"] else torch.float32
    if device.type == "mps" and dtype == torch.float64:
        print("MPS has no float64 — falling back to CPU for double precision.")
        device = torch.device("cpu")
    out = make_out_dir(cfg, "sanity_gaussian")
    print(f"profile={cfg['name']} device={device} dtype={dtype} out={out}")

    toy = make_toy_gaussian(cfg["data"]["n_points"], cfg["seed"], dtype)
    x = toy.sample(1, cfg["seed"])
    jc, sp = cfg["jacobian"], cfg["spectrum"]
    metrics, ok = {}, True

    for sigma in cfg["data"]["sigmas"]:
        den = AnalyticGaussianDenoiser(toy, sigma).to(device)
        y = corrupt(x, sigma, cfg["seed"])[0].to(device)

        eigvecs, eigvals, history = top_eigenpairs(
            den, y, sigma, k=sp["n_ev"], iters=sp["iters"],
            method=jc["method"], c=jc["c"], symmetrize=sp["symmetrize"])

        true_vecs, true_vals = toy.posterior_eigenpairs(sigma, sp["n_ev"])
        true_vecs, true_vals = true_vecs.to(device), true_vals.to(device)

        val_err = ((eigvals - true_vals).abs() / true_vals).cpu()
        overlap = (eigvecs.reshape(sp["n_ev"], -1)
                   * true_vecs.reshape(sp["n_ev"], -1)).sum(dim=1).abs().cpu()
        m = {
            "true_eigvals": true_vals.cpu().tolist(),
            "est_eigvals": eigvals.cpu().tolist(),
            "eigval_rel_err": val_err.tolist(),
            "eigvec_overlap": overlap.tolist(),
            "final_iter_overlap": history[-1].tolist(),
            "antisym_energy": antisym_energy(den, y),
            "step_size_sweep": sweep_step_size(
                den, y, cfg["diagnostics"]["step_size_sweep"], method=jc["method"]),
            "psd": psd_report(eigvals.cpu()),
        }
        metrics[f"sigma={sigma}"] = m

        print(f"\n== sigma={sigma} ==")
        for i in range(sp["n_ev"]):
            print(f"  ev{i}: true={true_vals[i]:.3e}  est={eigvals[i]:.3e}  "
                  f"rel_err={val_err[i]:.1%}  |cos|={overlap[i]:.4f}")
        print(f"  antisym_energy={m['antisym_energy']:.2e}  "
              f"sweep={ {c: f'{e:.1e}' for c, e in m['step_size_sweep'].items()} }")

        passed = bool((val_err <= VAL_RTOL).all() and (overlap >= VEC_OVERLAP_MIN).all())
        m["passed"] = passed
        ok &= passed

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n{'PASS' if ok else 'FAIL'} — metrics at {out / 'metrics.json'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
