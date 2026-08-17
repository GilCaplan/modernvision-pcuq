"""Phase-2 gate: vet the real (frozen) denoiser before trusting any spectra.

Loads Noise2Score3D, runs it on a noisy sphere, and reports the checks from the
proposal: does denoising actually reduce error, is the model permutation-equivariant
(ordering-preserving), how nonlinear is it (step-size sweep vs autograd JVP), how
asymmetric is its Jacobian, and can we extract a first spectrum end-to-end.

Usage:
    python scripts/check_denoiser.py --config configs/local.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pcuq.data import corrupt, fibonacci_sphere
from pcuq.denoisers import Noise2Score3DWrapper
from pcuq.diagnostics import antisym_energy, check_equivariance, sweep_step_size
from pcuq.spectrum import top_eigenpairs
from pcuq.utils import apply_overrides, get_device, load_config, make_out_dir, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg["seed"])
    device = get_device(cfg)
    if device.type == "mps":
        # Their scatter/graph ops are unverified on MPS; CPU is the safe Mac path.
        device = torch.device("cpu")
    out = make_out_dir(cfg, "check_denoiser")
    sigma = cfg["data"]["sigmas"][0]
    print(f"profile={cfg['name']} device={device} sigma={sigma} out={out}")

    x = fibonacci_sphere(cfg["data"]["n_points"], radius=1.0, dtype=torch.float32)[None]
    y = corrupt(x, sigma, cfg["seed"]).to(device)
    x = x.to(device)

    den = Noise2Score3DWrapper(cfg["denoiser"]["repo"], cfg["denoiser"]["checkpoint"],
                               sigma, device)

    t0 = time.time()
    with torch.no_grad():
        x_hat = den(y)
    fwd_s = time.time() - t0
    mse_noisy = float(((y - x) ** 2).mean())
    mse_denoised = float(((x_hat - x) ** 2).mean())
    print(f"forward: {fwd_s:.2f}s | MSE noisy {mse_noisy:.2e} -> denoised "
          f"{mse_denoised:.2e} ({'improves' if mse_denoised < mse_noisy else 'WORSE'})")

    equiv = check_equivariance(den, y[0])
    print(f"equivariance (relative error under permutation): {equiv:.2e}")

    metrics = {"forward_seconds": fwd_s, "mse_noisy": mse_noisy,
               "mse_denoised": mse_denoised, "equivariance_rel_err": equiv}

    jc, sp = cfg["jacobian"], cfg["spectrum"]
    try:
        metrics["step_size_sweep"] = sweep_step_size(
            den, y[0], cfg["diagnostics"]["step_size_sweep"], method=jc["method"])
        metrics["antisym_energy"] = antisym_energy(den, y[0], n_probes=2)
        print(f"step-size sweep (rel err vs autograd): "
              f"{ {c: f'{e:.2e}' for c, e in metrics['step_size_sweep'].items()} }")
        print(f"antisym energy: {metrics['antisym_energy']:.3f}")
    except Exception as e:  # autograd through their graph ops may not be viable
        metrics["autograd_error"] = repr(e)
        print(f"autograd through the model FAILED ({e!r}) — finite differences only.")

    t0 = time.time()
    eigvecs, eigvals, history = top_eigenpairs(
        den, y[0], sigma, k=sp["n_ev"], iters=sp["iters"],
        method=jc["method"], c=jc["c"], symmetrize=False)
    metrics["spectrum_seconds"] = time.time() - t0
    metrics["eigvals"] = eigvals.cpu().tolist()
    metrics["final_iter_overlap"] = history[-1].tolist()
    print(f"spectrum ({sp['n_ev']} ev, {sp['iters']} iters): "
          f"{metrics['spectrum_seconds']:.1f}s | eigvals {eigvals.cpu().numpy()} | "
          f"final overlap {history[-1].numpy()}")

    torch.save({"eigvecs": eigvecs.cpu(), "eigvals": eigvals.cpu(),
                "x": x.cpu(), "y": y.cpu(), "x_hat": x_hat.cpu()}, out / "spectrum.pt")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    gate = mse_denoised < mse_noisy and equiv < 1e-3
    print(f"\n{'PASS' if gate else 'FAIL'} (denoising improves + equivariant) — "
          f"artifacts at {out}")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
