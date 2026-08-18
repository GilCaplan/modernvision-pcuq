"""Main pipeline: data -> corrupt -> denoise -> spectrum -> diagnostics -> viz.

Loops over shapes x noise levels from the config; writes per-run artifacts
(spectrum .pt, mode figures) and a summary metrics.json.

Usage (see docs/WORKFLOW.md — smoke locally before any GPU run):
    python scripts/run_experiment.py --config configs/local.yaml
    python scripts/run_experiment.py --config configs/gpu.yaml
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pcuq.data import corrupt, load_modelnet, make_toy_gaussian
from pcuq.denoisers import AnalyticGaussianDenoiser, Noise2Score3DWrapper
from pcuq.diagnostics import (antisym_energy, antisym_energy_fd, check_equivariance,
                              psd_report, sweep_step_size, sweep_step_size_fd)
from pcuq.spectrum import top_eigenpairs
from pcuq.utils import apply_overrides, get_device, load_config, make_out_dir, set_seed
from pcuq.viz import plot_mode_sweep, plot_modes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--fresh", action="store_true",
                        help="recompute everything (default: resume, skipping "
                             "(shape, sigma) runs whose artifacts already exist)")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg["seed"])
    device = get_device(cfg)
    kind = cfg["denoiser"]["kind"]
    analytic = kind == "analytic_gaussian"
    if device.type == "mps" and not analytic:
        device = torch.device("cpu")  # their graph ops are unverified on MPS
    dtype = torch.float64 if (cfg["double_precision"] and analytic) else torch.float32
    if cfg["double_precision"] and not analytic:
        print("note: double_precision ignored for noise2score3d (float32 weights).")
    if device.type == "mps" and dtype == torch.float64:
        device = torch.device("cpu")  # MPS has no float64
    out = make_out_dir(cfg, "run_experiment")
    print(f"profile={cfg['name']} denoiser={kind} device={device} dtype={dtype} out={out}")

    toy = None
    if cfg["data"]["dataset"] == "toy_gaussian":
        toy = make_toy_gaussian(cfg["data"]["n_points"], cfg["seed"], dtype)
        xs = toy.sample(cfg["data"]["n_shapes"], cfg["seed"])
        shapes = [(f"toy{i}", xs[i]) for i in range(len(xs))]
    else:
        shapes = load_modelnet(cfg, dtype)
    if analytic and toy is None:
        raise SystemExit("analytic_gaussian denoiser only makes sense with toy_gaussian data")

    wrapper = None  # one model load, sigma updated per noise level
    if not analytic:
        wrapper = Noise2Score3DWrapper(cfg["denoiser"]["repo"],
                                       cfg["denoiser"]["checkpoint"],
                                       sigma=cfg["data"]["sigmas"][0], device=device)

    jc, sp, dg = cfg["jacobian"], cfg["spectrum"], cfg["diagnostics"]
    # Crash-safe: metrics.json and per-run .pt files are written after every run,
    # and a rerun resumes from them instead of recomputing (unless --fresh).
    metrics_path = out / "metrics.json"
    metrics = {} if args.fresh or not metrics_path.exists() \
        else json.loads(metrics_path.read_text())
    for sigma in cfg["data"]["sigmas"]:
        if analytic:
            den = AnalyticGaussianDenoiser(toy, sigma).to(device)
        else:
            wrapper.sigma = sigma
            den = wrapper
        # Symmetrization needs J^T v via autograd, which their ops don't support.
        symmetrize = sp["symmetrize"] and analytic

        # Real model: differentiate the smooth branch (graph pyramid frozen at the
        # anchor) — unfrozen finite differences are jump-dominated, see LOG.md.
        freeze = (not analytic) and cfg["denoiser"]["freeze_graph"]

        for si, (name, x) in enumerate(shapes):
            tag = f"{name}_sigma{sigma}"
            if tag in metrics and (out / f"{tag}.pt").exists():
                print(f"[{tag}] already done — skipping (use --fresh to redo)")
                continue
            t0 = time.time()
            x = x.to(device)
            y = corrupt(x[None], sigma, cfg["seed"] + si)[0].to(device)

            m = {}
            if si == 0 and dg["check_equivariance"]:  # needs real graph rebuilds
                m["equivariance_rel_err"] = check_equivariance(den, y)

            with den.graph_frozen() if freeze else contextlib.nullcontext():
                with torch.no_grad():
                    x_hat = den(y[None])[0]  # anchor forward (fills the graph cache)

                eigvecs, eigvals, history = top_eigenpairs(
                    den, y, sigma, k=sp["n_ev"], iters=sp["iters"],
                    method=jc["method"], c=jc["c"], symmetrize=symmetrize)

                m.update({
                    "mse_noisy": float(((y - x) ** 2).mean()),
                    "mse_denoised": float(((x_hat - x) ** 2).mean()),
                    "eigvals": eigvals.cpu().tolist(),
                    "final_iter_overlap": history[-1].tolist(),
                    "psd": psd_report(eigvals.cpu()),
                    # asymmetry of J restricted to the uncertainty subspace
                    "antisym_energy_subspace": antisym_energy_fd(
                        den, y, eigvecs, method=jc["method"], c=jc["c"]),
                })
                if si == 0:  # per-sigma diagnostics, once on the first shape
                    sweep = sweep_step_size if analytic else sweep_step_size_fd
                    m["step_size_sweep"] = sweep(den, y, dg["step_size_sweep"],
                                                 method=jc["method"])
                    if analytic:
                        m["antisym_energy_probes"] = antisym_energy(den, y)
            m["seconds"] = time.time() - t0
            metrics[tag] = m
            torch.save({"x": x.cpu(), "y": y.cpu(), "x_hat": x_hat.cpu(),
                        "eigvecs": eigvecs.cpu(), "eigvals": eigvals.cpu(),
                        "overlap_history": history}, out / f"{tag}.pt")
            plot_modes(x_hat, eigvecs, eigvals, out / f"{tag}_modes.png")
            plot_mode_sweep(x_hat, eigvecs[0], eigvals[0], out / f"{tag}_mode0_sweep.png")
            print(f"[{tag}] {m['seconds']:.1f}s | mse {m['mse_noisy']:.2e}->"
                  f"{m['mse_denoised']:.2e} | eigvals {eigvals.cpu().numpy()} | "
                  f"overlap {history[-1].numpy()}")

            with open(out / "metrics.json", "w") as f:  # rewrite as we go
                json.dump(metrics, f, indent=2)

    print(f"\ndone — {len(metrics)} runs, artifacts at {out}")


if __name__ == "__main__":
    main()
