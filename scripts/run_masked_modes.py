"""Region-restricted uncertainty modes — the proposal's centerpiece deliverable.

Whole-shape posteriors are nearly isotropic (Phase-3 finding), so this script asks
the question at region granularity, as the reference paper did with image patches:
for kNN patches seeded at shape extremities, extract the top eigenpairs of the
masked operator M·J·M and render interpretable mode figures.

Usage:
    python scripts/run_masked_modes.py --config configs/local.yaml
Scale knobs come from the config (data.*, spectrum.*, spectrum.mask.*).
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pcuq.data import corrupt, extremity_patch_masks, load_modelnet
from pcuq.denoisers import Noise2Score3DWrapper
from pcuq.spectrum import top_eigenpairs
from pcuq.utils import apply_overrides, get_device, load_config, make_out_dir, set_seed
from pcuq.viz import plot_mode_arrows, plot_mode_sweep, plot_modes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--fresh", action="store_true",
                        help="recompute everything (default: resume)")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg["seed"])
    device = get_device(cfg)
    if device.type == "mps" and cfg["device"] == "auto":
        device = torch.device("cpu")  # measured slower than CPU for the real model
    out = make_out_dir(cfg, "masked_modes")
    jc, sp, mk = cfg["jacobian"], cfg["spectrum"], cfg["spectrum"]["mask"]
    print(f"profile={cfg['name']} device={device} regions/shape={mk['n_regions']} "
          f"frac={mk['frac']} out={out}")

    shapes = load_modelnet(cfg, torch.float32)
    den = Noise2Score3DWrapper(cfg["denoiser"]["repo"], cfg["denoiser"]["checkpoint"],
                               sigma=cfg["data"]["sigmas"][0], device=device)

    metrics_path = out / "metrics.json"
    metrics = {} if args.fresh or not metrics_path.exists() \
        else json.loads(metrics_path.read_text())
    for sigma in cfg["data"]["sigmas"]:
        den.sigma = sigma
        for si, (name, x) in enumerate(shapes):
            masks = extremity_patch_masks(x, mk["n_regions"], mk["frac"])
            x = x.to(device)
            y = corrupt(x[None], sigma, cfg["seed"] + si)[0].to(device)
            for ri, mask in enumerate(masks):
                tag = f"{name}_sigma{sigma}_r{ri}"
                if tag in metrics and (out / f"{tag}.pt").exists():
                    print(f"[{tag}] already done — skipping")
                    continue
                t0 = time.time()
                freeze = cfg["denoiser"]["freeze_graph"]
                with den.graph_frozen() if freeze else contextlib.nullcontext():
                    with torch.no_grad():
                        x_hat = den(y[None])[0]  # anchor (fills graph cache)
                    eigvecs, eigvals, history = top_eigenpairs(
                        den, y, sigma, k=sp["n_ev"], iters=sp["iters"],
                        method=jc["method"], c=jc["c"], mask=mask.to(device))

                ev = eigvals.cpu()
                spread = float((ev[0] - ev[-1]) / ev[0].clamp(min=1e-30))
                metrics[tag] = {
                    "eigvals": ev.tolist(),
                    "spread_top_to_last": spread,
                    "final_iter_overlap": history[-1].tolist(),
                    "mask_points": int(mask.sum()),
                    "seconds": time.time() - t0,
                }
                torch.save({"x_hat": x_hat.cpu(), "eigvecs": eigvecs.cpu(),
                            "eigvals": ev, "mask": mask}, out / f"{tag}.pt")
                plot_modes(x_hat, eigvecs, eigvals, out / f"{tag}_modes.png",
                           mask=mask)
                plot_mode_sweep(x_hat, eigvecs[0], eigvals[0],
                                out / f"{tag}_mode0_sweep.png", mask=mask)
                plot_mode_arrows(x_hat, eigvecs[0], eigvals[0],
                                 out / f"{tag}_mode0_arrows.png", mask=mask)
                print(f"[{tag}] {metrics[tag]['seconds']:.1f}s | "
                      f"eigvals {ev.numpy()} | spread {spread:.0%} | "
                      f"overlap {history[-1].numpy()}")
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)

    print(f"\ndone — {len(metrics)} region runs, artifacts at {out}")


if __name__ == "__main__":
    main()
