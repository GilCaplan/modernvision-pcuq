"""ModelNet40 point clouds -> depth-map images -> the reference paper's OWN 2D
denoiser (external/GaussianDenoisingPosterior), through pcuq's jvp -> subspace
iteration -> diagnostics pipeline.

Cross-domain benchmark: same shapes as run_experiment.py's 3D pipeline, but
instead of Noise2Score3D on the raw point cloud, each shape is rendered to a
single-view depth map (pcuq.depth) and denoised with the paper's own MNIST CNN
or FFHQ DDPM — models trained on digits/faces, never on depth maps. The point
is to see how the reference method's own uncertainty estimate behaves under
that domain shift, compared to our in-domain 3D result.

    python scripts/run_depth2d.py --config configs/local.yaml
    python scripts/run_depth2d.py --config configs/gpu.yaml

NOTE: do not run this in the same process as anything importing pcuq.denoisers
(Noise2Score3D) — see denoisers2d.py's top-of-file warning about sys.path
collisions between the two vendored repos.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pcuq.data import corrupt, load_modelnet
from pcuq.depth import render_depth_map
from pcuq.diagnostics import antisym_energy, antisym_energy_fd, is_trustworthy
from pcuq.spectrum import top_eigenpairs
from pcuq.utils import apply_overrides, load_config, make_out_dir, set_seed
from pcuq.viz import plot_sweep_2d

RESOLUTION = {"mnist": 28, "ffhq": 256}  # fixed by each network's architecture


def to_denoiser_input(depth_img: torch.Tensor, denoiser: str) -> torch.Tensor:
    """(H, W) in [0, 1] -> the (C, H, W) tensor/range each 2D denoiser expects."""
    if denoiser == "mnist":
        return depth_img[None]  # (1, H, W), already [0, 1]
    return depth_img[None].expand(3, -1, -1) * 2 - 1  # (3, H, W), [-1, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg["seed"])
    d2 = cfg["depth2d"]
    device = torch.device("cpu")  # reference denoisers are smooth CNNs; CPU is fine
    resolution = RESOLUTION[d2["denoiser"]]
    out = make_out_dir(cfg, "run_depth2d")
    print(f"profile={cfg['name']} denoiser={d2['denoiser']} out={out}")

    shapes = load_modelnet({"data": d2["dataset"], "seed": cfg["seed"]}, torch.float32)

    if d2["denoiser"] == "mnist":
        from pcuq.denoisers2d import MNISTDenoiser2D
        den = MNISTDenoiser2D(d2["repo"], device)
    else:
        from pcuq.denoisers2d import FFHQDenoiser2D
        den = FFHQDenoiser2D(d2["repo"], device, from_t=d2["ffhq"]["from_t"])

    sp, jc, rn = d2["spectrum"], d2["jacobian"], d2["render"]

    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for name, pts in shapes:
        tag = f"{name}_{d2['denoiser']}"
        depth_img = render_depth_map(pts, resolution, axis=rn["axis"], dilate=rn["dilate"])
        x = to_denoiser_input(depth_img, d2["denoiser"])
        y = corrupt(x[None], den.sigma, seed=cfg["seed"])[0]
        with torch.no_grad():
            x_hat = den(y[None])[0]
        # Random-probe antisymmetry, independent of whether a subspace converges
        # or top_eigenpairs accepts it below -- always available, so an out-of-
        # domain shape that gets rejected still leaves a comparable number behind
        # instead of a gap (autograd-traceable: both 2D denoisers support it).
        antisym_probe = antisym_energy(den, y, n_probes=5)
        m = {"sigma": den.sigma, "antisym_energy_probe": antisym_probe,
             "covariance_kind": den.covariance_kind}

        try:
            eigvecs, eigvals, history, spectrum_diag = top_eigenpairs(
                den, y, den.sigma, k=sp["n_ev"], iters=sp["iters"],
                method=jc["method"], c=jc["c"], return_diagnostics=True)
        except ValueError as e:
            # Domain shift can legitimately make the local Jacobian too
            # asymmetric to call its eigenpairs a covariance (see docs/LOG.md
            # 2026-09-11 entries: this is the project's own established finding
            # for out-of-domain shapes, not a bug) -- record it and move on
            # instead of losing every remaining shape to one rejection.
            m["top_eigenpairs_rejected"] = str(e)
            metrics[tag] = m
            torch.save({"x": x, "y": y, "x_hat": x_hat, "depth_img": depth_img}, out / f"{tag}.pt")
            print(f"[{tag}] REJECTED by top_eigenpairs (too asymmetric) | "
                  f"antisym_probe {antisym_probe:.3f}")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            continue

        asym = antisym_energy_fd(den, y, eigvecs, method=jc["method"], c=jc["c"])
        m.update({
            "eigvals": eigvals.tolist(),
            "final_iter_overlap": history[-1].tolist(),
            "antisym_energy_subspace": asym,
            "subspace_principal_angle_history_degrees": [
                angles.tolist()
                for angles in spectrum_diag["subspace_principal_angles_degrees"]
            ],
            "final_subspace_max_angle_degrees": float(
                spectrum_diag["subspace_principal_angles_degrees"][-1].max()),
            "ritz_relative_residuals": spectrum_diag["ritz_relative_residuals"].tolist(),
        })
        m["trustworthy"] = is_trustworthy(m)
        metrics[tag] = m
        torch.save({"x": x, "y": y, "x_hat": x_hat, "eigvecs": eigvecs,
                    "eigvals": eigvals, "depth_img": depth_img,
                    "spectrum_diagnostics": spectrum_diag}, out / f"{tag}.pt")
        plot_sweep_2d(x_hat, eigvecs, eigvals, out / f"{tag}_sweep.png")
        print(f"[{tag}] eigvals {eigvals.numpy()} | eigval/sigma^2 "
              f"{(eigvals[0] / den.sigma**2).item():.2f} | antisym {asym:.3f} | "
              f"overlap {history[-1].numpy()}")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # NOTE: not merged into results/viewer_data_2d.json — that file's schema is
    # currently contested between run_images2d.py ("runs", stale/broken against
    # the file actually on disk) and run_fullspectrum2d.py ("images", quantized
    # base64 modes + PNG data URI, which viewer_template.html actually reads).
    # Wiring depth2d into the interactive viewer is a follow-up, not this step.
    print(f"\ndone — {len(metrics)} runs, artifacts at {out}")


if __name__ == "__main__":
    main()
