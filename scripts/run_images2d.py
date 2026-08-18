"""The original method's domain (2D images) through OUR pipeline.

Runs the reference repo's own denoisers (bundled MNIST CNN; DDPM FFHQ faces once
ffhq.pt is downloaded) through pcuq's jvp -> subspace iteration -> diagnostics —
the same code that produced the 3D results — for a like-for-like comparison.
Outputs sweep figures (the paper's signature visualization) and a viewer bundle.

    python scripts/run_images2d.py --model mnist
    python scripts/run_images2d.py --model ffhq --image <face.png> [--from-t 100]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pcuq.data import corrupt
from pcuq.diagnostics import antisym_energy_fd, sweep_step_size_fd
from pcuq.spectrum import top_eigenpairs
from pcuq.utils import set_seed
from pcuq.viz import plot_sweep_2d

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external/GaussianDenoisingPosterior"


def q(t, nd=4):
    return [round(float(v), nd) for v in t.reshape(-1)]


def run_one(den, name, x, mask, k, iters, c, out, viewer_runs, meta,
            method="autograd"):
    y = corrupt(x[None], den.sigma, seed=0)[0]
    with torch.no_grad():
        x_hat = den(y[None])[0]
    eigvecs, eigvals, history = top_eigenpairs(
        den, y, den.sigma, k=k, iters=iters, method=method, c=c, mask=mask)
    asym = antisym_energy_fd(den, y, eigvecs, method=method, c=c)
    print(f"[{name}] eigvals {eigvals.cpu().numpy()} | "
          f"eigval/σ² {(eigvals[0] / den.sigma**2).item():.2f} | antisym {asym:.3f} | "
          f"overlap {history[-1].numpy()}")
    torch.save({"x": x.cpu(), "y": y.cpu(), "x_hat": x_hat.cpu(),
                "eigvecs": eigvecs.cpu(), "eigvals": eigvals.cpu(),
                "mask": None if mask is None else mask.cpu()}, out / f"{name}.pt")
    plot_sweep_2d(x_hat.cpu(), eigvecs.cpu(), eigvals.cpu(), out / f"{name}_sweep.png")

    flat_mask = (torch.ones_like(x[0]) if mask is None else mask.to(x.dtype))
    mask_idx = flat_mask.reshape(-1).bool()
    viewer_runs.append({
        "tag": name, **meta,
        "c": x.shape[0], "h": x.shape[1], "w": x.shape[2],
        "sigma": round(den.sigma, 4),
        "eigvals": [float(v) for v in eigvals],
        "x_hat": q(x_hat.expand(x.shape[0], -1, -1) if x_hat.shape[0] != x.shape[0]
                   else x_hat, 3),
        "mask": [int(b) for b in mask_idx],
        "modes": [q(eigvecs[i].reshape(-1)[mask_idx], 4) for i in range(k)],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mnist", "ffhq"], default="mnist")
    parser.add_argument("--image", help="face image path (ffhq only)")
    parser.add_argument("--from-t", type=int, default=100, help="DDPM timestep")
    parser.add_argument("--digits", type=int, nargs="*", default=[2, 1, 18])
    parser.add_argument("--n-ev", type=int, default=4)
    parser.add_argument("--iters", type=int, default=25)
    # These are smooth CNNs (no graph rebuilds), so exact autograd JVPs work and
    # avoid float32 finite-difference noise, which power iteration otherwise
    # amplifies into spiky artifact modes.
    parser.add_argument("--method", choices=["autograd", "central", "bp"],
                        default="autograd")
    args = parser.parse_args()
    set_seed(0)
    device = torch.device("cpu")
    out = ROOT / f"outputs/images2d/{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    viewer_runs = []

    if args.model == "mnist":
        from torchvision.datasets import MNIST
        from pcuq.denoisers2d import MNISTDenoiser2D
        den = MNISTDenoiser2D(str(REPO), device)
        test = MNIST(root=str(ROOT / "data"), train=False, download=True)
        for i in args.digits:
            img, label = test[i]
            x = torch.from_numpy(__import__("numpy").array(img)).float()[None] / 255
            run_one(den, f"mnist{i}_digit{label}", x, None, args.n_ev, args.iters,
                    c=1e-3, out=out, viewer_runs=viewer_runs,
                    meta={"kind": "mnist", "label": int(label)}, method=args.method)
        sweep = sweep_step_size_fd(den, corrupt(x[None], den.sigma, 0)[0],
                                   [1e-2, 1e-3, 1e-4])
        print("step-size self-consistency:", {k: f"{v:.1e}" for k, v in sweep.items()})
    else:
        from PIL import Image
        import numpy as np
        from pcuq.denoisers2d import FFHQDenoiser2D
        den = FFHQDenoiser2D(str(REPO), device, from_t=args.from_t)
        img = Image.open(args.image).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1
        regions = {  # boxes in (y1, y2, x1, x2), tuned for aligned faces
            "eyes": (85, 135, 55, 200), "mouth": (160, 215, 80, 175),
        }
        for rname, (y1, y2, x1, x2) in regions.items():
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[:, y1:y2, x1:x2] = True
            run_one(den, f"face_{rname}_t{args.from_t}", x, mask, 3,
                    iters=8, c=1e-3, out=out, viewer_runs=viewer_runs,
                    meta={"kind": "ffhq", "label": rname}, method=args.method)

    bundle = ROOT / "results/viewer_data_2d.json"
    existing = json.loads(bundle.read_text())["runs"] if bundle.exists() else []
    new_tags = {r["tag"] for r in viewer_runs}
    existing = [r for r in existing if r["tag"] not in new_tags] + viewer_runs
    bundle.write_text(json.dumps({"runs": existing}, separators=(",", ":")))
    print(f"done — {len(viewer_runs)} runs, figures at {out}, "
          f"viewer bundle {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
