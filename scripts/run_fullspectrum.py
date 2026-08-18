"""Whole-shape rank-k spectrum per shape — the basis for free-form masks in the
viewer: for ANY point mask M, the masked covariance is (MV)Λ(MV)^T, so its
eigenmodes reduce to a k x k problem solvable client-side. Exports int16-quantized
modes to results/viewer_data_free.json.

    python scripts/run_fullspectrum.py
"""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from pcuq.data import corrupt, load_modelnet
from pcuq.denoisers import Noise2Score3DWrapper
from pcuq.spectrum import top_eigenpairs
from pcuq.utils import set_seed

ROOT = Path(__file__).resolve().parents[1]
K, SIGMA, ITERS = 24, 0.03, 15


def b64_int16(t: torch.Tensor):
    a = t.detach().cpu().numpy().astype(np.float32).reshape(-1)
    scale = float(np.abs(a).max()) or 1.0
    enc = base64.b64encode((a / scale * 32767).astype("<i2").tobytes()).decode()
    return enc, scale


def main() -> None:
    set_seed(0)
    cfg = {"seed": 0, "data": {"root": "data", "n_shapes": 3, "n_points": 2048,
                               "categories": ["chair", "lamp", "airplane"]}}
    shapes = load_modelnet(cfg, torch.float32)
    den = Noise2Score3DWrapper("external/Noise2Score3D",
                               "data/checkpoints/noise2score3d_step4500.pth",
                               sigma=SIGMA, device=torch.device("cpu"))
    out = ROOT / "outputs/fullspectrum"
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for si, (name, x) in enumerate(shapes):
        y = corrupt(x[None], SIGMA, seed=si)[0]
        with den.graph_frozen():
            with torch.no_grad():
                x_hat = den(y[None])[0]
            eigvecs, eigvals, history = top_eigenpairs(
                den, y, SIGMA, k=K, iters=ITERS, method="central", c=1e-3)
        print(f"[{name}] λ range [{eigvals.min():.2e}, {eigvals.max():.2e}] "
              f"| min overlap {history[-1].min():.3f}")
        torch.save({"x_hat": x_hat, "eigvecs": eigvecs, "eigvals": eigvals},
                   out / f"{name}.pt")
        modes_b64, scale = b64_int16(eigvecs)
        entries.append({
            "shape": name, "sigma": SIGMA, "k": K,
            "eigvals": [float(v) for v in eigvals],
            "x_hat": [round(float(v), 4) for v in x_hat.reshape(-1)],
            "modes_b64": modes_b64, "modes_scale": scale,
        })
    dst = ROOT / "results/viewer_data_free.json"
    dst.write_text(json.dumps({"free": entries}, separators=(",", ":")))
    print(f"done -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
