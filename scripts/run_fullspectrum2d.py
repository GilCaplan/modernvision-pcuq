"""Whole-image rank-k spectra for the viewer's draggable 2D masks.

Same construction as the 3D free-mask feature: precompute top-k eigenpairs of the
full-image posterior covariance once per image; any mask's covariance is then
(MV)Λ(MV)ᵀ — a k x k problem solved client-side. Faces use the DDPM at t=400
(reverse-mode products); MNIST uses exact autograd JVPs. Modes are stored at half
resolution for faces (display upsamples), int16-quantized; the base image ships
as a PNG data URI.

    python scripts/run_fullspectrum2d.py
"""

import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pcuq.data import corrupt
from pcuq.spectrum import top_eigenpairs
from pcuq.utils import set_seed

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external/GaussianDenoisingPosterior"


def b64_int16(a: np.ndarray):
    scale = float(np.abs(a).max()) or 1.0
    return base64.b64encode((a / scale * 32767).astype("<i2").tobytes()).decode(), scale


def png_uri(img01: torch.Tensor) -> str:  # (C,H,W) in [0,1]
    a = (img01.clamp(0, 1) * 255).byte().cpu().numpy()
    im = Image.fromarray(a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0),
                         "L" if a.shape[0] == 1 else "RGB")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def spectrum_entry(den, x, name, kind, label, k, iters, method, pool):
    y = corrupt(x[None], den.sigma, seed=0)[0]
    with torch.no_grad():
        x_hat = den(y[None])[0]
    eigvecs, eigvals, history = top_eigenpairs(
        den, y, den.sigma, k=k, iters=iters, method=method, c=1e-3)
    print(f"[{name}] λ0={eigvals[0]:.3g} ({(eigvals[0]/den.sigma**2):.2f}σ²) "
          f"| min overlap {history[-1].min():.3f}")
    torch.save({"x_hat": x_hat, "eigvecs": eigvecs, "eigvals": eigvals},
               ROOT / f"outputs/fullspectrum2d/{name}.pt")
    v = eigvecs
    if pool > 1:  # store modes at reduced res; renormalize each
        v = F.avg_pool2d(v, pool)
        v = v / v.reshape(k, -1).norm(dim=1).reshape(k, 1, 1, 1)
    disp = (x_hat + 1) / 2 if x_hat.min() < -0.01 else x_hat
    modes_b64, scale = b64_int16(v.numpy())
    return {"id": name, "kind": kind, "label": label,
            "w": x.shape[2], "h": x.shape[1], "c": x.shape[0],
            "mw": v.shape[3], "mh": v.shape[2],
            "sigma": round(float(den.sigma), 4), "k": k,
            "eigvals": [float(e) for e in eigvals],
            "png": png_uri(disp), "modes_b64": modes_b64, "modes_scale": scale}


def write_merged(entries):
    dst = ROOT / "results/viewer_data_2d.json"
    existing = json.loads(dst.read_text())["images"] if dst.exists() else []
    if not isinstance(existing, list):
        existing = []
    by_id = {e["id"]: e for e in existing}
    for e in entries:
        by_id[e["id"]] = e
    merged = sorted(by_id.values(), key=lambda e: (e["kind"] != "mnist", e["id"]))
    dst.write_text(json.dumps({"images": merged}, separators=(",", ":")))
    return dst, len(merged)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, nargs="*", default=[2, 1, 18],
                        help="MNIST test-set indices")
    parser.add_argument("--faces", nargs="*", default=["213", "227", "34"],
                        help="face ids under docs/resources/faces-<id>/")
    parser.add_argument("--replace", action="store_true",
                        help="replace the whole bundle (default: merge by id)")
    args = parser.parse_args()
    set_seed(0)
    (ROOT / "outputs/fullspectrum2d").mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    if args.replace:
        (ROOT / "results/viewer_data_2d.json").unlink(missing_ok=True)
    entries = []

    if args.digits:
        from torchvision.datasets import MNIST
        from pcuq.denoisers2d import MNISTDenoiser2D
        den = MNISTDenoiser2D(str(REPO), device)
        test = MNIST(root=str(ROOT / "data"), train=False, download=True)
        for i in args.digits:
            img, label = test[i]
            x = torch.from_numpy(np.array(img)).float()[None] / 255
            entries.append(spectrum_entry(den, x, f"mnist{i}", "mnist",
                                          f"digit {label}", k=12, iters=40,
                                          method="autograd", pool=1))
        write_merged(entries)

    if args.faces:
        from pcuq.denoisers2d import FFHQDenoiser2D
        den = FFHQDenoiser2D(str(REPO), device, from_t=400)
        for fid in args.faces:
            img = Image.open(REPO / f"docs/resources/faces-{fid}/ev1_0.00.png") \
                .convert("RGB").resize((256, 256))
            x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1
            entries.append(spectrum_entry(den, x, f"face{fid}", "ffhq",
                                          f"face {fid}", k=10, iters=6,
                                          method="bp", pool=2))
            write_merged(entries)  # incremental — each face takes ~13 min

    dst, n = write_merged(entries)
    print(f"done — bundle now has {n} images -> {dst} ({dst.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
