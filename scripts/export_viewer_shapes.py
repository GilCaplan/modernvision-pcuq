"""Export whole-shape and smooth modes of every run_experiment shape for the explorer.

The explorer's region runs cover 5 shapes; this adds all shapes of a run_experiment
run with smooth modes enabled (default outputs/smooth/run_experiment/, 50 shapes x 3
sigmas). Writes a small index (embedded in the page) plus one data file per sigma
that the page fetches when that sigma is opened:

    outputs/viewer_shapes/index.json
    outputs/viewer_shapes/shapes50_<sigma>.json

Clouds are int16 and modes int8 (per-mode scale), base64 — display precision only.
Rejected runs (symmetry gate) are listed in the index without modes.

    python scripts/export_viewer_shapes.py [--src outputs/smooth/run_experiment]
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def q8(v: np.ndarray) -> dict:
    s = float(np.abs(v).max()) or 1.0
    return {"scale": s, "b64": b64(np.round(v / s * 127).astype(np.int8))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="outputs/smooth/run_experiment")
    parser.add_argument("--out", default="outputs/viewer_shapes")
    args = parser.parse_args()
    src, out = ROOT / args.src, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((src / "metrics.json").read_text())

    index, files = [], {}
    for tag, m in sorted(metrics.items()):
        if not isinstance(m, dict) or "_sigma" not in tag:
            continue
        shape, sigma = tag.rsplit("_sigma", 1)
        sigma = float(sigma)
        d = torch.load(src / f"{tag}.pt", map_location="cpu", weights_only=False)
        x = d["x_hat"].numpy().astype(np.float64)
        s = float(np.abs(x).max())
        data = files.setdefault(sigma, {"clouds": {}, "modes": {}})
        key = f"{shape}|{sigma}"
        data["clouds"][key] = {"n": len(x), "scale": s,
                               "b64": b64(np.round(x / s * 32767).astype("<i2"))}
        # whole shape (top_eigenpairs); a rejected run has no eigenpairs
        entry = {"shape": shape, "sigma": sigma, "cloud": key, "n": len(x)}
        if "eigvecs" in d:
            ev = d["eigvals"].tolist()
            index.append({**entry, "region": "whole", "eigvals": ev,
                          "psd_ok": min(ev) > 0,
                          "conv_ok": min(m["final_iter_overlap"]) >= 0.9,
                          "trustworthy": bool(m.get("trustworthy"))})
            data["modes"][f"{key}|whole"] = [q8(v.numpy().reshape(-1)) for v in d["eigvecs"]]
        else:
            index.append({**entry, "region": "whole", "rejected": m.get("top_eigenpairs_rejected", "")})
        sm = m.get("smooth")
        if sm is None:
            continue
        if "smooth_rejected" in sm:
            index.append({**entry, "region": "smooth", "rejected": sm["smooth_rejected"]})
            continue
        ds = torch.load(src / f"{tag}_smooth.pt", map_location="cpu", weights_only=True)
        ev = ds["eigvals"].tolist()
        index.append({**entry, "region": "smooth", "eigvals": ev, "psd_ok": min(ev) > 0,
                      "conv_ok": True, "trustworthy": True,
                      "basis_dimension": sm["basis_dimension"]})
        data["modes"][f"{key}|smooth"] = [q8(v.numpy().reshape(-1)) for v in ds["eigvecs"]]

    (out / "index.json").write_text(json.dumps(index))
    for sigma, data in files.items():
        p = out / f"shapes50_{sigma}.json"
        p.write_text(json.dumps(data))
        print(f"{p.name}: {len(data['clouds'])} clouds, {len(data['modes'])} mode sets, "
              f"{p.stat().st_size / 1e6:.1f} MB")
    ok = sum(1 for e in index if "eigvals" in e)
    print(f"index.json: {len(index)} runs ({ok} with modes, {len(index) - ok} rejected)")


if __name__ == "__main__":
    main()
