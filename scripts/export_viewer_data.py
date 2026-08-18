"""Export the 3D viewer bundle: every usable region run from the mask GRID.

Reads outputs/masked-grid/masked_modes (6 extremity regions x 2 sigmas per shape)
so the viewer can move the mask across the shape. Clouds are stored once per
(shape, sigma) and referenced by runs; coordinates quantized to 4 decimals.

    python scripts/export_viewer_data.py
"""

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/masked-grid/masked_modes"
MIN_OVERLAP = 0.9


def q(t, nd=4):
    return [round(float(v), nd) for v in t.reshape(-1)]


SOURCES = [
    # (dir under outputs/, method label, drop non-converged runs?)
    ("masked-grid", "frozen", True),
    # the naive baseline port is kept even when broken — being broken is the point
    ("masked-grid-unfrozen", "naive", False),
]


def main() -> None:
    clouds, runs, dropped = {}, [], 0
    for dirname, method, filter_conv in SOURCES:
        src = ROOT / "outputs" / dirname / "masked_modes"
        if not (src / "metrics.json").exists():
            print(f"({dirname} not found — skipping)")
            continue
        metrics = json.loads((src / "metrics.json").read_text())
        for tag, m in sorted(metrics.items()):
            conv_ok = min(m["final_iter_overlap"]) >= MIN_OVERLAP
            if filter_conv and not conv_ok:
                dropped += 1
                continue
            shape, rest = tag.rsplit("_sigma", 1)
            sigma, region = rest.split("_r")
            d = torch.load(src / f"{tag}.pt", weights_only=True)
            key = f"{shape}|{sigma}"
            if key not in clouds:
                clouds[key] = q(d["x_hat"])
            runs.append({
                "tag": f"{tag}[{method}]", "shape": shape, "sigma": float(sigma),
                "region": int(region), "cloud": key, "method": method,
                "spread": round(m["spread_top_to_last"], 3),
                "psd_ok": bool(min(m["eigvals"]) > 0), "conv_ok": conv_ok,
                "eigvals": [float(v) for v in d["eigvals"]],
                "mask_idx": d["mask"].nonzero().squeeze(1).tolist(),
                "modes": [q(d["eigvecs"][i][d["mask"]])
                          for i in range(len(d["eigvals"]))],
            })
    out = ROOT / "results/viewer_data.json"
    out.write_text(json.dumps({"clouds": clouds, "runs": runs},
                              separators=(",", ":")))
    print(f"{len(runs)} region runs ({dropped} dropped for convergence), "
          f"{len(clouds)} clouds -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
