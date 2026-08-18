"""Export a compact JSON bundle of masked-mode results for the interactive viewer.

Picks the best converged, PSD region runs (like build_results.pick_exemplars but
broader), quantizes coordinates to 4 decimals, and writes results/viewer_data.json
consumed by the viewer page. Rerun after new masked runs:

    python scripts/export_viewer_data.py
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/masked/masked_modes"
MAX_RUNS = 16


def q(t, nd=4):
    return [round(float(v), nd) for v in t.reshape(-1)]


def main() -> None:
    metrics = json.loads((SRC / "metrics.json").read_text())
    scored = sorted(
        ((m["spread_top_to_last"], tag) for tag, m in metrics.items()
         if min(m["final_iter_overlap"]) >= 0.95 and m["eigvals"][-1] > 0),
        reverse=True)

    runs, seen = [], set()
    for spread, tag in scored:                    # diverse: one per (shape, sigma)
        shape, rest = tag.rsplit("_sigma", 1)
        sigma, region = rest.split("_r")
        if (shape, sigma) in seen:
            continue
        seen.add((shape, sigma))
        d = torch.load(SRC / f"{tag}.pt", weights_only=True)
        runs.append({
            "tag": tag, "shape": shape, "sigma": float(sigma), "region": int(region),
            "spread": round(spread, 3),
            "eigvals": [float(v) for v in d["eigvals"]],
            "x_hat": q(d["x_hat"]),
            "mask": [int(b) for b in d["mask"]],
            # eigvecs stored masked-points-only to keep the bundle small
            "modes": [q(d["eigvecs"][i][d["mask"]]) for i in range(len(d["eigvals"]))],
        })
        if len(runs) == MAX_RUNS:
            break

    out = ROOT / "results/viewer_data.json"
    out.write_text(json.dumps({"runs": runs}, separators=(",", ":")))
    print(f"{len(runs)} runs -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if not runs:
        sys.exit("no eligible runs found")


if __name__ == "__main__":
    main()
