"""Build the results/ folder: summary charts + exemplar figures + README.

Aggregates whatever exists under outputs/ (phase3 sweep, ablation-unfrozen,
masked modes) into a small curated, git-tracked results/ directory that shows
what the project did at a glance. Rerun any time; it refreshes in place:

    python scripts/build_results.py
"""

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"

# Validated categorical palette (light mode) — see dataviz palette reference.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a94"
INK, INK2 = "#333333", "#666666"
TRAIN_SIGMA_MAX = 0.034  # Noise2Score3D training range, models/KPconv.py:159


def load(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


def style(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.8")
    ax.grid(axis="y", color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


def by_sigma(metrics):
    out = {}
    for tag, m in metrics.items():
        sig = float(tag.split("sigma")[1].split("_")[0])
        out.setdefault(sig, []).append(m)
    return dict(sorted(out.items()))


def chart_calibration(phase3):
    """Top eigenvalue / sigma^2 per shape, across noise levels."""
    groups = by_sigma(phase3)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rng = np.random.default_rng(0)
    for i, (sig, runs) in enumerate(groups.items()):
        vals = np.array([m["eigvals"][0] / sig**2 for m in runs])
        shown = np.clip(vals, -2, 4)
        x = i + rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(x, shown, s=12, color=BLUE, alpha=0.45, linewidths=0)
        ax.hlines(np.median(vals), i - 0.22, i + 0.22, color=INK, lw=2)
        n_clip = int(((vals < -2) | (vals > 4)).sum())
        if n_clip:
            ax.annotate(f"{n_clip} shapes\nbeyond view", (i + 0.26, 3.6),
                        fontsize=7, color=INK2)
    ax.axhline(1.0, color=GRAY, lw=1.2, ls="--")
    ax.annotate("exact-MMSE bound (λ₀ = σ²)", (-0.45, 0.72), fontsize=8, color=INK2)
    labels = []
    for sig in groups:
        labels.append(f"σ={sig}" + ("\n(beyond training\nrange σ≤0.034)"
                                    if sig > TRAIN_SIGMA_MAX else ""))
    ax.set_xticks(range(len(groups)), labels)
    ax.set_ylabel("top eigenvalue / σ²", fontsize=9, color=INK)
    style(ax, "Calibration: top eigenvalue vs the σ² bound (dot = shape, bar = median)")
    fig.tight_layout()
    fig.savefig(FIG / "calibration_vs_sigma.png", dpi=170)
    plt.close(fig)
    return {sig: (float(np.median([m["eigvals"][0] / sig**2 for m in runs])),
                  sum(1 for m in runs for v in m["eigvals"] if v < 0),
                  float(np.median([min(m["final_iter_overlap"]) for m in runs])))
            for sig, runs in groups.items()}


def chart_ablation(frozen_runs, unfrozen_runs, sigma):
    """Frozen-graph vs graph-rebuilt finite differences, three metrics."""
    def med(runs, f):
        return float(np.median([f(m) for m in runs]))

    panels = [
        ("top eigenvalue / σ²", lambda m: m["eigvals"][0] / sigma**2),
        ("Jacobian asymmetry", lambda m: m["antisym_energy_subspace"]),
        ("convergence (min overlap)", lambda m: min(m["final_iter_overlap"])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for ax, (name, f) in zip(axes, panels):
        vals = [med(unfrozen_runs, f), med(frozen_runs, f)]
        bars = ax.bar(["graph\nrebuilt", "graph\nfrozen"], vals, width=0.55,
                      color=[ORANGE, BLUE])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}" if abs(v) >= 0.1 else f"{v:.3f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, color=INK)
        style(ax, name)
        ax.margins(y=0.18)
    fig.suptitle(f"Why the graph must be frozen (σ={sigma}, medians)",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "frozen_vs_rebuilt.png", dpi=170)
    plt.close(fig)


def chart_structure(phase3, masked):
    """Eigenvalue spread (top-3): whole shapes vs extremity regions."""
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    rng = np.random.default_rng(1)
    cols, ticks = [], []
    slot = 0
    for sig in (0.02, 0.03):
        whole = [(m["eigvals"][0] - m["eigvals"][2]) / m["eigvals"][0]
                 for m in by_sigma(phase3).get(sig, [])
                 if min(m["final_iter_overlap"]) > 0.9 and m["eigvals"][2] > 0]
        region = [m["spread_top_to_last"] for m in by_sigma(masked).get(sig, [])
                  if min(m["final_iter_overlap"]) > 0.9 and m["eigvals"][-1] > 0]
        for vals, color, name in ((whole, BLUE, "whole shape"),
                                  (region, AQUA, "extremity region")):
            if not vals:
                continue
            v = np.array(vals)
            x = slot + rng.uniform(-0.13, 0.13, len(v))
            ax.scatter(x, v, s=12, color=color, alpha=0.55, linewidths=0)
            ax.hlines(np.median(v), slot - 0.22, slot + 0.22, color=INK, lw=2)
            cols.append(slot)
            ticks.append(f"{name}\nσ={sig}")
            slot += 1
        slot += 0.4
    ax.set_xticks(cols, ticks)
    ax.set_ylabel("top-3 eigenvalue spread (λ₀−λ₂)/λ₀", fontsize=9, color=INK)
    style(ax, "Structure lives in regions: spectra are flat on whole shapes,\n"
              "anisotropic on extremity patches (converged, PSD runs only)")
    fig.tight_layout()
    fig.savefig(FIG / "whole_vs_region_spread.png", dpi=170)
    plt.close(fig)


def pick_exemplars(masked, n=6):
    """Best converged, high-spread, PSD region runs — one per (category, sigma)."""
    scored = []
    for tag, m in masked.items():
        if min(m["final_iter_overlap"]) >= 0.95 and m["eigvals"][-1] > 0:
            scored.append((m["spread_top_to_last"], tag))
    scored.sort(reverse=True)
    picked, seen = [], set()
    for spread, tag in scored:
        key = (tag.split("_")[0], tag.split("sigma")[1].split("_")[0])
        if key in seen:
            continue
        seen.add(key)
        picked.append((tag, spread))
        if len(picked) == n:
            break
    src = ROOT / "outputs/masked/masked_modes"
    for tag, _ in picked:
        for suffix in ("modes", "mode0_arrows", "mode0_sweep"):
            f = src / f"{tag}_{suffix}.png"
            if f.exists():
                shutil.copy(f, FIG / f.name)
    return picked


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    phase3 = load("outputs/phase3/run_experiment/metrics.json")
    masked = load("outputs/masked/masked_modes/metrics.json")
    ablation = load("outputs/ablation-unfrozen/run_experiment/metrics.json") \
        or load("outputs/local/run_experiment/metrics.json")  # early 10-shape grid
    if phase3 is None:
        sys.exit("no phase3 results found — run the sweep first (docs/WORKFLOW.md)")

    calib = chart_calibration(phase3)
    frozen02 = by_sigma(phase3)[0.02]
    unfrozen02 = by_sigma(ablation)[0.02] if ablation else []
    if unfrozen02:
        chart_ablation(frozen02, unfrozen02, 0.02)
    exemplars = pick_exemplars(masked) if masked else []
    if masked:
        chart_structure(phase3, masked)

    lines = [
        "# Results",
        "",
        "*(Generated by `python scripts/build_results.py` — rerun after new "
        "experiments to refresh. Full narrative: [docs/LOG.md](../docs/LOG.md).)*",
        "",
        "**What this project does:** estimates structured posterior uncertainty "
        "(top covariance eigenpairs) directly from the Jacobian of a frozen "
        "point-cloud denoiser (Noise2Score3D, ICCV 2025) — no retraining, no "
        "sampling — adapting Manor & Michaeli (ICLR 2024) from images to 3D "
        "point clouds (ModelNet40).",
        "",
        "## 1. The method is calibrated — until the denoiser leaves its training range",
        "",
        "![calibration](figures/calibration_vs_sigma.png)",
        "",
        "Exact MMSE theory bounds posterior eigenvalues by σ². Per σ "
        "(median top-eigval/σ² · negative eigvals · median convergence):",
        "",
        "| σ | λ₀/σ² | negative eigvals | convergence |",
        "|---|---|---|---|",
    ]
    for sig, (med, neg, conv) in calib.items():
        note = " ⚠ beyond training range" if sig > TRAIN_SIGMA_MAX else ""
        lines.append(f"| {sig}{note} | {med:.2f} | {neg} | {conv:.3f} |")
    lines += [
        "",
        "## 2. The 3D-specific obstacle: graph rebuilds, and the fix",
        "",
        "The KPConv denoiser rebuilds its voxel/neighbor graph every forward pass; "
        "finite differences across that rebuild measure O(1) discrete jumps, not "
        "the derivative. Freezing the anchor's graph (`freeze_graph`) "
        "differentiates the smooth branch:",
        "",
        "![ablation](figures/frozen_vs_rebuilt.png)",
        "",
        "## 3. Uncertainty structure lives in regions, not whole shapes",
        "",
        "Whole-shape posteriors are near-isotropic (flat spectra). Restricting the "
        "operator to extremity patches (the reference paper's patch-mask move, in "
        "3D) exposes real anisotropy:",
        "",
        "![structure](figures/whole_vs_region_spread.png)",
        "",
        "## 4. The modes themselves",
        "",
        "Best converged, PSD region runs (spread = top-to-last eigenvalue gap). "
        "Per exemplar: region modes, mode-0 direction arrows, mode-0 sweep "
        "x̂ ± t·√λ·v:",
        "",
    ]
    for tag, spread in exemplars:
        lines += [
            f"### {tag}  (spread {spread:.0%})",
            "",
            f"![{tag} modes](figures/{tag}_modes.png)",
            f"![{tag} arrows](figures/{tag}_mode0_arrows.png)",
            f"![{tag} sweep](figures/{tag}_mode0_sweep.png)",
            "",
        ]
    (OUT / "README.md").write_text("\n".join(lines))
    print(f"results/ built: {len(list(FIG.glob('*.png')))} figures, "
          f"README with {len(exemplars)} exemplars")


if __name__ == "__main__":
    main()
