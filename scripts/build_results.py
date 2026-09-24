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
sys.path.insert(0, str(ROOT / "src"))  # for the depth-projection example only

# Validated categorical palette (light mode) — see dataviz palette reference.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a94"
INK, INK2 = "#333333", "#666666"
TRAIN_SIGMA_MAX = 0.034  # Noise2Score3D training range, models/KPconv.py:159

# Interpretation of the smooth-mode run (docs/LOG.md 2026-09-24); kept next to the
# numbers it describes so a rerun with different results gets re-read, not reused.
SMOOTH_FINDING = (
    "**Along smooth deformations the denoiser behaves almost like the identity.** "
    "In range, the smooth eigenvalues sit at σ² (λ₀ ≈ 1.01–1.03σ², and even the "
    "smallest of the ~84 smooth-subspace eigenvalues is ~0.76–0.90σ²), with a top-5 "
    "spread of about 1%, flatter than the whole-shape spectrum. Noise2Score3D does "
    "not shrink smooth, non-rigid displacements, so its uncertainty over them is the "
    "full noise level and nearly the same for every smooth deformation: no single "
    "bend or stretch stands out. Rejections are not the same shapes as the "
    "whole-shape gate's: some shapes the baseline rejects pass on the smooth "
    "subspace and vice versa. At σ=0.05 every smooth run is rejected (median "
    "asymmetry 0.13), consistent with the out-of-range breakdown in §1.")
SMOOTH_STABILITY = (
    "Eigenvalues reproduce exactly. Individual smooth modes do not (top-mode "
    "overlap ranges from 0.04 to 0.89 across shapes), but about three directions "
    "of the top-5 subspace do: 2–4 principal angles per shape fall below the "
    "chance minimum. That is better than the whole-shape modes (§3: top-mode "
    "overlap 0.04, max angle 89°), yet still not a per-mode property: read the "
    "figures below as one noisy observation.")


def load(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


def is_current(metrics) -> bool:
    """True if a metrics.json comes from the 2026-09-16+ pipeline (Rayleigh-Ritz,
    symmetry gate): only those runs record covariance_kind or a rejection. A
    machine can still hold older runs in outputs/phase3 or outputs/masked; those
    must never silently replace current numbers in the report."""
    return bool(metrics) and any(
        isinstance(m, dict) and ("covariance_kind" in m or "top_eigenpairs_rejected" in m)
        for m in metrics.values())


def first_current(*paths):
    """(metrics, path) of the first current-pipeline source, else (None, None)."""
    for path in paths:
        metrics = load(path)
        if is_current(metrics):
            return metrics, path
    return None, None


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
        if tag.startswith("_") or "sigma" not in tag:
            continue  # e.g. "_provenance" -- a run-level key, not a per-shape tag
        sig = float(tag.split("sigma")[1].split("_")[0])
        out.setdefault(sig, []).append(m)
    return dict(sorted(out.items()))


def _trustworthy(m: dict, min_convergence: float = 0.9,
                 max_ritz_residual: float = 0.15) -> bool:
    """Mirrors pcuq.diagnostics.is_trustworthy (task 9) -- duplicated rather than
    imported so this numpy/matplotlib-only script doesn't need to import torch.
    Keep thresholds in sync with that function if they ever change."""
    if "trustworthy" in m:
        return m["trustworthy"]
    if "ritz_relative_residuals" not in m:
        return False  # older metrics.json predates this diagnostic
    return (min(m["final_iter_overlap"]) >= min_convergence
           and max(m["ritz_relative_residuals"]) <= max_ritz_residual)


def chart_calibration(phase3):
    """Top eigenvalue / sigma^2 per shape, across noise levels.

    Noise2Score3D is an unconditioned score network with sigma manually
    substituted into Tweedie's formula (covariance_kind =
    frozen_pyramid_sensitivity, see src/pcuq/denoisers.py) -- there is no
    per-sigma verification that it is the MMSE denoiser AT that sigma, so this
    is sigma^2*J behaving like a sensitivity operator across noise levels, not
    a calibration sweep of a proven fixed-sigma estimator (docs/LOG.md
    2026-09-16). A run top_eigenpairs rejects outright (too asymmetric to call
    covariance eigenpairs at all) is excluded from the stats below and counted
    separately -- silently including those, as the pre-2026-09-16 pipeline did,
    overstates how uniformly trustworthy any of this is.
    """
    groups = by_sigma(phase3)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rng = np.random.default_rng(0)
    ok_groups = {sig: [m for m in runs if "top_eigenpairs_rejected" not in m]
                for sig, runs in groups.items()}
    for i, (sig, runs) in enumerate(ok_groups.items()):
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
    ax.annotate("sigma^2 bound if this were exact MMSE (lambda0 = sigma^2)",
               (-0.45, 0.72), fontsize=8, color=INK2)
    labels = []
    for sig in groups:
        labels.append(f"σ={sig}" + ("\n(beyond training\nrange σ≤0.034)"
                                    if sig > TRAIN_SIGMA_MAX else ""))
    ax.set_xticks(range(len(groups)), labels)
    ax.set_ylabel("top eigenvalue / σ²", fontsize=9, color=INK)
    style(ax, "Sensitivity vs noise level: top eigenvalue / σ² (dot = shape, "
              "bar = median; rejected shapes excluded, see table)")
    fig.tight_layout()
    fig.savefig(FIG / "calibration_vs_sigma.png", dpi=170)
    plt.close(fig)
    return {sig: (float(np.median([m["eigvals"][0] / sig**2 for m in runs])) if runs else None,
                  sum(1 for m in runs for v in m["eigvals"] if v < 0),
                  float(np.median([min(m["final_iter_overlap"]) for m in runs])) if runs else None,
                  len(groups[sig]) - len(runs), len(groups[sig]),
                  sum(1 for m in groups[sig] if _trustworthy(m)))
            for sig, runs in ok_groups.items()}


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
                 if "top_eigenpairs_rejected" not in m
                 and min(m["final_iter_overlap"]) > 0.9 and m["eigvals"][2] > 0]
        region = [m["spread_top_to_last"] for m in by_sigma(masked).get(sig, [])
                  if "top_eigenpairs_rejected" not in m
                  and min(m["final_iter_overlap"]) > 0.9 and m["eigvals"][-1] > 0]
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


def render_depth_examples(resolution_lo=28, resolution_hi=256):
    """A few ModelNet40 shapes rendered as depth-map images (scripts/run_depth2d.py's
    side quest), at both the MNIST CNN's native input resolution (28 -- tiny and
    blocky, what that model actually receives) and a higher scale (256, FFHQ DDPM's
    native resolution) for comparison. Pure geometry (pcuq.depth.render_depth_map),
    no denoiser/checkpoint needed. Skipped (returns None) if ModelNet40.zip isn't
    already cached locally -- this script never triggers that ~2GB download itself.
    """
    if not (ROOT / "data/ModelNet40.zip").exists():
        return None
    import torch
    from pcuq.data import load_modelnet
    from pcuq.depth import render_depth_map

    cfg = {"data": {"root": str(ROOT / "data"), "n_shapes": 3, "n_points": 2048,
                    "categories": ["chair", "table", "guitar"]}, "seed": 0}
    shapes = load_modelnet(cfg, torch.float32)

    fig, axes = plt.subplots(len(shapes), 2, figsize=(4.4, 2.2 * len(shapes)))
    for row, (name, pts) in enumerate(shapes):
        for col, res in enumerate((resolution_lo, resolution_hi)):
            img = render_depth_map(pts, res, axis=2, dilate=1).numpy()
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                label = f"{res}x{res}" + (" (MNIST CNN input)" if res == resolution_lo
                                          else " (FFHQ DDPM scale)")
                ax.set_title(label, fontsize=8, color=INK)
            if col == 0:
                ax.set_ylabel(name, fontsize=8, color=INK)
    fig.suptitle("Depth-map projection of a 3D point cloud (axis=2, dilate=1) --\n"
                "the image actually fed to the reference paper's own 2D denoisers",
                fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "depth_projection_examples.png", dpi=170)
    plt.close(fig)
    return [name for name, _ in shapes]


def pick_exemplars(masked, masked_src, n=6):
    """Best converged, high-spread, PSD, TRUSTWORTHY region runs — one per
    (category, sigma). `trustworthy` (task 9, pcuq.diagnostics.is_trustworthy) is
    required in addition to the original convergence/PSD bar: it also catches the
    symmetry-gate/Ritz-residual failure modes those two checks alone miss (see
    docs/LOG.md 2026-09-16) -- exemplars predating that flag were never checked
    against it."""
    scored = []
    for tag, m in masked.items():
        if (m.get("trustworthy", False) and min(m["final_iter_overlap"]) >= 0.95
                and m["eigvals"][-1] > 0):
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
    for tag, _ in picked:
        for suffix in ("modes", "mode0_arrows", "mode0_sweep"):
            f = masked_src / f"{tag}_{suffix}.png"
            if f.exists():
                shutil.copy(f, FIG / f.name)
    return picked


def _spread(eigvals):
    """Top-to-last gap, meaningful only for a positive spectrum (None otherwise)."""
    return (eigvals[0] - eigvals[-1]) / eigvals[0] if min(eigvals) > 0 else None


def _median_spread(runs):
    vals = [v for v in (_spread(m["eigvals"]) for m in runs) if v is not None]
    return float(np.median(vals)) if vals else None


def chart_smooth(smooth):
    """Smooth, non-rigid modes (spectrum.smooth_eigenpairs) next to the whole-shape
    baseline of the SAME runs: top eigenvalue / sigma^2 and top-k spread."""
    groups = by_sigma(smooth)
    rows = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    rng = np.random.default_rng(2)
    ticks, cols = [], []
    for i, (sig, runs) in enumerate(groups.items()):
        whole = [m for m in runs if "top_eigenpairs_rejected" not in m]
        sm_all = [m["smooth"] for m in runs if "smooth" in m]
        sm = [x for x in sm_all if "smooth_rejected" not in x]
        rows[sig] = {
            "n": len(sm_all), "rejected": len(sm_all) - len(sm),
            "whole_rejected_smooth_ok": sum(1 for m in runs if "top_eigenpairs_rejected" in m
                                            and "smooth_rejected" not in m.get("smooth", {"smooth_rejected": 1})),
            "lam0": float(np.median([x["eigvals"][0] / sig**2 for x in sm])) if sm else None,
            "lam0_whole": float(np.median([m["eigvals"][0] / sig**2 for m in whole])) if whole else None,
            "spread": _median_spread(sm),
            "spread_whole": _median_spread(whole),
            "with_negative": sum(1 for x in sm if x["negative_eigenvalues"] > 0),
            "asymmetry": float(np.median([x["asymmetry_relative"] for x in sm])) if sm else None,
        }
        for ax, f in ((axes[0], lambda e: e[0] / sig**2), (axes[1], _spread)):
            for j, (vals, color) in enumerate(((whole, BLUE), (sm, ORANGE))):
                v = np.array([y for y in (f(m["eigvals"]) for m in vals) if y is not None])
                if not len(v):
                    continue
                x = 2.6 * i + j + rng.uniform(-0.13, 0.13, len(v))
                ax.scatter(x, np.clip(v, -2, 4), s=10, color=color, alpha=0.5, linewidths=0)
                ax.hlines(np.median(v), 2.6 * i + j - 0.22, 2.6 * i + j + 0.22, color=INK, lw=2)
        cols.append(2.6 * i + 0.5)
        ticks.append(f"σ={sig}" + ("\n(beyond\ntraining range)" if sig > TRAIN_SIGMA_MAX else ""))
    axes[0].axhline(1.0, color=GRAY, lw=1.2, ls="--")
    for ax, title in ((axes[0], "top eigenvalue / σ²"),
                      (axes[1], "top-k spread (λ₀−λₖ₋₁)/λ₀, positive spectra")):
        ax.set_xticks(cols, ticks)
        style(ax, title)
    axes[1].scatter([], [], color=BLUE, label="whole shape (baseline)")
    axes[1].scatter([], [], color=ORANGE, label="smooth, rigid motion removed")
    axes[1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Smooth deformation modes vs whole-shape modes (same runs; "
                 "rejected runs excluded)", fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "smooth_vs_whole.png", dpi=170)
    plt.close(fig)
    return rows


def pick_smooth_exemplars(smooth, src, n=3, sigma=0.02):
    """Trustworthy, all-positive smooth runs at sigma, largest spread, one per
    category. Falls back to smooth figures already committed in results/figures."""
    scored = sorted(((_spread(m["smooth"]["eigvals"]), tag) for tag, m in smooth.items()
                     if tag.endswith(f"sigma{sigma}") and m.get("smooth", {}).get("trustworthy")
                     and min(m["smooth"]["eigvals"]) > 0), reverse=True)
    picked, seen = [], set()
    for spread, tag in scored:
        if tag.split("_")[0] in seen:
            continue
        files = [src / f"{tag}_smooth_{s}.png" for s in ("modes", "mode0_arrows", "mode0_sweep")]
        if not all(f.exists() for f in files):
            continue
        seen.add(tag.split("_")[0])
        for f in files:
            shutil.copy(f, FIG / f.name)
        picked.append((tag, spread))
        if len(picked) == n:
            break
    if not picked:
        picked = [(f.name[:-len("_smooth_modes.png")], None)
                  for f in sorted(FIG.glob("*_smooth_modes.png"))]
    return picked


def smooth_section(rows, audit, exemplars):
    """README section 5: smooth, non-rigid deformation modes (docs/SMOOTH_MODES.md)."""
    fmt = lambda v, f: f.format(v) if v is not None else "—"
    lines = [
        "## 5. Smooth deformation modes: rigid motion removed",
        "",
        "`spectrum.smooth_eigenpairs` (docs/SMOOTH_MODES.md) restricts σ²·J to smooth "
        "displacement fields: the lowest graph-Laplacian frequencies on x̂, with global "
        "translations and infinitesimal rotations projected out (~84 directions). The "
        "reduced matrix is solved exactly and passes the same symmetry gate as the "
        "whole-shape spectrum. Same 50 shapes and noise draws as §1, whole-shape "
        "baseline recomputed alongside:",
        "",
        "![smooth vs whole](figures/smooth_vs_whole.png)",
        "",
        "| σ | smooth λ₀/σ² | smooth top-5 spread | whole-shape λ₀/σ² | whole-shape top-5 spread* "
        "| smooth rejected | runs with a negative smooth eigenvalue |",
        "|---|---|---|---|---|---|---|",
    ]
    for sig, r in rows.items():
        note = " ⚠" if sig > TRAIN_SIGMA_MAX else ""
        lines.append(
            f"| {sig}{note} | {fmt(r['lam0'], '{:.3f}')} | {fmt(r['spread'], '{:.1%}')} | "
            f"{fmt(r['lam0_whole'], '{:.2f}')} | {fmt(r['spread_whole'], '{:.1%}')} | "
            f"{r['rejected']}/{r['n']} | {r['with_negative']}/{r['n'] - r['rejected']} |")
    lines += ["", "*Spreads use only runs whose top-5 eigenvalues are all positive. "
              "The whole-shape columns come from this re-run, not §1: in range they "
              "match §1 closely; at σ=0.05 they differ (§1: −0.53σ², 13/50 rejected), "
              "that regime being ill-conditioned enough for machine-level "
              "floating-point differences to show.", "", SMOOTH_FINDING, ""]
    if audit and audit.get("summary", {}).get("median_max_subspace_angle_degrees") is not None:
        a = audit["summary"]
        lines += [
            f"**Direction stability (new noise seed, same points, {a['n']} trustworthy "
            f"σ=0.02 shapes, `scripts/audit_smooth_seed_stability.py`):** top eigenvalue "
            f"ratio {a['median_top_eigval_ratio']:.3f}; top-mode overlap "
            f"{a['median_top_mode_overlap']:.2f}; principal angles between the two top-5 "
            f"subspaces span {a['median_min_subspace_angle_degrees']:.0f}°–"
            f"{a['median_max_subspace_angle_degrees']:.0f}° (medians), against "
            f"{a['chance_median_min_angle_degrees']:.0f}°–"
            f"{a['chance_median_max_angle_degrees']:.0f}° for two random 5-dimensional "
            "subspaces of the same smooth basis. " + SMOOTH_STABILITY,
            "",
        ]
    if exemplars:
        lines += ["Example smooth modes at σ=0.02 (mode magnitudes, mode-0 arrows, "
                  "mode-0 sweep x̂ ± t·√λ·v), each for one noisy observation:", ""]
        for tag, _ in exemplars:
            lines += [f"### {tag} (smooth)", "",
                      f"![{tag} smooth modes](figures/{tag}_smooth_modes.png)",
                      f"![{tag} smooth arrows](figures/{tag}_smooth_mode0_arrows.png)",
                      f"![{tag} smooth sweep](figures/{tag}_smooth_mode0_sweep.png)", ""]
    return lines


def _load_unfrozen_ablation(path: str):
    """outputs/<x>/run_experiment/metrics.json is a generic smoke-test scratch
    location -- it can (and, mid-2026-09, did) hold whatever was last run there:
    toy-analytic data, or real-model data with freezing ON. covariance_kind alone
    can't tell frozen from unfrozen real-model runs apart (both are
    frozen_pyramid_sensitivity), so this only trusts a directory as genuinely
    "the unfrozen ablation" if its sibling config.json (written by make_out_dir)
    confirms freeze_graph was actually off."""
    metrics = load(path)
    cfg = load(str(Path(path).parent / "config.json"))
    if metrics is None or cfg is None or cfg.get("denoiser", {}).get("freeze_graph", True):
        return None
    return metrics


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    # Current-pipeline sources only; results/metrics/raw/ is the durable fallback
    # when outputs/ has been cleared (archive_metrics.py).
    phase3, _ = first_current("outputs/gpu/run_experiment/metrics.json",
                              "results/metrics/raw/gpu__run_experiment.json",
                              "outputs/phase3/run_experiment/metrics.json")
    masked, masked_path = first_current("outputs/gpu/masked_modes/metrics.json",
                                        "outputs/masked/masked_modes/metrics.json")
    masked_chart = masked or first_current("results/metrics/raw/gpu__masked_modes.json")[0]
    ablation = _load_unfrozen_ablation("outputs/ablation-unfrozen/run_experiment/metrics.json")
    if not is_current(ablation):
        ablation = None  # a pre-fix ablation would mix pipelines; keep the committed chart
    smooth, smooth_path = first_current("outputs/smooth/run_experiment/metrics.json",
                                        "results/metrics/raw/smooth__run_experiment.json")
    smooth_audit = (load("outputs/smooth/audit_smooth_seed_stability/metrics.json")
                    or load("results/metrics/raw/smooth__audit_smooth_seed_stability.json"))
    if phase3 is None:
        sys.exit("no current-pipeline sweep found — run the sweep first (docs/WORKFLOW.md)")

    calib = chart_calibration(phase3)
    frozen02 = [m for m in by_sigma(phase3)[0.02] if "top_eigenpairs_rejected" not in m]
    unfrozen02 = [m for m in (by_sigma(ablation).get(0.02, []) if ablation else [])
                 if "top_eigenpairs_rejected" not in m]
    if unfrozen02 and frozen02:
        chart_ablation(frozen02, unfrozen02, 0.02)
    exemplars = pick_exemplars(masked, (ROOT / masked_path).parent) if masked else []
    if masked_chart:
        chart_structure(phase3, masked_chart)
    smooth_rows = chart_smooth(smooth) if smooth and any("smooth" in m for m in smooth.values()) else None
    smooth_ex = (pick_smooth_exemplars(smooth, (ROOT / smooth_path).parent)
                 if smooth_rows else [])
    depth_example_shapes = render_depth_examples()

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
        "## 1. Sensitivity vs noise level — in-range agreement with the σ² bound, with caveats",
        "",
        "![calibration](figures/calibration_vs_sigma.png)",
        "",
        "**Not a calibration sweep.** Noise2Score3D is an unconditioned score "
        "network with σ manually substituted into Tweedie's formula at query "
        "time — there is no per-σ verification that it is *the* MMSE denoiser "
        "at that σ (`covariance_kind = frozen_pyramid_sensitivity`, see "
        "`src/pcuq/denoisers.py`). σ²·J is a local sensitivity operator whose "
        "behavior we track across noise levels, not a proven posterior "
        "covariance being calibrated. Exact MMSE theory would bound its "
        "eigenvalues by σ²; how closely this frozen network's sensitivity "
        "tracks that bound in-range, and how it degrades out of range, is the "
        "actual result. Per σ (median top-eigval/σ² · negative eigvals · "
        "median convergence, over runs the symmetry gate accepted):",
        "",
        "| σ | λ₀/σ² | negative eigvals | convergence | rejected by symmetry gate | trustworthy |",
        "|---|---|---|---|---|---|",
    ]
    for sig, (med, neg, conv, n_rej, n_total, n_trust) in calib.items():
        note = " ⚠ beyond training range" if sig > TRAIN_SIGMA_MAX else ""
        med_s = f"{med:.2f}" if med is not None else "—"
        conv_s = f"{conv:.3f}" if conv is not None else "—"
        lines.append(f"| {sig}{note} | {med_s} | {neg} | {conv_s} | "
                     f"{n_rej}/{n_total} ({n_rej/n_total:.0%}) | "
                     f"{n_trust}/{n_total} ({n_trust/n_total:.0%}) |")
    lines += [
        "",
        "**`trustworthy`** (not rejected AND convergence ≥0.9 AND max Ritz "
        "residual ≤0.15, `pcuq.diagnostics.is_trustworthy`) is the number to "
        "quote — at σ=0.05 it's much lower than \"not rejected\" alone would "
        "suggest, because most accepted runs there are still poorly converged. "
        "A rejected run means the local Jacobian was too asymmetric for "
        "`top_eigenpairs` to call its eigenpairs a covariance at all — the "
        "pre-2026-09-16 pipeline never rejected anything, so these "
        "contributed unlabeled, possibly-meaningless numbers into earlier "
        "versions of this table (see docs/LOG.md 2026-09-16 entries). At the "
        "highest σ here, most of the *accepted* runs are also poorly "
        "converged — a bare median eigenvalue at that σ needs this context, "
        "not just the number.",
    ]
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
        "Freezing helps, but freezing *everything* (topology and coarse-level "
        "point coordinates) turned out to be more than necessary: a weaker "
        "`graph_topology_frozen` variant (topology fixed, coarse coordinates "
        "recomputed from the perturbed input) is never rejected by the "
        "symmetry gate across a real-shape audit where the fully-frozen "
        "default was rejected on 2/3 shapes, and is now the default "
        "(`denoiser.graph_freeze_variant`, docs/LOG.md 2026-09-16).",
        "",
        "## 3. Uncertainty structure lives in regions, not whole shapes",
        "",
        "Whole-shape posteriors are near-isotropic (flat spectra). Restricting the "
        "operator to extremity patches (the reference paper's patch-mask move, in "
        "3D) exposes real anisotropy:",
        "",
        "![structure](figures/whole_vs_region_spread.png)",
        "",
        "Flat whole-shape spectra aren't just a cosmetic near-tie: a 2026-09-16 "
        "stability check re-noised 15 trustworthy whole-shape runs with a new "
        "seed (same points) and found the top eigenVALUE stable (median ratio "
        "0.95) but the reported top-5 eigenVECTOR subspace essentially "
        "orthogonal to the original (median max principal angle 89.4°) — a "
        "direct consequence of the near-degenerate spectrum (median 16.6% "
        "top-to-5th spread) leaving the eigenbasis numerically ill-defined. "
        "**Whole-shape mode *directions* are not a reproducible property of the "
        "shape; only the top eigenvalue's magnitude is.** Point resampling was "
        "far gentler (eigenvalue shift ~8%). The natural follow-up question — "
        "are region-restricted modes more directionally stable, given their "
        "less-degenerate spectra? — was tested and answered **no**: median max "
        "subspace angle for regions under the identical new-seed check is "
        "**89.34°**, statistically the same as whole-shape's 89.4°. "
        "Region-restriction fixes eigenvalue spread; it does not fix direction "
        "reproducibility. **Every mode-direction figure below should be read as "
        "the mode for one specific noisy observation, not a reproducible "
        "geometric property of the shape** (docs/LOG.md, docs/PLAN.md).",
        "",
        "## 4. The modes themselves",
        "",
        "Best converged, PSD region runs (spread = top-to-last eigenvalue gap). "
        "Per exemplar: region modes, mode-0 direction arrows, mode-0 sweep "
        "x̂ ± t·√λ·v:",
        "",
    ]
    if exemplars:
        for tag, spread in exemplars:
            lines += [
                f"### {tag}  (spread {spread:.0%})",
                "",
                f"![{tag} modes](figures/{tag}_modes.png)",
                f"![{tag} arrows](figures/{tag}_mode0_arrows.png)",
                f"![{tag} sweep](figures/{tag}_mode0_sweep.png)",
                "",
            ]
        n_exemplars = len(exemplars)
    else:
        # No outputs/masked/masked_modes/metrics.json to pick fresh exemplars from
        # (a common state -- outputs/ is disposable scratch, see WORKFLOW.md).
        # Silently regenerating an empty section 4 here would DELETE the existing
        # exemplar galleries from the committed README instead of leaving them —
        # this actually happened once (2026-09-16). Preserve whatever section 4
        # the current file on disk already has instead of dropping it.
        old = (OUT / "README.md").read_text(encoding="utf-8") if (OUT / "README.md").exists() else ""
        marker = "\n### "  # first exemplar heading -- the section-4 intro
                          # paragraph above is already in `lines`, don't duplicate it
        if "## 4. The modes themselves" in old and marker in old:
            start = old.index(marker) + 1
            # Stop before the next level-2 heading (e.g. a "## 5." added later) so
            # that later sections aren't swallowed into this preserved blob and
            # then duplicated when they're independently regenerated below.
            next_h2 = old.find("\n## ", start)
            chunk = old[start:next_h2] if next_h2 != -1 else old[start:]
            lines += chunk.rstrip("\n").split("\n") + [""]
            n_exemplars = chunk.count(marker) + chunk.startswith("### ")
        else:
            n_exemplars = 0

    if smooth_rows:
        lines += smooth_section(smooth_rows, smooth_audit, smooth_ex)

    if depth_example_shapes:
        lines += [
            "## 6. Side quest: shapes as depth-map images",
            "",
            "`scripts/run_depth2d.py` (docs/LOG.md 2026-09-11/16) renders each "
            "ModelNet40 shape into a single-view depth-map image and runs it "
            "through the reference paper's OWN 2D denoiser (MNIST CNN or FFHQ "
            "DDPM) instead of Noise2Score3D — a cross-domain comparison: what "
            "happens to the reference method's own uncertainty estimate when "
            "fed a photo of a 3D shape instead of a digit or a face? "
            f"({', '.join(depth_example_shapes)} shown here, at the resolution "
            "each denoiser actually receives):",
            "",
            "![depth projections](figures/depth_projection_examples.png)",
            "",
            "The 28x28 MNIST-scale image is genuinely this blocky — a full "
            "point cloud compressed to fewer pixels than it has dimensions of "
            "variation. Finding (50 shapes, MNIST CNN, re-tallied under the "
            "fixed pipeline 2026-09-24): two failure tiers. **11/50 (22%) are "
            "rejected outright** — the operator on the top subspace is too "
            "asymmetric to call a covariance, which the original eigensolver "
            "couldn't detect. **21/50 (42%) are accepted but non-PSD** "
            "(negative eigenvalues with near-zero asymmetry on the top subspace), "
            "a mechanism distinct from the 3D noise-range breakdown above. Only "
            "18/50 (36%) give a PSD spectrum. The old ~50% non-PSD figure mixed "
            "the first two tiers. See docs/LOG.md 2026-09-24.",
        ]

    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"results/ built: {len(list(FIG.glob('*.png')))} figures, "
          f"README with {n_exemplars} exemplars"
          + (" (preserved from disk, no fresh masked-modes data)" if not exemplars else ""))


if __name__ == "__main__":
    main()
