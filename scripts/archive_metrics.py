"""Archive aggregate run metrics into results/metrics/ so analysis survives outputs/.

`outputs/` is disposable scratch (WORKFLOW.md) and is currently empty: the phase-3
sweep, the ablation and the mask grid are gone, so every number not already in
results/ would cost a ~4.5GB re-download plus hours of recompute to recover. This
script makes the *aggregates* durable and git-trackable — they are kilobytes, and
they are what the remaining analysis actually reads.

Three tiers, all idempotent:

1. raw/     — copies every outputs/<profile>/<experiment>/metrics.json it finds.
              Authoritative, full per-run detail. Run this after any experiment.
2. derived/ — rebuilds per-run tables from the committed viewer bundles
              (results/viewer_data*.json), which embed real measured eigenpairs.
              Needs no outputs/ and no model — this is how today's numbers survive.
3. summary/ — medians transcribed from results/README.md and docs/LOG.md, for runs
              whose raw artifacts are already lost.

    python scripts/archive_metrics.py
"""

import base64
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/metrics"


def participation(v: np.ndarray) -> tuple[float, float]:
    """Concentration of one mode's energy. v is (n, d) per-element displacements.

    Returns (participation_ratio, top_element_energy_fraction): PR ~= how many
    elements actually carry the mode (1 = a single-element spike, n = uniform).
    The eigenvalue spread says nothing about this (measured r=0.06), so a mode can
    be well-separated and still be one point moving.
    """
    e = (v ** 2).sum(axis=1)
    tot = e.sum()
    if tot <= 0:
        return 0.0, 0.0
    e = e / tot
    return float(1.0 / (e ** 2).sum()), float(e.max())


def local_spacing(pts: np.ndarray, k: int = 6) -> np.ndarray:
    """Mean distance to the k nearest neighbours, per point — the density proxy for
    R6 (does the calibration drift track local point spacing?)."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, :k].mean(axis=1)


def archive_raw() -> list[str]:
    """Copy authoritative metrics.json files out of outputs/ (if any exist)."""
    dst = OUT / "raw"
    dst.mkdir(parents=True, exist_ok=True)
    found = []
    for p in sorted((ROOT / "outputs").glob("*/*/metrics.json")):
        name = f"{p.parent.parent.name}__{p.parent.name}.json"
        shutil.copyfile(p, dst / name)
        found.append(name)
    return found


def derive_region_runs() -> dict:
    """Per-run table for the 3D mask grid, from results/viewer_data.json."""
    src = json.loads((ROOT / "results/viewer_data.json").read_text())
    clouds = {k: np.array(v, dtype=np.float64).reshape(-1, 3)
              for k, v in src["clouds"].items()}
    spacing = {k: local_spacing(v) for k, v in clouds.items()}

    rows = []
    for r in src["runs"]:
        idx = np.array(r["mask_idx"])
        n = len(idx)
        modes = [np.array(m, dtype=np.float64).reshape(n, 3) for m in r["modes"]]
        prs = [participation(m) for m in modes]
        ev = r["eigvals"]
        rows.append({
            "tag": r["tag"], "shape": r["shape"], "sigma": r["sigma"],
            "region": r["region"], "method": r["method"],
            "n_mask": n,
            "eigvals": ev,
            "lam0_over_sigma2": ev[0] / r["sigma"] ** 2,
            "spread_top_to_last": r["spread"],
            "psd_ok": r["psd_ok"], "conv_ok": r["conv_ok"],
            # derived here, from the stored eigenvectors — no new runs
            "participation_ratio": [p[0] for p in prs],
            "top_point_energy": [p[1] for p in prs],
            "mask_spacing_median": float(np.median(spacing[r["cloud"]][idx])),
            "shape_spacing_median": float(np.median(spacing[r["cloud"]])),
        })
    return {
        "source": "results/viewer_data.json (exported from outputs/masked-grid*)",
        "caveats": [
            "FROZEN RUNS ARE CONVERGENCE-FILTERED: export_viewer_data.py dropped "
            "frozen runs with min overlap < 0.9 (51 of 60 kept), so conv_ok is True "
            "for every frozen row by construction. Do not compute a convergence rate "
            "from this table; use summary/masked_gallery.json.",
            "naive (graph-rebuilt) runs are unfiltered — kept broken on purpose.",
            "Per-run antisymmetry and raw overlap values were never exported and are "
            "lost with outputs/.",
        ],
        "runs": rows,
    }


def derive_whole_shape() -> dict:
    """Whole-shape rank-24 spectra @ sigma=0.03, from results/viewer_data_free.json."""
    src = json.loads((ROOT / "results/viewer_data_free.json").read_text())
    rows = []
    for e in src["free"]:
        pts = np.array(e["x_hat"], dtype=np.float64).reshape(-1, 3)
        n, k, sig = len(pts), e["k"], e["sigma"]
        v = (np.frombuffer(base64.b64decode(e["modes_b64"]), dtype="<i2")
             .astype(np.float64).reshape(k, n, 3) * e["modes_scale"] / 32767)
        prs = [participation(v[i]) for i in range(k)]
        ev = e["eigvals"]
        rows.append({
            "shape": e["shape"], "sigma": sig, "k": k, "n_points": n,
            "eigvals": ev,
            "lam0_over_sigma2": ev[0] / sig ** 2,
            "spread_top5": (ev[0] - ev[4]) / ev[0],
            "participation_ratio": [p[0] for p in prs],
            "top_point_energy": [p[1] for p in prs],
            "spacing_median": float(np.median(local_spacing(pts))),
        })
    return {"source": "results/viewer_data_free.json (outputs/fullspectrum)",
            "note": "whole-shape (unmasked) spectra — the flat-spectrum baseline "
                    "that region masking is compared against.",
            "runs": rows}


def derive_images2d() -> dict:
    """2D cross-check runs, from results/viewer_data_2d.json."""
    src = json.loads((ROOT / "results/viewer_data_2d.json").read_text())
    rows = []
    for e in src["images"]:
        k, c, mh, mw, sig = e["k"], e["c"], e["mh"], e["mw"], e["sigma"]
        v = (np.frombuffer(base64.b64decode(e["modes_b64"]), dtype="<i2")
             .astype(np.float64).reshape(k, c * mh * mw, 1) * e["modes_scale"] / 32767)
        prs = [participation(v[i]) for i in range(k)]
        ev = e["eigvals"]
        rows.append({
            "id": e["id"], "kind": e["kind"], "label": e["label"],
            "sigma": sig, "k": k, "mode_pixels": c * mh * mw,
            "eigvals": ev,
            "lam0_over_sigma2": ev[0] / sig ** 2,
            "participation_ratio": [p[0] for p in prs],
            "top_pixel_energy": [p[1] for p in prs],
        })
    return {
        "source": "results/viewer_data_2d.json (outputs/fullspectrum2d)",
        "note": "MNIST modes are stored at full resolution; face modes at half "
                "(pool=2), so face participation ratios are in half-res pixels.",
        "runs": rows,
    }


# Transcribed, with provenance, because the raw runs no longer exist. Every number
# below is quoted from the cited file — nothing here is recomputed or inferred.
SUMMARY = {
    "calibration_by_sigma": {
        "source": "results/README.md table 1; docs/LOG.md 2026-08-17 Phase-3 entry "
                  "and 2026-08-18 comprehensive-evidence entry",
        "raw_artifacts": "outputs/phase3/run_experiment/ — LOST (outputs/ is empty)",
        "design": "50 shapes x 5 categories per sigma, 5 eigenpairs each "
                  "(250 eigenvalues per sigma), 15 iters, frozen graph",
        "training_sigma_range": [0.004, 0.034],
        "rows": [
            {"sigma": 0.01, "median_lam0_over_sigma2": 1.06, "n_negative_eigvals": 0,
             "median_convergence": 0.999, "antisym_energy": 0.001,
             "lam0_over_sigma2_range": [0.81, 1.59]},
            {"sigma": 0.02, "median_lam0_over_sigma2": 1.39, "n_negative_eigvals": 2,
             "median_convergence": 0.996, "antisym_energy": 0.009,
             "lam0_over_sigma2_range": [1.12, 2.33]},
            {"sigma": 0.03, "median_lam0_over_sigma2": 1.91, "n_negative_eigvals": 19,
             "median_convergence": 0.934, "antisym_energy": None,
             "lam0_over_sigma2_range": None},
            {"sigma": 0.05, "median_lam0_over_sigma2": 0.72, "n_negative_eigvals": 191,
             "median_convergence": 0.408, "antisym_energy": None,
             "lam0_over_sigma2_range": [-7, 7], "beyond_training_range": True},
        ],
        "lost": "per-shape values and per-shape local density — this blocks R6 at "
                "sweep scale; region-scale density is in derived/region_runs.json",
    },
    "calibration_by_sigma_2026_09_16": {
        "source": "docs/LOG.md 2026-09-16 'Full n=50 confirmation' entry",
        "raw_artifacts": "results/metrics/raw/gpu__run_experiment.json — PRESENT "
                         "(this re-run's raw metrics.json was archived, not lost)",
        "design": "Same 50 shapes x 5 categories as calibration_by_sigma above, "
                  "re-run under the 2026-09-16 pipeline: Rayleigh-Ritz eigensolver "
                  "with a symmetry-gate rejection (task 2/4), graph_topology_frozen "
                  "default (task 6, was graph_frozen/'full' for the entry above), "
                  "composite 'trustworthy' flag (task 9: not rejected AND "
                  "convergence>=0.9 AND max Ritz residual<=0.15, see "
                  "pcuq.diagnostics.is_trustworthy). sigma=0.03 not re-run.",
        "training_sigma_range": [0.004, 0.034],
        "rows": [
            {"sigma": 0.01, "median_lam0_over_sigma2": 1.079, "n_negative_eigvals": 1,
             "median_convergence": 0.9985, "antisym_energy": 0.0010,
             "lam0_over_sigma2_range": [1.03, 1.56], "rejected": "5/50 (10%)",
             "trustworthy": "43/50 (86%)"},
            {"sigma": 0.02, "median_lam0_over_sigma2": 1.396, "n_negative_eigvals": 2,
             "median_convergence": 0.9960, "antisym_energy": 0.0079,
             "lam0_over_sigma2_range": [1.19, 1.90], "rejected": "8/50 (16%)",
             "trustworthy": "36/50 (72%)"},
            {"sigma": 0.05, "median_lam0_over_sigma2": -0.533, "n_negative_eigvals": 156,
             "median_convergence": 0.6161, "antisym_energy": 0.0093,
             "lam0_over_sigma2_range": [-7.18, 6.52], "beyond_training_range": True,
             "rejected": "13/50 (26%)", "trustworthy": "8/50 (16%)",
             "note": "26% rejected outright, but a further 58% pass the symmetry "
                     "gate and are STILL not trustworthy (poor convergence or high "
                     "Ritz residual) -- the rejection rate alone badly understates "
                     "the breakdown at this sigma; trustworthy is the number to "
                     "quote, see LOG.md"},
        ],
        "vs_calibration_by_sigma": "in-distribution medians (0.01, 0.02) replicate "
                                   "the entry above almost exactly despite the "
                                   "pipeline changes; sigma=0.05 moved from a "
                                   "badly-non-converged 0.72 (conv 0.408) to a "
                                   "better-converged but more decisively negative "
                                   "-0.533 (conv 0.616) -- same conclusion "
                                   "(breaks down out of range), stronger evidence. "
                                   "Neither the rejected-shape fraction (10-26%) nor "
                                   "the trustworthy fraction (86%/72%/16%) existed as "
                                   "a measurable quantity before this pipeline "
                                   "version -- both are new, not moved numbers.",
    },
    "seed_stability_2026_09_16": {
        "source": "docs/LOG.md 2026-09-16 'Task 9 second half' entry",
        "raw_artifacts": "results/metrics/raw/gpu__audit_seed_stability.json — PRESENT",
        "design": "15 shapes trustworthy @ sigma=0.02 in the n=50 run above; "
                  "re-noised with a new seed (same points) and separately "
                  "resampled (same mesh, different points, same seed formula).",
        "finding": "eigenVALUE magnitude is stable under both perturbations; "
                   "eigenVECTOR direction for whole-shape spectra is NOT stable "
                   "under a new noise seed -- see seed_stability.median_max_"
                   "subspace_angle_degrees below (89.4 = essentially orthogonal). "
                   "Mechanism: near-degenerate top-5 eigenvalues (median spread "
                   "16.6%) leave the eigenbasis numerically ill-defined, so small "
                   "perturbations rotate it freely. Point resampling is much "
                   "gentler than reseeding (median eigenvalue shift ~8% vs. "
                   "eigenvectors going to ~90 degrees apart under reseeding).",
        "seed_stability": {"n": 15, "n_rejected_under_new_seed": 3,
                           "still_trustworthy": 12, "median_top_mode_overlap": 0.041,
                           "median_top_eigval_ratio": 0.952,
                           "median_max_subspace_angle_degrees": 89.36,
                           "median_ref_eigval_spread": 0.166},
        "resample_stability": {"n": 15, "n_rejected_under_resampling": 1,
                               "still_trustworthy": 14,
                               "median_eigval0_over_sigma2_original": 1.457,
                               "median_eigval0_over_sigma2_resampled": 1.337},
        "caveat": "n=15, one sigma (0.02) only, whole-shape spectra only. Whether "
                 "region-restricted (masked) modes are more directionally stable "
                 "-- predicted by this mechanism, since regions have less "
                 "degenerate spectra per Phase 3.5 -- is untested (needs a "
                 "full-scale masked-modes run first, see PLAN.md).",
    },
    "ablation_frozen_vs_rebuilt": {
        "source": "docs/LOG.md 2026-08-18 comprehensive-evidence entry; "
                  "results/figures/frozen_vs_rebuilt.png",
        "raw_artifacts": "outputs/ablation-unfrozen/ — LOST",
        "design": "15 shapes @ sigma=0.02, medians",
        "graph_rebuilt": {"lam0_over_sigma2": 2.91, "antisym_energy": 0.34,
                          "convergence": 0.18},
        "graph_frozen": {"lam0_over_sigma2": 1.39, "antisym_energy": 0.009,
                         "convergence": 1.00},
    },
    "masked_gallery": {
        "source": "docs/LOG.md 2026-08-18 comprehensive-evidence entry",
        "raw_artifacts": "outputs/masked_modes/ — LOST (distinct from the mask GRID "
                         "in derived/region_runs.json: different regions per shape)",
        "design": "60 region runs, 15 shapes x 2 sigmas x 2 regions",
        "sigma_0.02": {"converged_and_psd": 28, "total": 30, "max_spread": 0.33},
        "sigma_0.03": {"converged_and_psd": 22, "total": 30, "median_spread": 0.12,
                       "max_spread": 0.31},
    },
    "naive_baseline_mask_grid": {
        "source": "docs/LOG.md 2026-08-18 two-domain entry",
        "design": "30 regions @ sigma=0.03, graph rebuilt (naive port)",
        "converged": 3, "total": 30, "with_negative_eigvals": 11,
        "median_lam0_over_sigma2": 4,
        "note": "per-run rows for these 30 are recoverable in "
                "derived/region_runs.json (method='naive')",
    },
    "gates": {
        "source": "docs/PLAN.md, docs/LOG.md",
        "permutation_equivariance_rel_err": 2e-8,
        "phase1_eigval_rel_err_max": 0.003,
        "phase1_eigvec_abs_cos_min": 0.995,
        "step_size_plateau_c": 1e-3,
    },
}

README = """# results/metrics/ — durable aggregates

*(Generated by `python scripts/archive_metrics.py` — rerun after any experiment.)*

`outputs/` is disposable scratch (WORKFLOW.md) — it gets cleared between machines/
sessions with no warning. This directory is the git-tracked survival copy of the
numbers the report needs, so no analysis requires a ~4.5GB re-download plus hours of
recompute, and no future run's raw data gets silently lost the way earlier runs did
(see summary/'s "LOST" `raw_artifacts` notes below — that's what re-running this
script after every experiment prevents going forward).

Three tiers, in descending order of authority. **Check which tier a number comes from
before quoting it in the report.**

| Tier | What | Authority |
|---|---|---|
| `raw/` | verbatim `metrics.json` copied from `outputs/`, whatever exists at archive time ({n_raw} files currently) | authoritative — full per-run detail |
| `derived/` | per-run tables rebuilt from the committed viewer bundles | authoritative for the fields present; some fields were never exported |
| `summary/` | medians transcribed from `results/README.md` and `docs/LOG.md` | summary only unless its `raw_artifacts` field points at a `raw/` file that still exists — check before assuming per-shape data is gone |

## Files

- `derived/region_runs.json` — {n_region} 3D mask-grid runs (frozen + naive baseline):
  eigenvalues, spread, PSD/convergence flags, mask size, and per-run **local point
  spacing** (the R6 density input). Read its `caveats` before computing any rate from
  it — the frozen rows are convergence-filtered.
- `derived/whole_shape_spectra.json` — {n_whole} whole-shape rank-24 spectra @ σ=0.03,
  the flat-spectrum baseline that region masking is measured against.
- `derived/images2d.json` — {n_2d} 2D cross-check runs (MNIST + DDPM faces).
- `summary/*.json` — the phase-3 calibration sweep, its 2026-09-16 n=50 re-run under
  the corrected pipeline (graph_topology_frozen + symmetry-gate rejection — see its
  `vs_calibration_by_sigma` field for what changed and what didn't), the 2026-09-16
  seed/resample stability audit (whole-shape mode DIRECTIONS are noise artifacts;
  magnitude is not — see its `finding` field), the frozen-vs-rebuilt ablation, the
  60-run masked gallery, the naive baseline, and the validation gates.
- `raw/gpu__run_experiment.json` — the full 150-row (shape, sigma) detail behind the
  2026-09-16 re-run above, including which shapes/sigmas got rejected and why.
- `raw/gpu__audit_seed_stability.json` — the full per-shape detail (mode overlap,
  subspace angles, eigenvalue ratios) behind the seed/resample stability audit.

## Derived columns added here

`participation_ratio` and `top_point_energy` measure how concentrated a mode is: PR ≈
how many points (or pixels) actually carry it, so PR ≈ 1 is a single-element spike and
PR ≈ n is a uniform field. Both are computed from the stored eigenvectors — no new
runs. They are **not** interchangeable with `spread_top_to_last` (the eigenvalue gap):
the measured correlation between them is r = 0.06.
"""


def main() -> None:
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    (OUT / "summary").mkdir(parents=True, exist_ok=True)

    found = archive_raw()
    print(f"raw/     : {len(found)} metrics.json copied from outputs/"
          + (f" ({', '.join(found)})" if found else " — outputs/ is empty"))

    derived = {
        "region_runs": derive_region_runs(),
        "whole_shape_spectra": derive_whole_shape(),
        "images2d": derive_images2d(),
    }
    for name, payload in derived.items():
        p = OUT / "derived" / f"{name}.json"
        p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"derived/ : {name}.json — {len(payload['runs'])} runs "
              f"({p.stat().st_size / 1e3:.0f} kB)")

    for name, payload in SUMMARY.items():
        (OUT / "summary" / f"{name}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"summary/ : {len(SUMMARY)} transcribed tables")

    (OUT / "README.md").write_text(README.format(
        n_region=len(derived["region_runs"]["runs"]),
        n_whole=len(derived["whole_shape_spectra"]["runs"]),
        n_2d=len(derived["images2d"]["runs"]),
        n_raw=len(found)), encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"\ndone -> {OUT} ({total / 1e3:.0f} kB total)")


if __name__ == "__main__":
    main()
