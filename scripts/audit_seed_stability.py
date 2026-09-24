"""Task-9 (second half) stability audit: does a reported sensitivity mode survive
a different noise realization, and does it survive a different point sampling of
the same mesh?

Two checks, both reusing the already-computed, trustworthy sigma=0.02 shapes from
outputs/gpu/run_experiment/ as the reference (task 8: don't recompute what's
already saved and archived):

  seed_stability     -- same points (x), a NEW noise draw at the same sigma. Point
                        correspondence is exact, so eigenvectors from the original
                        and re-noised run are directly comparable: |<v_old, v_new>|
                        per mode, plus whether the re-noised run is still
                        `trustworthy` at all.
  resample_stability -- same mesh, a DIFFERENT point sampling (load_modelnet with a
                        different seed selects the same shapes by name -- shape
                        selection doesn't depend on seed, only per-shape point
                        sampling does, see pcuq.data.load_modelnet). Point
                        correspondence is NOT preserved across two independent
                        samplings of the same continuous surface, so only
                        eigenVALUE statistics are compared, not eigenvector overlap.

Small scale by design (a subset of the already-trustworthy shapes, not all 50) --
a stability check, not a new sweep at full sample size.

Usage:
    python scripts/audit_seed_stability.py --config configs/gpu.yaml --override device=cpu
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pcuq.data import corrupt, load_modelnet
from pcuq.denoisers import Noise2Score3DWrapper
from pcuq.diagnostics import is_trustworthy
from pcuq.spectrum import top_eigenpairs
from pcuq.utils import apply_overrides, load_config, make_out_dir, set_seed

N_SHAPES_PER_CHECK = 15
NEW_SEED = 999  # distinct from cfg["seed"] (0) used everywhere else this project


def run_one(den, y, sigma, sp, jc, trust_seed):
    """Returns (eigvecs, m) on success, or (None, m) if top_eigenpairs rejects the
    new noise draw / resampling -- a previously-trustworthy shape failing under a
    perturbation is itself a real stability result, not a script error."""
    try:
        with den.graph_topology_frozen():
            with torch.no_grad():
                den(y[None])
            torch.manual_seed(trust_seed)
            eigvecs, eigvals, history, diag = top_eigenpairs(
                den, y, sigma, k=sp["n_ev"], iters=sp["iters"], method=jc["method"],
                c=jc["c"], return_diagnostics=True)
    except ValueError as e:
        return None, {"top_eigenpairs_rejected": str(e), "trustworthy": False}
    m = {"eigvals": eigvals.tolist(), "final_iter_overlap": history[-1].tolist(),
        "ritz_relative_residuals": diag["ritz_relative_residuals"].tolist()}
    m["trustworthy"] = is_trustworthy(m)
    return eigvecs, m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg["seed"])
    sigma = 0.02  # the clean, in-distribution row -- isolates stability from the
                  # already-documented sigma=0.05 breakdown
    jc, sp = cfg["jacobian"], cfg["spectrum"]
    out = make_out_dir(cfg, "audit_seed_stability")
    ref_dir = Path(cfg.get("out_dir", "outputs")) / "gpu" / "run_experiment"
    print(f"profile={cfg['name']} sigma={sigma} out={out} ref={ref_dir}")

    ref_metrics = json.loads((ref_dir / "metrics.json").read_text())
    trustworthy_tags = sorted(k for k, m in ref_metrics.items()
                              if k.endswith(f"sigma{sigma}") and m.get("trustworthy"))
    subset = trustworthy_tags[:N_SHAPES_PER_CHECK]
    print(f"{len(trustworthy_tags)} trustworthy @ sigma={sigma} in the reference run; "
          f"using {len(subset)} for each check")

    den = Noise2Score3DWrapper(cfg["denoiser"]["repo"], cfg["denoiser"]["checkpoint"],
                               sigma=sigma, device=torch.device("cpu"))

    results = {"seed_stability": {}, "resample_stability": {}}

    # --- Check 1: same points, new noise draw ---------------------------------
    print("\n=== seed stability (same points, new noise) ===")
    for tag in subset:
        name = tag.split("_sigma")[0]
        ref = torch.load(ref_dir / f"{tag}.pt", weights_only=False)
        x, ref_eigvecs, ref_eigvals = ref["x"], ref["eigvecs"], ref["eigvals"]
        y_new = corrupt(x[None], sigma, NEW_SEED)[0]
        eigvecs_new, m_new = run_one(den, y_new, sigma, sp, jc, cfg["seed"])
        results["seed_stability"][name] = m_new

        if eigvecs_new is None:
            print(f"[{name}] REJECTED under the new noise seed "
                  f"(was trustworthy under the original)")
            continue
        k = min(len(ref_eigvals), len(m_new["eigvals"]))
        eigvals = torch.tensor(m_new["eigvals"])
        ref_flat, new_flat = ref_eigvecs[:k].reshape(k, -1), eigvecs_new[:k].reshape(k, -1)
        overlap = (ref_flat * new_flat).sum(dim=1).abs()
        ratio = (eigvals[:k] / ref_eigvals[:k])
        # Per-mode overlap conflates two different questions: is the SPAN of the
        # top-k modes stable (subspace principal angles), or is each individual
        # eigenVECTOR within it stable (per-mode overlap above)? Near-degenerate
        # eigenvalues (a near-flat top-k spectrum) let the subspace be stable while
        # individual directions rotate freely inside it -- distinguishing these
        # matters for whether "the top mode" is a meaningful reported direction.
        principal_angles = torch.rad2deg(torch.acos(
            torch.linalg.svdvals((ref_flat @ new_flat.T).cpu()).clamp(-1, 1)))
        m_new["mode_overlap_vs_original_seed"] = overlap.tolist()
        m_new["eigval_ratio_vs_original_seed"] = ratio.tolist()
        m_new["subspace_principal_angles_degrees_vs_original_seed"] = principal_angles.tolist()
        m_new["ref_eigval_spread"] = float((ref_eigvals[0] - ref_eigvals[k - 1])
                                           / ref_eigvals[0].clamp(min=1e-30))
        print(f"[{name}] still_trustworthy={m_new['trustworthy']} | "
              f"mode_overlap={[f'{v:.3f}' for v in overlap.tolist()]} | "
              f"eigval_ratio={[f'{v:.2f}' for v in ratio.tolist()]} | "
              f"subspace_angles_deg={[f'{v:.1f}' for v in principal_angles.tolist()]}")

    # --- Check 2: same mesh, different point sampling --------------------------
    print("\n=== resample stability (same mesh, different point sampling) ===")
    resampled_cfg = dict(cfg, seed=cfg["seed"] + 12345)  # shape SELECTION is seed-
    # independent (pcuq.data.load_modelnet); only per-shape point sampling moves.
    resampled_shapes = dict(load_modelnet(resampled_cfg, torch.float32))
    for tag in subset:
        name = tag.split("_sigma")[0]
        ref = ref_metrics[tag]
        x_new = resampled_shapes[name]
        y_new = corrupt(x_new[None], sigma, resampled_cfg["seed"])[0]
        eigvecs_new, m_new = run_one(den, y_new, sigma, sp, jc, cfg["seed"])
        m_new["eigval0_over_sigma2_original"] = ref["eigvals"][0] / sigma**2
        results["resample_stability"][name] = m_new

        if eigvecs_new is None:
            print(f"[{name}] REJECTED under resampling "
                  f"(was trustworthy under the original sampling)")
            continue
        m_new["eigval0_over_sigma2_resampled"] = m_new["eigvals"][0] / sigma**2
        print(f"[{name}] still_trustworthy={m_new['trustworthy']} | "
              f"lambda0/sigma^2: {m_new['eigval0_over_sigma2_original']:.2f} -> "
              f"{m_new['eigval0_over_sigma2_resampled']:.2f}")

    # --- Summary ------------------------------------------------------------
    # (rejected rows lack the fields below by construction -- excluded from
    # those specific medians, but still counted in n / still_trustworthy)
    seed_rows = list(results["seed_stability"].values())
    seed_ok = [r for r in seed_rows if "mode_overlap_vs_original_seed" in r]
    resample_rows = list(results["resample_stability"].values())
    resample_ok = [r for r in resample_rows if "eigval0_over_sigma2_resampled" in r]
    summary = {
        "seed_stability": {
            "n": len(seed_rows),
            "n_rejected_under_new_seed": len(seed_rows) - len(seed_ok),
            "still_trustworthy": sum(1 for r in seed_rows if r["trustworthy"]),
            "median_top_mode_overlap": st.median(
                r["mode_overlap_vs_original_seed"][0] for r in seed_ok) if seed_ok else None,
            "median_top_eigval_ratio": st.median(
                r["eigval_ratio_vs_original_seed"][0] for r in seed_ok) if seed_ok else None,
            "median_max_subspace_angle_degrees": st.median(
                max(r["subspace_principal_angles_degrees_vs_original_seed"])
                for r in seed_ok) if seed_ok else None,
            "median_ref_eigval_spread": st.median(
                r["ref_eigval_spread"] for r in seed_ok) if seed_ok else None,
        },
        "resample_stability": {
            "n": len(resample_rows),
            "n_rejected_under_resampling": len(resample_rows) - len(resample_ok),
            "still_trustworthy": sum(1 for r in resample_rows if r["trustworthy"]),
            "median_eigval0_over_sigma2_original": st.median(
                r["eigval0_over_sigma2_original"] for r in resample_rows),
            "median_eigval0_over_sigma2_resampled": st.median(
                r["eigval0_over_sigma2_resampled"] for r in resample_ok) if resample_ok else None,
        },
    }
    results["summary"] = summary
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))

    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ndone — metrics at {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
