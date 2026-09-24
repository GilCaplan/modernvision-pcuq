"""Follow-up to the 2026-09-16 whole-shape seed-stability finding (docs/LOG.md):
whole-shape "top uncertainty mode" DIRECTIONS turned out to be noise-realization
artifacts (median max subspace angle 89.4 degrees under a new noise seed), because
whole-shape spectra are nearly degenerate (median top-5 spread 16.6%). Phase 3.5's
region-restricted spectra are less degenerate (more spread) -- this tests the
predicted consequence: are region-restricted mode DIRECTIONS more stable under the
exact same "same points, new noise seed" perturbation?

Reuses a fresh region-mode reference run (outputs/gpu/masked_modes/, run under the
current pipeline -- graph_topology_frozen, symmetry-gate rejection, trustworthy
flag) rather than the historical masked_modes exemplars, which predate all of this
session's pipeline fixes and were never checked against these criteria.

Same points, new noise draw only (no resample-stability leg here -- the whole-shape
audit already established resampling is gentler than reseeding; this only needs to
answer the seed-stability question for regions).

Usage:
    python scripts/audit_region_seed_stability.py --config configs/gpu.yaml --override device=cpu
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

N_REGIONS_TO_TEST = 15
NEW_SEED = 999  # same offset used by audit_seed_stability.py, for comparability


def run_one(den, y, sigma, sp, jc, mask, trust_seed):
    """Mirrors audit_seed_stability.run_one, restricted to the region mask."""
    try:
        with den.graph_topology_frozen():
            with torch.no_grad():
                den(y[None])
            torch.manual_seed(trust_seed)
            eigvecs, eigvals, history, diag = top_eigenpairs(
                den, y, sigma, k=sp["n_ev"], iters=sp["iters"], method=jc["method"],
                c=jc["c"], mask=mask, return_diagnostics=True)
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
    sigma = 0.02  # matches audit_seed_stability.py's whole-shape check exactly
    jc, sp = cfg["jacobian"], cfg["spectrum"]
    out = make_out_dir(cfg, "audit_region_seed_stability")
    ref_dir = Path(cfg.get("out_dir", "outputs")) / cfg["name"] / "masked_modes"
    print(f"profile={cfg['name']} sigma={sigma} out={out} ref={ref_dir}")

    ref_metrics = json.loads((ref_dir / "metrics.json").read_text())
    trustworthy_tags = sorted(k for k, m in ref_metrics.items()
                              if f"_sigma{sigma}_" in k and m.get("trustworthy"))
    subset = trustworthy_tags[:N_REGIONS_TO_TEST]
    print(f"{len(trustworthy_tags)} trustworthy regions @ sigma={sigma} in the "
          f"reference run; using {len(subset)}")

    # Deterministic re-derivation of the clean points (load_modelnet needs no
    # randomness beyond the seed already in cfg -- same shapes, same points).
    shapes_by_name = dict(load_modelnet(cfg, torch.float32))

    den = Noise2Score3DWrapper(cfg["denoiser"]["repo"], cfg["denoiser"]["checkpoint"],
                               sigma=sigma, device=torch.device("cpu"))

    results = {}
    print("\n=== region seed stability (same points, same mask, new noise) ===")
    for tag in subset:
        name = tag.split("_sigma")[0]
        ref = torch.load(ref_dir / f"{tag}.pt", weights_only=False)
        mask, ref_eigvecs, ref_eigvals = ref["mask"], ref["eigvecs"], ref["eigvals"]
        x = shapes_by_name[name]
        y_new = corrupt(x[None], sigma, NEW_SEED)[0]
        eigvecs_new, m_new = run_one(den, y_new, sigma, sp, jc, mask, cfg["seed"])
        results[tag] = m_new

        if eigvecs_new is None:
            print(f"[{tag}] REJECTED under the new noise seed "
                  f"(was trustworthy under the original)")
            continue
        k = min(len(ref_eigvals), len(m_new["eigvals"]))
        eigvals = torch.tensor(m_new["eigvals"])
        ref_flat = ref_eigvecs[:k].reshape(k, -1)
        new_flat = eigvecs_new[:k].reshape(k, -1)
        overlap = (ref_flat * new_flat).sum(dim=1).abs()
        ratio = eigvals[:k] / ref_eigvals[:k]
        principal_angles = torch.rad2deg(torch.acos(
            torch.linalg.svdvals((ref_flat @ new_flat.T).cpu()).clamp(-1, 1)))
        m_new["mode_overlap_vs_original_seed"] = overlap.tolist()
        m_new["eigval_ratio_vs_original_seed"] = ratio.tolist()
        m_new["subspace_principal_angles_degrees_vs_original_seed"] = principal_angles.tolist()
        m_new["ref_eigval_spread"] = float((ref_eigvals[0] - ref_eigvals[k - 1])
                                           / ref_eigvals[0].clamp(min=1e-30))
        print(f"[{tag}] still_trustworthy={m_new['trustworthy']} | "
              f"mode_overlap={[f'{v:.3f}' for v in overlap.tolist()]} | "
              f"eigval_ratio={[f'{v:.2f}' for v in ratio.tolist()]} | "
              f"subspace_angles_deg={[f'{v:.1f}' for v in principal_angles.tolist()]}")

    rows = list(results.values())
    ok = [r for r in rows if "mode_overlap_vs_original_seed" in r]
    summary = {
        "n": len(rows),
        "n_rejected_under_new_seed": len(rows) - len(ok),
        "still_trustworthy": sum(1 for r in rows if r["trustworthy"]),
        "median_top_mode_overlap": st.median(
            r["mode_overlap_vs_original_seed"][0] for r in ok) if ok else None,
        "median_top_eigval_ratio": st.median(
            r["eigval_ratio_vs_original_seed"][0] for r in ok) if ok else None,
        "median_max_subspace_angle_degrees": st.median(
            max(r["subspace_principal_angles_degrees_vs_original_seed"])
            for r in ok) if ok else None,
        "median_ref_eigval_spread": st.median(
            r["ref_eigval_spread"] for r in ok) if ok else None,
    }
    results["summary"] = summary
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print("\ncompare median_max_subspace_angle_degrees to the whole-shape result "
          "(89.4 degrees, docs/LOG.md 2026-09-16) -- lower means regions really are "
          "more directionally stable, as their less-degenerate spectra predict.")

    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ndone — metrics at {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
