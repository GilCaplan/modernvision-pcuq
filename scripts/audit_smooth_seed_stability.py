"""Seed-stability audit for the smooth, non-rigid modes (spectrum.smooth_eigenpairs).

Same question as audit_seed_stability.py asked of the whole-shape and region modes
(docs/LOG.md 2026-09-16: both gave ~89 degree max subspace angles under a new noise
seed): does a reported smooth mode survive a different noise realization of the
same points? Point correspondence is exact (same x, new noise), so eigenvectors are
compared directly in R^{3N}.

Smooth modes live in a ~84-dimensional basis, so an angle near 90 degrees is also
what two unrelated subspaces would give. The script therefore reports a chance
level: the same statistics for random k-dimensional subspaces of the reference
run's own basis.

Reference: the trustworthy sigma=0.02 smooth runs of a run_experiment.py run with
spectrum.smooth.enabled (default outputs/smooth/run_experiment/).

Usage:
    python scripts/audit_smooth_seed_stability.py --config configs/gpu.yaml \\
        --override name=smooth device=cpu
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pcuq.data import corrupt
from pcuq.denoisers import Noise2Score3DWrapper
from pcuq.spectrum import smooth_eigenpairs
from pcuq.utils import apply_overrides, load_config, make_out_dir, set_seed

N_SHAPES = 15
NEW_SEED = 999  # same new seed as audit_seed_stability.py
N_CHANCE = 200


def angles_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Principal angles between the row spaces of orthonormal a, b (k, D)."""
    return torch.rad2deg(torch.acos(torch.linalg.svdvals(a @ b.T).clamp(-1, 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--ref", default="smooth", help="profile name of the reference run")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg["seed"])
    sigma = 0.02  # in-distribution row, as in audit_seed_stability.py
    jc, sm = cfg["jacobian"], cfg["spectrum"]["smooth"]
    out = make_out_dir(cfg, "audit_smooth_seed_stability")
    ref_dir = Path(cfg.get("out_dir", "outputs")) / args.ref / "run_experiment"
    ref_metrics = json.loads((ref_dir / "metrics.json").read_text())
    tags = sorted(t for t, m in ref_metrics.items()
                  if t.endswith(f"sigma{sigma}") and m.get("smooth", {}).get("trustworthy"))
    subset = tags[:N_SHAPES]
    print(f"{len(tags)} trustworthy smooth runs @ sigma={sigma} in {ref_dir}; using {len(subset)}")

    den = Noise2Score3DWrapper(cfg["denoiser"]["repo"], cfg["denoiser"]["checkpoint"],
                               sigma=sigma, device=torch.device("cpu"))
    frozen = (den.graph_topology_frozen if cfg["denoiser"].get("graph_freeze_variant", "full")
              == "topology" else den.graph_frozen)
    gen = torch.Generator().manual_seed(cfg["seed"])
    rows = {}
    for tag in subset:
        name = tag.split("_sigma")[0]
        ref = torch.load(ref_dir / f"{tag}_smooth.pt", weights_only=True)
        k = len(ref["eigvals"])
        y_new = corrupt(ref["x"][None], sigma, NEW_SEED)[0]
        try:
            with frozen():
                with torch.no_grad():
                    x_hat = den(y_new[None])[0]
                new = smooth_eigenpairs(den, y_new, x_hat, sigma, k=k,
                                        n_basis=sm.get("n_basis", 30),
                                        n_neighbors=sm.get("n_neighbors", 16),
                                        method=jc["method"], c=jc["c"],
                                        batch_jvp=jc.get("batch_jvp", 2))
        except ValueError as e:
            rows[name] = {"smooth_rejected": str(e)}
            print(f"[{name}] REJECTED under the new noise seed")
            continue
        a = ref["eigvecs"].reshape(k, -1).double()
        b = new["eigvecs"].reshape(k, -1).double()
        ang = angles_deg(a, b)
        # chance: random k-dim subspaces of the reference basis (3N x d)
        basis = ref["basis"].double()
        chance = []
        for _ in range(N_CHANCE):
            q1, _ = torch.linalg.qr(torch.randn(basis.shape[1], k, generator=gen, dtype=torch.float64))
            q2, _ = torch.linalg.qr(torch.randn(basis.shape[1], k, generator=gen, dtype=torch.float64))
            chance.append(angles_deg((basis @ q1).T, (basis @ q2).T))
        chance = torch.stack(chance)
        lam = ref["eigvals"].double()
        rows[name] = {
            "eigval_ratio_vs_original_seed": (new["eigvals"] / lam).tolist(),
            "mode_overlap_vs_original_seed": (a * b).sum(1).abs().tolist(),
            "subspace_principal_angles_degrees_vs_original_seed": ang.tolist(),
            "chance_median_max_angle_degrees": float(chance.max(1).values.median()),
            "chance_median_min_angle_degrees": float(chance.min(1).values.median()),
            "ref_eigval_spread": float((lam[0] - lam[-1]) / lam[0].clamp(min=1e-30)),
            "ref_eigval0_over_sigma2": float(lam[0] / sigma**2),
            "basis_dimension": int(basis.shape[1]),
        }
        r = rows[name]
        print(f"[{name}] angles={[f'{v:.1f}' for v in ang.tolist()]} "
              f"(chance max {r['chance_median_max_angle_degrees']:.1f}, "
              f"min {r['chance_median_min_angle_degrees']:.1f}) | "
              f"top overlap {r['mode_overlap_vs_original_seed'][0]:.3f} | "
              f"spread {r['ref_eigval_spread']:.4f}")

    ok = [r for r in rows.values() if "eigval_ratio_vs_original_seed" in r]
    med = lambda f: st.median(f(r) for r in ok) if ok else None
    summary = {
        "n": len(rows), "n_rejected_under_new_seed": len(rows) - len(ok),
        "median_top_eigval_ratio": med(lambda r: r["eigval_ratio_vs_original_seed"][0]),
        "median_top_mode_overlap": med(lambda r: r["mode_overlap_vs_original_seed"][0]),
        "median_max_subspace_angle_degrees": med(
            lambda r: max(r["subspace_principal_angles_degrees_vs_original_seed"])),
        "median_min_subspace_angle_degrees": med(
            lambda r: min(r["subspace_principal_angles_degrees_vs_original_seed"])),
        "chance_median_max_angle_degrees": med(lambda r: r["chance_median_max_angle_degrees"]),
        "chance_median_min_angle_degrees": med(lambda r: r["chance_median_min_angle_degrees"]),
        "median_ref_eigval_spread": med(lambda r: r["ref_eigval_spread"]),
        "median_ref_eigval0_over_sigma2": med(lambda r: r["ref_eigval0_over_sigma2"]),
    }
    print("\n=== summary ===\n" + json.dumps(summary, indent=2))
    (out / "metrics.json").write_text(json.dumps({"runs": rows, "summary": summary}, indent=2))
    print(f"\ndone — metrics at {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
