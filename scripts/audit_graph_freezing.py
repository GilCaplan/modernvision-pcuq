"""Task-6-style audit: how should the frozen Noise2Score3D wrapper differentiate
through the model's discrete voxel/radius graph pyramid?

Their forward rebuilds that graph from scratch every call, so a naive finite
difference y +/- c*v recomputes discrete voxel/neighbor assignments at the
perturbed point too -- a perturbation that nudges a point across a voxel boundary
flips the graph, and the finite difference picks up an O(1) jump instead of the
local derivative (docs/LOG.md 2026-08-17/18). Compares three ways of handling this,
all implemented in the wrapper only (external/ untouched):

  regular              -- rebuild the graph fresh at every perturbed forward
                           (the literal derivative of "the model", jumps and all)
  graph_frozen         -- topology AND coarse-level point coordinates both frozen
                           at the anchor; only level-0 points vary (current default)
  graph_topology_frozen -- topology (voxel-group membership, radius-search edges)
                           frozen at the anchor, but coarse-level point COORDINATES
                           are recomputed as voxel means of the perturbed level-0
                           input through that fixed membership (denoisers.py)

Each function actually being differentiated is a different closed-form map from
level-0 points to output; only `graph_frozen`/`graph_topology_frozen`'s maps are
smooth in their perturbation. `regular`'s is piecewise-smooth with jump
discontinuities at voxel/radius boundaries -- top_eigenpairs correctly refuses it
outright (see antisym rejection below), so `regular` is diagnosed directly here
instead of going through the eigensolver's strict gate.

Small scale by design (n_points, n_shapes, CPU) -- a correctness/stability audit,
not a scaled experiment; not profile-driven like run_experiment.py's n_points. Uses
real ModelNet40 shapes (already cached locally) rather than one fixed toy anchor,
so the comparison isn't a single-sample fluke.

Usage:
    python scripts/audit_graph_freezing.py --config configs/local.yaml
"""

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pcuq.data import corrupt, load_modelnet
from pcuq.denoisers import Noise2Score3DWrapper
from pcuq.diagnostics import _unit_probes, antisym_energy_fd, sweep_step_size_fd
from pcuq.spectrum import top_eigenpairs
from pcuq.utils import apply_overrides, load_config, make_out_dir, set_seed

N_POINTS = 256
N_SHAPES = 3
STEP_SIZES = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]
VARIANTS = ("regular", "graph_frozen", "graph_topology_frozen")


def new_context(den: Noise2Score3DWrapper, name: str):
    """A used @contextlib.contextmanager CM can't be re-entered -- callers need a
    fresh one per `with` block, hence a factory rather than a pre-built dict."""
    if name == "regular":
        return contextlib.nullcontext()
    if name == "graph_frozen":
        return den.graph_frozen()
    return den.graph_topology_frozen()


def run_variant(den: Noise2Score3DWrapper, y: torch.Tensor, name: str, sigma: float,
                jc: dict, sp: dict, seed: int) -> dict:
    v_pair = _unit_probes(y, 2, seed=1)  # shared probes across variants/shapes
    with new_context(den, name):
        with torch.no_grad():
            den(y[None])  # anchor forward — builds/caches the pyramid at y for the
                          # frozen variants (must happen before any jvp perturbs the
                          # input, or the "anchor" would end up being y+c*v)
        sweep = sweep_step_size_fd(den, y, STEP_SIZES, method=jc["method"])
        antisym = antisym_energy_fd(den, y, v_pair, method=jc["method"], c=jc["c"])
    m = {"step_size_self_consistency": sweep, "antisym_energy_probe": antisym}

    if name != "regular":  # top_eigenpairs correctly refuses `regular` (too asymmetric)
        try:
            with new_context(den, name):
                with torch.no_grad():
                    den(y[None])
                torch.manual_seed(seed)
                _, eigvals, history, diag = top_eigenpairs(
                    den, y, sigma, k=sp["n_ev"], iters=sp["iters"], method=jc["method"],
                    c=jc["c"], return_diagnostics=True)
            m.update({
                "eigvals": eigvals.tolist(),
                "eigval_over_sigma_sq": (eigvals / sigma**2).tolist(),
                "final_iter_overlap": history[-1].tolist(),
                "final_subspace_max_angle_degrees": float(
                    diag["subspace_principal_angles_degrees"][-1].max()),
                "ritz_relative_residuals": diag["ritz_relative_residuals"].tolist(),
            })
        except ValueError as e:  # even the current default can occasionally fail
            m["top_eigenpairs_rejected"] = str(e)  # this exact gate -- worth surfacing, not masking
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg["seed"])
    sigma = cfg["data"]["sigmas"][0]
    jc, sp = cfg["jacobian"], cfg["spectrum"]
    out = make_out_dir(cfg, "audit_graph_freezing")
    print(f"profile={cfg['name']} sigma={sigma} n_points={N_POINTS} "
          f"n_shapes={N_SHAPES} out={out}")

    shapes = load_modelnet(
        {"data": {"root": cfg["data"]["root"], "n_shapes": N_SHAPES,
                  "n_points": N_POINTS, "categories": cfg["data"]["categories"]},
         "seed": cfg["seed"]}, torch.float32)

    den = Noise2Score3DWrapper(cfg["denoiser"]["repo"], cfg["denoiser"]["checkpoint"],
                               sigma=sigma, device=torch.device("cpu"))

    metrics = {}
    for si, (name, x) in enumerate(shapes):
        y = corrupt(x[None], sigma, cfg["seed"] + si)[0]
        print(f"\n### shape {name} ###")
        for variant in VARIANTS:
            m = run_variant(den, y, variant, sigma, jc, sp, cfg["seed"])
            metrics[f"{name}__{variant}"] = m
            extra = ""
            if "top_eigenpairs_rejected" in m:
                extra = " | top_eigenpairs: REJECTED (too asymmetric)"
            elif variant != "regular":
                extra = (f" | eigvals/sigma^2={m['eigval_over_sigma_sq']} | "
                        f"conv={m['final_iter_overlap']}")
            print(f"  [{variant}] antisym={m['antisym_energy_probe']:.3f} | "
                  f"c={jc['c']:.0e} self-consistency="
                  f"{m['step_size_self_consistency'][jc['c']]:.2e}{extra}")

    print(f"\n=== medians across {N_SHAPES} shapes (at c={jc['c']:.0e}) ===")
    summary = {}
    for variant in VARIANTS:
        rows = [metrics[k] for k in metrics if k.endswith(f"__{variant}")]
        entry = {
            "antisym_energy_probe": statistics.median(r["antisym_energy_probe"] for r in rows),
            "step_size_self_consistency_at_c": statistics.median(
                r["step_size_self_consistency"][jc["c"]] for r in rows),
        }
        if variant != "regular":
            ok_rows = [r for r in rows if "top_eigenpairs_rejected" not in r]
            entry["n_rejected"] = len(rows) - len(ok_rows)
            if ok_rows:
                entry["eigval0_over_sigma_sq"] = statistics.median(
                    r["eigval_over_sigma_sq"][0] for r in ok_rows)
                entry["final_iter_overlap_min"] = statistics.median(
                    min(r["final_iter_overlap"]) for r in ok_rows)
        summary[variant] = entry
        print(f"  {variant}: {entry}")
    metrics["summary"] = summary

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\ndone — metrics at {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
