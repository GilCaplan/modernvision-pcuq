# Experiment & Decision Log

*(Append-only, newest entry on top. One entry per: experiment result worth keeping,
design decision, surprise, or dead end. `outputs/` is disposable; this file is not.)*

Entry template:

```markdown
## YYYY-MM-DD — <short title>
**Who:** · **Machine:** mac | gpu-vm · **Config:** configs/<file> (+ overrides)
**What:** one paragraph — what was run/decided and why.
**Result:** numbers, paths to figures under outputs/, or the decision taken.
**Next:** what this implies.
```

---

## 2026-09-16 — Closing the region-mode gap: fresh validated exemplars, and a surprising negative result on direction stability
**Who:** Claude (with Galit) · **Machine:** windows (CPU, ~45 min total) · **Config:**
configs/gpu.yaml (`--override device=cpu data.n_shapes=15 data.sigmas=[0.02]`)
**What:** Closed the gap flagged in the "ready to submit?" discussion — the 6 region-
mode exemplar figures in `results/README.md` §4 predated every pipeline fix this
session made (Rayleigh-Ritz, symmetry gate, `graph_topology_frozen`, `trustworthy`)
and had never been re-validated against them. Three steps:
1. **Fresh region-mode reference run** under the current pipeline: 15 shapes x 3
   regions x sigma=0.02 (46 region runs — `run_masked_modes.py` already had
   `graph_freeze_variant`/`trustworthy` wired in from earlier work, so no code
   changes needed there). **87% trustworthy, 9% rejected** — a *better* rate than
   whole-shape's 72%/16% at the same sigma, though not the question this was really
   about (see step 2).
2. **The actual follow-up question**: does region-restriction, which already gives
   more eigenvalue *spread* (Phase 3.5), also give more directionally *stable*
   modes under a new noise seed, as the flat-spectrum mechanism from the whole-shape
   finding would predict? New script `scripts/audit_region_seed_stability.py`
   (same design as `audit_seed_stability.py`: same points, same region mask, new
   noise draw, correspondence-exact eigenvector comparison), 15 trustworthy regions.
   **Result: no. Median max subspace angle = 89.34 degrees — statistically
   indistinguishable from whole-shape's 89.4 degrees.** Region-restriction fixes
   eigenvalue degeneracy/spread; it does NOT fix eigenvector-direction reproducibility
   across noise realizations. This was a real, honest test of a stated prediction,
   and the prediction was wrong — worth recording as such, not quietly dropped.
   2/15 regions (13%) also got rejected outright under the new seed, comparable to
   whole-shape's 20%.
3. **Fixed `scripts/build_results.py` to find and use this fresh data**: it had the
   same stale `outputs/phase3`/`outputs/masked` path assumptions already fixed for
   `run_experiment.py`'s data (2026-09-16, above) but never applied to masked-modes
   loading or exemplar picking; added a `outputs/gpu/masked_modes/` fallback.
   `pick_exemplars` now also requires `trustworthy` (previously only checked
   convergence + PSD, which misses the symmetry-gate/Ritz-residual failure modes).
   Also fixed two latent crashes this exposed: `chart_structure` and `chart_ablation`
   iterated raw per-shape dicts without filtering `top_eigenpairs_rejected` entries
   (which lack `eigvals` entirely) — same class of bug as the build_results.py fixes
   earlier, in code paths not exercised until real rejected-run data existed.
**Result:** `results/README.md` §4 now shows **5 fresh exemplars** (table_0393,
lamp_0127, airplane_0629, chair_0891, guitar_0156, all sigma=0.02_r*), all
`trustworthy` under the current pipeline — replacing the 6 pre-fix ones. Fewer sigma
values represented (0.02 only, vs. the old set's 0.02/0.03 mix) since that's what
this validation pass covered; restoring that diversity would need another run, not
done here. `pytest tests/` — 34 passed, no regressions.
**What this means for the report — say this explicitly:** exemplar mode-direction
figures (whole-shape or region) should be captioned as *the mode for one specific
noisy observation*, not as a reproducible geometric property of the shape — the
seed-stability results above (both whole-shape and region) show a different noise
draw of the identical point cloud gives an ~orthogonal reported direction. The
`trustworthy` flag validates internal self-consistency for that one observation; it
says nothing about cross-seed reproducibility, which is a materially different (and
currently unmet) bar. This applies to every mode-arrow/mode-sweep figure in the
report, old or new.
**Next:** none required — this was the explicitly requested gap-closing pass. Genuine
open items, lower priority: repeat with a second sigma for exemplar diversity;
investigate whether ANY operator modification (larger regions? different masking?)
achieves cross-seed direction stability, since neither tested approach does.

## 2026-09-16 — Task 9 second half: whole-shape mode DIRECTIONS are noise artifacts; magnitude is not
**Who:** Claude (with Galit) · **Machine:** windows (CPU, ~15 min) · **Config:**
configs/gpu.yaml (`--override device=cpu`), new script `scripts/audit_seed_stability.py`
**What:** The deferred half of task 9 — stability under a second noise seed, and
agreement under point resampling — for the 15 shapes that were `trustworthy` at
sigma=0.02 in the n=50 run above. Two checks, both reusing that run's saved `x`/
`eigvecs` rather than recomputing everything (task 8's spirit):
1. **Seed stability** — same points `x`, a *new* noise draw at the same sigma.
   Point correspondence is exact, so eigenvectors are directly comparable.
2. **Resample stability** — same mesh, a *different* point sampling
   (`load_modelnet` with a different seed selects the identical 50 shapes by name —
   shape selection doesn't depend on seed, only per-shape point sampling does).
   Correspondence isn't preserved across resamplings, so only eigenvalue magnitude
   is compared here, not eigenvector direction.
**Result — a real, previously-uncharacterized finding: eigenVALUE magnitude is
stable; the reported eigenVECTOR DIRECTION for whole-shape spectra is not, and the
mechanism is exactly the "flat spectrum" already on record.**
- Seed stability (n=15, all previously trustworthy): **3/15 (20%) got REJECTED by
  the symmetry gate under just a new noise draw** — "trustworthy" is a real
  classifier, not a permanent property. Of the 12 that stayed trustworthy: median
  top-eigenvalue ratio vs. the original seed = **0.95** (stable) but median
  **top-mode cosine overlap = 0.041** and median **max subspace principal angle =
  89.4 DEGREES** — the entire top-5 eigenspace is essentially ORTHOGONAL between two
  noise draws of the exact same point cloud. Median reference eigenvalue spread
  (top-to-5th, at the original seed) = **16.6%** — a near-degenerate (flat) local
  spectrum, exactly the condition under which individual eigenvectors are
  numerically ill-defined and a small perturbation (a different noise draw) can
  rotate the reported eigenbasis arbitrarily within that near-degenerate subspace.
  This is a *mechanism*, not just a restatement, for Phase 3's "whole-shape spectra
  are nearly flat" finding: flat spectra don't just mean "the eigenvalues are
  similar," they mean **the specific reported eigenvector directions are not
  reproducible** — a materially stronger claim than has been made about this so far.
- Resample stability (n=15, same shapes): only **1/15 (7%) rejected** under
  resampling. Of the 14 that stayed trustworthy, median top-eigval/sigma^2 moved
  from 1.46 to 1.34 (~8%, comparable to ordinary shape-to-shape variation already
  documented at this sigma) — magnitude survives a different point sampling far
  better than it survives a different noise draw.
**Bug caught while building this:** `run_one()`'s first draft let a `ValueError`
from `top_eigenpairs` (a previously-trustworthy shape getting rejected under the
perturbation, which happened for 3+1 of the 30 total sub-runs) propagate and crash
the whole audit rather than being recorded as a result — the exact same class of bug
already fixed in `run_experiment.py`/`run_masked_modes.py`/`run_depth2d.py`, just in
new code. Fixed the same way (try/except, record, continue) before the full run.
**What this means for the report:** whole-shape "top uncertainty mode" *direction*
figures (if any exist) should not be presented as a stable geometric property of the
shape — only the top eigenvalue's magnitude (relative to sigma^2) should be. This is
an argument *for* Phase 3.5's region-restricted approach beyond "regions have more
spread" — regions may also have more *directionally stable* modes, though that
itself hasn't been tested (region spectra are less degenerate per Phase 3.5, so the
mechanism identified here predicts they should be more stable, but this session
didn't run that check — real compute, not just aggregation, see Next).
**Next:** the natural, well-motivated follow-up this finding predicts — repeat this
exact seed-stability check on `run_masked_modes.py`'s region-restricted eigenvectors
instead of whole-shape ones, to test whether less-degenerate regional spectra also
give more reproducible mode directions. Not done here (needs a full-scale masked-
modes reference run first, then the stability audit on top of it — a bigger lift
than "a small finish").

## 2026-09-16 — Tasks 7 & 8 & half of 9: reframed the report, made reproducibility real, added a composite trustworthy flag
**Who:** Claude (with Galit) · **Machine:** windows (CPU) · **Config:** n/a (docs +
tooling, plus re-running `build_results.py`/`archive_metrics.py` against the n=50
run above)
**What — task 7 (stop calling it calibration):** `results/README.md` §1 literally
said "The method is calibrated — until the denoiser leaves its training range."
Rewrote it (and `scripts/build_results.py`, which generates it) to "Sensitivity vs
noise level," with an explicit paragraph on why: Noise2Score3D is an unconditioned
score network with sigma manually substituted into Tweedie's formula, with no per-
sigma proof it's the MMSE denoiser there (`covariance_kind = frozen_pyramid_
sensitivity`). Added a matching caveat to `docs/PROJECT.md`'s math section (the
`Cov[X|Y=y] = sigma^2 * dD/dy` identity is exact only when `D` is verifiably MMSE at
that sigma — true for the toy denoisers, not established for Noise2Score3D). Added a
one-line pointer in §2 to the graph_topology_frozen finding.
**What — task 8 (reproducibility, not scale):** most of this already existed as a
side effect of this session's work (`make_out_dir` already saves resolved config,
seed is in it, diagnostics are saved per shape). Two real gaps closed: (1) added
`Noise2Score3DWrapper.checkpoint_sha256` (streaming SHA256, ~8s for the 293MB
checkpoint, computed once per wrapper construction) and wired it into
`run_experiment.py`/`run_masked_modes.py`/`check_denoiser.py`'s saved metrics as
`_provenance`; (2) ran `scripts/archive_metrics.py` to copy the n=50 run's raw
metrics.json into `results/metrics/raw/` (git-tracked) — this is exactly the step
that, skipped, is why every *other* historical run's raw data is marked "LOST" in
that directory's summary tables.
**What — task 9, first half only (composite trustworthy flag; second-seed/resampling
stability deferred, per explicit instruction):** added `pcuq.diagnostics.
is_trustworthy(m)` — not rejected by the symmetry gate AND convergence >=0.9 AND max
Ritz residual <=0.15 (thresholds picked from the actual n=50 residual/convergence
distributions, not round numbers). Wired into all three real-model scripts. This
turns three diagnostics someone previously had to cross-reference by hand into one
number: **86% / 72% / 16% trustworthy at sigma = 0.01 / 0.02 / 0.05** — sharper than
the rejection-rate framing alone, since it also catches the sigma=0.05 shapes that
pass the symmetry gate but never converged (58% of that sigma's total, on top of the
26% rejected outright).
**Bug found and fixed while doing this (real, not hypothetical):**
1. Regenerating `results/README.md` via `build_results.py` silently DELETED
   section 4's 6 exemplar galleries, because `outputs/masked/masked_modes/
   metrics.json` doesn't exist in this environment and the script had no fallback
   for "no fresh data, but a section already exists in the committed file" — it just
   wrote an empty section 4. Caught it via `git diff` before it landed; fixed the
   script to preserve the on-disk section 4 verbatim when it has nothing fresher, so
   this can't happen again to whoever runs it next.
2. `build_results.py`'s ablation chart had a stale fallback path
   (`outputs/local/run_experiment/metrics.json`, described as "early 10-shape grid")
   that, this session, actually held 2 shapes of REAL Noise2Score3D data from an
   earlier smoke test — with `graph_freeze_variant: topology` (the new default),
   i.e. FROZEN data, not the "graph rebuilt" data the ablation chart claims to show.
   `covariance_kind` alone couldn't tell frozen from unfrozen real-model runs apart
   (both are `frozen_pyramid_sensitivity`); regenerating would have silently
   overwritten `frozen_vs_rebuilt.png` with a frozen-vs-frozen chart mislabeled as
   frozen-vs-rebuilt. Caught via `git diff` on the image file before it landed (a
   plot without the same file also implies the same underlying misleading logic
   would trigger for anyone else running this script). Fixed by checking the run's
   sibling `config.json` for `freeze_graph: false` before trusting a directory as
   genuine unfrozen-ablation data, instead of guessing from `covariance_kind`.
**Result:** `pytest tests/` — 34 passed (4 new for `is_trustworthy`), no regressions.
`results/README.md` and `results/metrics/` regenerated correctly (verified via
`git diff` after each step, twice catching real data-loss/mislabeling bugs before
they landed — see above). `checkpoint_sha256 = 8e8765c6...` recorded for the n=50
run's exact weights.
**Next (explicitly deferred, per instruction):** task 9's second half — stability
under a second noise seed, agreement under point resampling — needs new compute
(~another full sweep), not just aggregation over what's already measured.
**Who:** Claude (with Galit) · **Machine:** windows (CPU, resumed from the n=15 run
below, ~80 more min) · **Config:** configs/gpu.yaml (`--override device=cpu`, all 50
shapes, 5 categories, sigma in {0.01, 0.02, 0.05}) — the full historical sample size.
**What:** Resumed the n=15 run below to the full 150 (shape, sigma) pairs (50 shapes
reused across all 3 sigmas, same deterministic `load_modelnet` seed). All 150
completed or were cleanly rejected — no crash.

| sigma | old median λ0/σ² (n=250 evs) | **new median λ0/σ²** | old n_neg | **new n_neg** | old conv. | **new conv.** | **rejected (shapes)** |
|---|---|---|---|---|---|---|---|
| 0.01 | 1.06 | **1.079** | 0% | **0.4%** | 0.999 | **0.9985** | **5/50 (10%)** |
| 0.02 | 1.39 | **1.396** | 0.8% | **1.0%** | 0.996 | **0.9960** | **8/50 (16%)** |
| 0.05 (beyond [0.004,0.034]) | 0.72 | **-0.533** | 76.4% | **84.3%** | 0.408 | **0.616** | **13/50 (26%)** |

**In-distribution (0.01, 0.02): even closer to the old headline at full sample size**
than the n=15 partial run suggested (1.079 vs the n=15 run's 1.10; 1.396 vs 1.41) —
essentially exact agreement with the pre-fix pipeline's 1.06σ²/1.39σ² now that sample
size matches. This is about as clean a confirmation as this kind of check gets.
**Rejection rate is real and now shows a clean monotonic trend with sigma** — 10% ->
16% -> 26%, tracking distance from the training range — which the n=15 subsample's
noisier 20%/27%/27% didn't clearly show (small-sample variance, not a different
result). **44% of the 50 shapes (22/50) get rejected by the symmetry gate at >=1
sigma**, including 10% at sigma=0.01, squarely in-distribution. Asymmetry on rejected
shapes: median 11.6%, range 5.7-91.9% (n=26 rejections total) — mostly borderline,
some severe.
**Correction to the n=15 entry below: sigma=0.05's apparent convergence recovery
(0.973) was a small-sample fluke, not the real picture.** At n=50, convergence at
sigma=0.05 is 0.616 median — better than the old pipeline's 0.408, but far from
clean. Worse: **65% of the shapes that PASS the symmetry gate at sigma=0.05 still
have convergence below 0.9** (min observed: 0.014 — essentially unconverged). The
symmetry gate and convergence are different checks catching different failure modes;
passing one doesn't imply the other. Net: at sigma=0.05, only a minority of shapes
(roughly (1-0.26)*(1-0.65) ≈ 26%) are both un-rejected AND well-converged — the
breakdown beyond the training range is worse and more pervasive than either the old
number (0.72, badly non-converged) or the n=15 partial run suggested.
**Result — unchanged conclusion, much stronger evidence:** the headline in-distribution
calibration (~1.06-1.4σ²) is confirmed almost exactly at matching sample size. The
out-of-range breakdown is confirmed and is worse than previously reported once
convergence is accounted for, not better. The previously invisible ~10-26%
rejection rate (rising with sigma) is a genuine, reproducible new finding, not a
sampling artifact — recommend reporting it explicitly rather than only quoting
convergence/antisym medians computed over the "passing" subset.
**Next:** decide how to handle sigma=0.05 in the report given ~74% of shapes are
either rejected outright or poorly converged there — quoting a single median
eigenvalue for that sigma without this context would be misleading either direction.
PLAN.md's depth2d re-tally and the sigma=0.03 breakdown-boundary run remain open.

## 2026-09-16 — Full-scale confirmation run (15 shapes x 3 sigma, topology-frozen default): matches Phase 3 in-distribution, sharpens the out-of-range finding
**Who:** Claude (with Galit) · **Machine:** windows (CPU, ~28 min wall time) · **Config:**
configs/gpu.yaml (`--override device=cpu data.n_shapes=15`) — real settings otherwise
(5 categories round-robin, sigma in {0.01, 0.02, 0.05}, n_ev=5, iters=15,
`graph_freeze_variant: topology`, the new default from the graph-freezing audit above).
**What:** First full-scale (N=2048) run since this session's fixes (Rayleigh-Ritz
finalization, symmetry-gate rejection, `graph_topology_frozen` default, crash-safe
rejection handling). 45 (shape, sigma) pairs attempted, all completed or cleanly
rejected — no crash, confirming the run_experiment.py fix above.
**Result — vs. the old (`full`-freeze, pre-task-2 eigensolver) Phase-3 headline**
(`results/metrics/summary/calibration_by_sigma.json`, 250 eigenvalues/sigma, 50
shapes x 5 categories):

| sigma | old median λ0/σ² | old range | old conv. | old antisym | **new median λ0/σ²** | **new range** | **new conv.** | **new antisym** | **rejected** |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 1.06 | [0.81, 1.59] | 0.999 | 0.001 | **1.095** | [1.05, 1.56] | 0.998 | 0.0008 | 3/15 (20%) |
| 0.02 | 1.39 | [1.12, 2.33] | 0.996 | 0.009 | **1.410** | [1.26, 1.74] | 0.996 | 0.0086 | 4/15 (27%) |
| 0.05 (beyond training range [0.004, 0.034]) | 0.72 | [-7, 7] | 0.408 | — | **-2.49** | [-5.46, 5.88] | **0.973** | 0.011 | 4/15 (27%) |

**In-distribution (σ=0.01, 0.02): the calibration headline replicates almost exactly**
despite the new default (`topology`, not `full`) and the stricter gate — median λ0/σ²
within 0.02-0.03 of the old figure, antisym within 0.001, convergence identical to 3
decimal places. n=0 negative eigenvalues at both sigmas (60 and 55 eigenvalues
respectively, all kept runs). This is the clean confirmation this run was for: the
1.06σ² / 1.39σ² headline is real and survives the pipeline's numerical corrections.
**Beyond training range (σ=0.05): still breaks down, but the number itself moved and
is now more trustworthy.** Old convergence was 0.408 (badly non-converged — those old
eigenvalues were themselves numerically suspect). New convergence is 0.973 — the
`topology` variant converges properly even out of range — and with trustworthy
numbers, the breakdown looks *slightly worse*: 52/55 negative eigenvalues (94.5%) vs
the old 191/250 (76.4%), median λ0/σ² actually negative (-2.49) rather than the old
0.72 (which averaged in a lot of non-converged noise). Net: **the "breaks down beyond
training range" finding is confirmed, not undermined** — the previous number was
noisier, not more correct.
**New finding, invisible in the old pipeline: 8/15 shapes (53%) get rejected by the
symmetry gate at at least one sigma** — including at σ=0.01, squarely inside the
training range (3/15, 20%). Measured relative asymmetry on the rejected shapes ranges
from borderline (7.8-9.6%, just over the 5% cutoff) to severe (74.5%, 91.9%). The old
eigensolver never rejected anything — these shapes used to silently contribute
possibly-meaningless eigenvalues into every historical aggregate (the 1.06/1.39
headline above, the 15-shape ablation, the masked-region gallery). Which shapes get
rejected is sigma-dependent, not purely shape-intrinsic (e.g. chair_0890 passes at
sigma=0.01 but is rejected at 0.02 and 0.05).
**Caveats:** n=15 shapes here vs n=50 in the historical sweep — medians are less
robust, and the two runs weren't necessarily built from the identical 15 shapes (both
use load_modelnet's seeded round-robin over the same 5 categories, so there's
category-level but not exact shape-level overlap with the historical n=50 run, whose
raw shape identities are themselves lost — see `ablation_frozen_vs_rebuilt.json`'s
"LOST" raw_artifacts note). Full-scale (50 shapes) confirmation is still open.
**Next:** run the full `python scripts/run_experiment.py --config configs/gpu.yaml
--override device=cpu` (all 50 shapes, ~112 min estimated) to get the rejection rate
and calibration numbers at the historical sample size before finalizing report
figures; decide whether to report the ~20-27% rejection rate as its own headline
number (it likely should be — it's a materially different picture from "convergence
0.999, antisym 0.001" read in isolation).

## 2026-09-16 — run_experiment.py/run_masked_modes.py would have crashed on the first full-scale run
**Who:** Claude (with Galit) · **Machine:** windows (CPU) · **Config:** configs/gpu.yaml
(`--override data.n_shapes=2 data.sigmas=[0.02] device=cpu`, calibrating timing before
a full run)
**What:** While timing a full-scale run to answer "how do I run the full pipeline,"
`chair_0890` (real ModelNet shape, N=2048, graph_topology_frozen) got rejected by
`top_eigenpairs`'s symmetry gate — and the script crashed outright, losing the shape
loop, exactly the same class of bug just fixed in run_depth2d.py (see entry below) but
never applied to the two primary 3D experiment scripts. Since metrics.json is only
written after a (shape, sigma) tag completes, this meant one bad shape anywhere in a
50-shape sweep would halt the run and keep halting on rerun (resume logic would just
walk back into the same rejection). Applied the same fix: `top_eigenpairs` wrapped in
try/except ValueError in both scripts, recording `"top_eigenpairs_rejected"` +
partial (mse-only) metrics for that (shape, sigma) or (shape, sigma, region) and
continuing, instead of losing the rest of the run.
**Result:** rerun of the same 2-shape calibration now completes cleanly: `chair_0890`
logged as rejected, `airplane_0627` succeeded normally (45.1s at gpu.yaml's real
n_ev=5/iters=15 settings). `pytest tests/` — 30 passed, no regression.
**Next:** none — this was blocking, now fixed. Full-scale run instructions below.

## 2026-09-16 — depth2d benchmark aligned with tasks 2-4; found a sharper failure mode
**Who:** Claude (with Galit) · **Machine:** windows (CPU) · **Config:** configs/local.yaml
(depth2d.dataset.n_shapes overridden to 4, then 10, for smoke checks)
**What:** Brought `scripts/run_depth2d.py` (the 2026-09-11 depth-map side quest) in
line with this session's spectrum/diagnostics changes: (1) wired
`top_eigenpairs(..., return_diagnostics=True)`, saving the same
`subspace_principal_angle_history_degrees` / `final_subspace_max_angle_degrees` /
`ritz_relative_residuals` fields `run_experiment.py` and `run_masked_modes.py` already
write (task 3); `covariance_kind` was already wired (task 4, prior session). (2) Task
2's Rayleigh-Ritz symmetry gate turned out to make the script crash outright on
out-of-domain shapes it used to just report a (possibly meaningless) number for —
wrapped the `top_eigenpairs` call in try/except so a rejected shape is recorded as
`"top_eigenpairs_rejected": <message>` and the run continues, instead of losing every
later shape to one bad one; matches this project's own standing policy of treating
breakdown as a finding, not an error (see the original 2026-09-11 LOG entries). Also
added a random-probe `antisym_energy` reading (autograd, n_probes=5) that's collected
*before* attempting `top_eigenpairs`, so a rejected shape still leaves a comparable
number behind instead of a gap.
**Bug found and fixed (unrelated to the above, uncovered by it):** `diagnostics.py`'s
`_unit_probes` hardcoded `.reshape(n, 1, 1)` when normalizing random probe directions
— correct only for rank-2 signals ((N,3) point clouds). The depth2d script never
exercised `antisym_energy`/`sweep_step_size` (only their `_fd` siblings, which take an
explicit V rather than generating one), so this was latent until the new
`antisym_energy` call above hit it on rank-3 (C,H,W) image tensors: silently
mis-broadcast into a wrong shape rather than an obvious crash. Fixed to reshape by
`y.dim()` generally. `pytest tests/` — 30 passed, no regression on the 3D path.
**Result — new finding, not a regression:** re-running depth2d locally (10 shapes,
MNIST, sigma=140.25/255) with the fixed pipeline: **3/10 shapes are now REJECTED
outright** (chair_0890, airplane_0627, lamp_0125 — antisym_probe 0.999-1.004, i.e.
Jv and J^Tv are essentially orthogonal, not mildly skewed) vs **7/10 succeed with
near-zero antisym** (0.000-0.002) as before, several still showing the previously-
documented negative-eigenvalue/low-antisym breakdown (airplane_0628, table_0394,
lamp_0126, guitar_0157). The old eigensolver (pre-task-2) never rejected anything —
it always produced *a* number via per-vector Rayleigh quotients on the arbitrary
QR-rotated basis, so the 3 now-rejected shapes previously contributed silently-
questionable eigenvalues to the 2026-09-11 aggregate stats (the ~50%-non-PSD,
44-52%-across-renderings headline). This doesn't overturn that finding — it sharpens
it into two distinct failure tiers: mild (non-PSD via low antisym, still reportable)
and total (antisym ~1, now correctly refused rather than silently reported).
**Next:** re-run the full depth2d sweep (50 shapes x 5 categories, matches gpu.yaml's
depth2d block) with the fixed pipeline and retally the non-PSD rate split by these two
tiers before the report repeats the old 44-52% figure verbatim; `n_rejected` per
category would make a clean addition to the category-breakdown table already planned
in PLAN.md's depth2d side quest section.

## 2026-09-16 — Graph-freezing audit: a weaker "topology only" freeze beats the current default on real shapes
**Who:** Claude (with Galit) · **Machine:** windows (CPU, no GPU needed) · **Config:**
configs/local.yaml (+ overrides, see below)
**What:** Task-6-style audit of `Noise2Score3DWrapper.graph_frozen()` as a deliberate
finite-difference surrogate, not automatically "the derivative of the model." First,
setup: fetched `external/Noise2Score3D` (git clone) and the 293MB checkpoint (curl) —
no GPU needed, both are plain downloads (docs/WORKFLOW.md), consistent with this
project's prior finding that the real model runs fine on CPU. That surfaced a missing
dependency, `open3d` (only used by their `kernel_points.py` to cache/read tiny KPConv
kernel-point dispositions as PLY files, all of which already ship pre-cached in the
repo) — shimmed it in `denoisers.py` (`_shim_open3d_if_missing`, same pattern as the
existing pykeops shim: implements just the narrow binary-PLY read/write path they
use, no external/ edits, no heavy binary dependency). This unblocked
`tests/test_phase2.py`'s two previously-skipped real-model tests (now 28/28 passing,
0 skipped) and `check_denoiser.py`.
**Investigated:** `build_grid_and_radius_graph_pyramid` (external/.../graph_pyramid.py)
rebuilds the whole voxel/radius-search graph from scratch every forward; a finite
difference y +/- c*v therefore risks flipping discrete assignments at the perturbed
point, producing O(1) jumps instead of a local derivative — this is why
`graph_frozen()` exists (freezes topology AND coarse-level point coordinates at the
anchor). Read `grid_subsample_pack_mode` (coarsening = an exact voxel-grid mean,
discrete group assignment + averaging) closely enough to implement the "if feasible"
third variant from the task list: `graph_topology_frozen()` — freezes discrete
voxel-group membership and all radius-search edges at the anchor (same as
`graph_frozen`), but recomputes coarse-level point COORDINATES as voxel means of the
perturbed level-0 input through that fixed membership, instead of freezing those
coordinates too. Implemented entirely in `denoisers.py` (`_voxel_group_ids`,
`_voxel_mean`, pointer comment to `grid_subsample.py`'s `ravel_hash_func`) — no
external/ edits. Verified exactness: at zero perturbation, `graph_topology_frozen`
reproduces `graph_frozen`'s coarse points bit-for-bit (max abs diff 0.0 across all 5
pyramid levels) — same math, different path, not an approximation of it. Also fixed
`check_denoiser.py`, which never froze the graph at all (a pre-existing bug that used
to silently pass under the old lenient Rayleigh-quotient eigensolver and now correctly
crashes under task 2's Rayleigh-Ritz symmetry gate) — now uses the same
freeze/freeze_variant logic as the experiment scripts.
**Result — `scripts/audit_graph_freezing.py`, 3 real ModelNet chairs, N=256, sigma=0.02**
(fixed tiny scale by design, not profile-driven — a correctness audit, not an
experiment): `regular` (no freezing) has no step-size plateau (self-consistency
0.63/0.81/5.4 at c=1e-3 across the 3 shapes) — confirms the jump-dominated failure
mode directly, consistent with the (raw-artifacts-lost) 2026-08-18 ablation summary
in `results/metrics/summary/ablation_frozen_vs_rebuilt.json` (antisym 0.34 vs 0.009,
n=15 @ full scale). New finding: **`graph_frozen` (current default) was REJECTED by
`top_eigenpairs`'s symmetry gate on 2/3 shapes** (relative asymmetry 7.3%, gate at
5%) — the current default itself is not reliably trustworthy on real shapes.
**`graph_topology_frozen` was never rejected (0/3)**, with comparable or better
step-size self-consistency at c=1e-3 (median 0.012 vs 0.011 — statistically a wash)
and a smaller, likely-less-inflated top eigenvalue (median 1.59 sigma^2 vs 2.14
sigma^2) — freezing coarse coordinates outright, not just topology, appears to
over-attribute local sensitivity to the level-0-only pathway.
**Decision:** `graph_topology_frozen` is the more stable choice — made it the new
default via a config knob, `denoiser.graph_freeze_variant: topology | full` (added
to both profiles). **This changes numbers for any new real-model run relative to
Phase 3/3.5's published figures** (e.g. the "calibrated at 1.06sigma^2 @ sigma=0.01"
headline used `full`-equivalent freezing) — override to `graph_freeze_variant: full`
for exact parity with pre-2026-09-16 results if the report needs to reproduce them
verbatim; a fresh run under `topology` is not a regression, just a different (more
defensible) choice of which function is being differentiated.
**Verified locally, no GPU:** `pytest tests/` — 30 passed (2 new regression tests:
zero-perturbation exactness, and that `graph_topology_frozen` converges + passes the
symmetry gate on a small anchor). `sanity_gaussian.py` PASS. `check_denoiser.py
--config configs/local.yaml` now PASS (previously crashed). `run_experiment.py
--override denoiser.kind=noise2score3d data.dataset=modelnet40` and
`run_masked_modes.py`, both with the real model and the new `topology` default, ran
end-to-end with no rejections.
**Next:** the 3-shape/N=256 audit is a smoke-scale signal, not the full-scale A/B
table Phase 3's "unfrozen ablation slice" follow-up still wants — worth a larger
(15-50 shape, N=2048) confirmation of `topology` vs `full` before quoting it as the
report's headline calibration number; likely still doable locally on CPU given
today's per-shape timings, no GPU required. Also open: task 1 (reframe the project's
"posterior covariance" claim for Noise2Score3D as sensitivity, not MMSE — this session's
`covariance_kind` taxonomy is the mechanism, the report language itself is still TODO)
and task 7 (stop treating the arbitrary-sigma sweep as calibration evidence).

## 2026-09-16 — Second toy prior: Gaussian mixture gives legitimate eigenvalues above sigma^2
**Who:** Claude · **Machine:** windows · **Config:** configs/local.yaml
**What:** `AnalyticGaussianDenoiser`'s ground truth (Cov[X|Y] = sigma^2 C(C+sigma^2 I)^-1)
always has eigenvalues <= sigma^2 for a single Gaussian prior — a limitation of the
toy, not a general property of posterior covariances, and worth correcting before it
gets over-read as "eigenvalues above sigma^2 mean something's wrong." Added
`pcuq.data.ToyGMM`/`make_toy_gmm`: a K-component Gaussian-mixture prior (each
component its own random eigenbasis, so genuinely non-Gaussian/multimodal, not a
rotated single blob) with closed-form posterior mean AND covariance via the law of
total covariance over Bayes responsibilities — `Cov[X|Y] = E_k[C_k] + Cov_k[m_k(Y)]`,
anchor-dependent unlike the single-Gaussian case. `pcuq.denoisers.AnalyticGMMDenoiser`
(`covariance_kind = exact_mmse_fixed_sigma`) implements the matching D(y) = E[X|Y=y]
exactly, fully vectorized (einsum/softmax) so `torch.func.jvp` traces through it —
this exercises the SAME jvp -> subspace-iteration -> Rayleigh-Ritz pipeline as the
real experiments, not a separate code path. At an anchor placed at the midpoint
between two component means (genuinely ambiguous mode membership), the top
eigenvalue legitimately exceeds sigma^2 even though every individual component's own
posterior covariance stays <= sigma^2 — mode-uncertainty, not a bug. Getting this
required tuning `amp` (within-component prior variance) well below sigma^2 so each
component's posterior mean stays near its own mu_k instead of collapsing toward y
(first attempt with amp=1e-2, mean_sep=0.03 failed — see commit history — because
high `amp` gives near-unity Tweedie gain, m_k(y)~=y for both components regardless of
separation, erasing the between-component term).
**Result:** 3 new pytest cases (`test_gmm_*`, tests/test_phase1.py) — denoiser output
matches the analytic posterior mean exactly (atol 1e-10); numeric spectrum matches
analytic ground-truth eigenpairs (rel err < 1e-3, |cos| > 0.999); top eigenvalue does
exceed sigma^2. `python -m pytest tests/ -q` — 26 passed, 2 skipped (pre-existing).
Also wired into `scripts/sanity_gaussian.py` as a second Phase-1 gate block (fixed
tiny size — dense eigh per component, not profile-scaled) alongside the existing
single-Gaussian check: `--config configs/local.yaml` → PASS, both toys 0.0% eigenvalue
error / |cos|=1.0000, GMM toy top eigval/sigma^2 = **5.42** (comfortably above 1.0).
**Next:** task 6 (codex's plan) — audit graph_frozen() as a deliberate finite-
difference surrogate, comparing regular vs. frozen-pyramid vs. fixed-topology finite
differences on the real Noise2Score3D model.

## 2026-09-16 — Denoisers/artifacts tagged with an explicit covariance_kind taxonomy
**Who:** Claude · **Machine:** windows · **Config:** configs/local.yaml
**What:** sigma^2 * J is only the exact posterior covariance when D is verifiably the
MMSE denoiser for a known prior at exactly that sigma; nothing in the code previously
stopped a figure or summary from treating every Jacobian spectrum that way. Added a
three-way `covariance_kind` classification (`pcuq.denoisers.COVARIANCE_KINDS`):
`exact_mmse_fixed_sigma` (closed-form ground truth — `AnalyticGaussianDenoiser`),
`approximate_mmse_fixed_sigma` (a network trained at/near a fixed, known sigma — the
reference paper's own `MNISTDenoiser2D`/`FFHQDenoiser2D`, used by the depth2d side
quest), and `frozen_pyramid_sensitivity` (no verified fixed-sigma MMSE target —
`Noise2Score3DWrapper`: an unconditioned score network with sigma manually substituted
into Tweedie's formula, differentiated through a frozen graph pyramid). Every
`Denoiser` subclass now sets this as a class attribute; `spectrum.top_eigenpairs`
raises `ValueError` up front if it's unset or not one of the three values, so a new
denoiser can't silently produce a spectrum with no label. All four scripts that save
spectrum metrics (`sanity_gaussian.py`, `run_experiment.py`, `run_masked_modes.py`,
`run_depth2d.py`, `check_denoiser.py`) now write `covariance_kind` into `metrics.json`
alongside each run's eigenvalues.
**Result:** `python -m pytest tests/ -q` — 23 passed, 2 skipped (pre-existing checkpoint
gap), including a new test that an unclassified denoiser is rejected by
`top_eigenpairs`. `sanity_gaussian.py --config configs/local.yaml` PASS.
`run_experiment.py --config configs/local.yaml --fresh` runs end-to-end;
`metrics.json` shows `"covariance_kind": "exact_mmse_fixed_sigma"` for both toy runs,
as expected for `AnalyticGaussianDenoiser`. `run_masked_modes.py`/`run_depth2d.py`/
`check_denoiser.py` were reviewed by inspection only (same pre-existing missing-
checkpoint gap as the 2026-09-16 convergence-diagnostics entry below) — their
`den.covariance_kind` wiring mirrors `run_experiment.py`'s.
**Next:** task 5 (codex's plan) — a 3D Gaussian-mixture toy prior with an analytic
posterior, which would be the first `exact_mmse_fixed_sigma` case with legitimate
eigenvalues above sigma^2 and a non-Gaussian ground truth.

## 2026-09-16 — Convergence diagnostics: principal angles + Ritz residuals added alongside existing overlap history
**Who:** Claude · **Machine:** windows · **Config:** configs/local.yaml
**What:** Continuation of the Rayleigh--Ritz fix below. Per-vector overlap alone is
unreliable when the top subspace rotates internally or eigenvalues are close, so
`top_eigenpairs(..., return_diagnostics=True)` now also reports, without removing the
existing `history` overlap list: (1) principal angles (degrees, via SVD of the
Gram matrix) between each iteration's subspace and the previous one, and (2) per-mode
relative Ritz residuals `||sigma^2 J v - lambda v|| / max(||sigma^2 J v||, |lambda|)`
from the finalized Rayleigh--Ritz eigenpairs. `run_experiment.py` and
`run_masked_modes.py` both opt in and save `subspace_principal_angle_history_degrees`,
`final_subspace_max_angle_degrees`, and `ritz_relative_residuals` into `metrics.json`
and the per-run `.pt` bundle (`spectrum_diagnostics`), alongside the untouched
`final_iter_overlap`.
**Result:** `python -m pytest tests/ -q` — 22 passed, 2 skipped (pre-existing: real
Noise2Score3D checkpoint not present in this checkout). `sanity_gaussian.py` PASS
(rel_err 0.0%, |cos|=1.0). `run_experiment.py --config configs/local.yaml` runs
end-to-end; sampled metrics.json shows sane values (e.g. final_subspace_max_angle
0.0-0.03deg, ritz_relative_residuals ~3e-4, matching the near-1.0 overlap already
reported). `run_masked_modes.py` was not exercised end-to-end here — it requires the
real `Noise2Score3DWrapper`, and `external/Noise2Score3D` (distinct from the vendored
`external/GaussianDenoisingPosterior`) isn't checked out in this environment; this is
a pre-existing setup gap, not something this change touched, and the wiring in that
script mirrors `run_experiment.py`'s (reviewed by inspection).
**Next:** decide a trustworthy-mode threshold (e.g. max angle + max residual both
below some bound) if/when a mode-filtering step is added; currently the diagnostics
are surfaced but not yet used to gate anything downstream.

## 2026-09-16 — Spectrum extraction corrected with Rayleigh--Ritz
**Who:** Codex · **Machine:** windows · **Config:** not run (CPU-only unit-test change)
**What:** Replaced the final diagonal-Rayleigh-quotient heuristic in
`pcuq.spectrum.top_eigenpairs` with Rayleigh--Ritz diagonalization of the converged
subspace. Added coverage for a deliberately rotated invariant subspace, symmetric
indefinite operators, nearly repeated eigenvalues, and rejection of materially
nonsymmetric projected operators when presented as covariance spectra.
**Result:** The implementation and tests are in place. Validation is pending because
this checkout currently has no callable Python interpreter (`python`, `py`, and `uv`
are absent from PATH), so no test result is recorded yet.
**Next:** Run `python -m pytest tests/ -q` locally before any real-model or GPU work.

## 2026-09-11 — Depth-map 2D benchmark: scaffolded, smoke-tested, first (striking) result
**Who:** Claude (with Galit) · **Machine:** windows (fresh checkout, CPU) · **Config:**
configs/local.yaml (depth2d block)
**What:** New side quest: render ModelNet40 shapes to depth-map images and run them
through the reference paper's OWN 2D denoiser (MNIST CNN / FFHQ DDPM, already vendored
code in `denoisers2d.py`) instead of Noise2Score3D — testing the original method out of
its training domain on a depth-map "photo," as a cross-domain comparison against our
in-domain 3D result. Added `pcuq.depth.render_depth_map` (single-view orthographic
z-buffer via `scatter_reduce`, hole-filling via max-pool dilation — pure synthetic unit
tests, no download needed), `scripts/run_depth2d.py`, and a `depth2d:` config block in
both profiles (local: MNIST CNN, no extra download; gpu: FFHQ DDPM, needs `ffhq.pt`).
This machine started as a fresh Windows checkout (`external/`, `data/`, `outputs/` all
empty gitignored scratch); did the one-time setup: `pip install -r requirements.txt`,
cloned `external/GaussianDenoisingPosterior`, ran `pytest tests/` (18 passed, 2 skipped
— Noise2Score3D checkpoint still not downloaded, unrelated to this side quest).
**Bug found & fixed:** `run_depth2d.py`'s first draft copied `run_images2d.py`'s viewer-
bundle merge (`results/viewer_data_2d.json["runs"]`) — that key doesn't exist on the
file actually on disk (`{"images": [...]}`, written by `run_fullspectrum2d.py` in a
different, quantized/base64 schema that `viewer_template.html` actually reads).
`run_images2d.py` itself would crash the same way if run today — a pre-existing,
unrelated staleness, not something this change caused. Fixed by not writing into that
shared file at all (out of scope for this step); `run_depth2d.py` now only writes its
own `outputs/<profile>/run_depth2d/` artifacts (metrics.json, per-shape .pt, sweep PNG).
**Result (2 chairs, σ=0.02, MNIST CNN, single -Z view, 28x28):**
`chair_0890`: eigvals [4.09, 1.17] — **13.5σ²** top mode, converged (overlap ~1.0/0.97).
`chair_0891`: eigvals [-2.45, -13.27] — **both negative** (non-PSD), antisym 0.73 (vs
≤0.01 typical in-domain). The denoised anchor (t=0 in `chair_0890_mnist_sweep.png`) is
nearly uniform black — the MNIST CNN collapses an unfamiliar depth-map "digit" toward
its training prior (mostly-background canvas) instead of reconstructing the shape, yet
the estimated uncertainty modes still carry large, structured off-anchor swings. This
looks like the domain-shift breakdown the side quest set out to find: extreme/negative
eigenvalues and high antisymmetry, the same signature Noise2Score3D showed beyond its
training σ range, now triggered by domain shift instead of noise-level extrapolation.
Two shapes is not a claim, just a promising first look.
**Next:** More shapes/categories to see if this is systematic; try `axis`/`dilate`
render knobs to see if a "nicer" depth map changes the picture; decide whether to chase
`ffhq.pt` (source URL not recorded anywhere in the repo — a separate pre-existing gap)
for a natural-image comparison point.

## 2026-09-11 — Depth-map benchmark scaled up (15 shapes, 5 categories): breakdown is systematic, and it's PSD-violation, not asymmetry
**Who:** Claude (with Galit) · **Machine:** windows (CPU) · **Config:** configs/local.yaml
(depth2d: n_shapes 2→15, categories chair→[chair,airplane,table,lamp,guitar] mirroring
gpu.yaml's set; also fixed `spectrum.iters` 8→25 — `run_fullspectrum2d.py` needed 40
iters for this same MNIST CNN, 8 was copied from the unrelated FFHQ example and
under-converged 2/15 runs, min overlap as low as 0.07)
**Decision (asked & answered):** keep this training-free/frozen-model, don't train a
depth-map-specific denoiser — training would abandon the project's core "zero training
compute" premise for a much bigger side project, and the domain-shift framing already
extends the project's real finding (Phase-3: calibrated in-range, breaks down out of
training range) onto a *second axis* (wrong domain, not wrong σ) — a stronger report
point than a well-fitted bespoke model would give.
**Result (15 shapes, σ=0.02, MNIST CNN, single -Z view, 28x28, now iters=25):**
convergence is no longer a confound (min per-run overlap: median 1.000, range
[0.63, 1.00], vs [0.07, 1.00] at iters=8) — the breakdown below is real, not
under-convergence. **7/15 shapes have at least one negative eigenvalue, 3/15 both
negative**; top eigval/σ² median 13.5, range [-8.7, 184] (vs 1.06–1.39 in-domain for
Noise2Score3D). **Antisym energy is mostly LOW even where eigenvalues are negative**
(median 0.001; e.g. airplane_0628 and guitar_0158 are both fully negative with antisym
0.000-0.001) — unlike the 3D noise-extrapolation breakdown, where negative eigenvalues
co-occurred with HIGH antisym (~0.3-0.5, unfrozen graph rebuilds). Reading: the
Manor-Michaeli symmetric-PSD covariance identity assumes D is (close to) the true MMSE
denoiser for the actual data distribution feeding it; nothing guarantees that when D is
a digit denoiser fed a chair's depth map — the negative eigenvalues here look like an
honest consequence of asking an invalid question, not a numerical artifact, and the
existing antisym-energy diagnostic alone would have missed most of these (a genuinely
different failure signature from the in-project noise-range breakdown, worth its own
report paragraph rather than being folded into the same "high antisym = untrustworthy"
story).
**Next:** qualitative figures for a few representative shapes (`outputs/local/run_depth2d/*_sweep.png`);
decide whether `ffhq.pt` is worth chasing for a natural-image comparison point.

## 2026-09-11 — Depth-map benchmark: scaled to 50 shapes + render-choice robustness check
**Who:** Claude (with Galit) · **Machine:** windows (CPU) · **Config:** configs/local.yaml
(depth2d.dataset.n_shapes 15→50, matching gpu.yaml's shape count; 99s wall time, still
local/CPU) + two `--override` robustness variants (`depth2d.render.axis=0`,
`depth2d.render.dilate=0`), each also 50 shapes, ~100s.
**What:** User asked "is this enough or dig deeper" — answered by (1) scaling the
n=15 smoke result to n=50 (project's own bar for "systematic", per Phase-3), and (2)
a negative-control robustness check: does the breakdown depend on my depth-map
rendering choices (viewing axis, hole-filling), or is it about the domain shift itself?
**Result — robust across renderings (headline claim):** fraction of shapes with >=1
negative eigenvalue stays in a narrow band regardless of rendering: baseline
(axis=2,dilate=1) 25/50=50%, axis=0 26/50=52%, dilate=0 22/50=44%; median antisym
stays ~0.000-0.001 in all three. This is the part safe to state as a general finding: **~half of ModelNet40
depth maps produce a non-PSD implied covariance under the frozen MNIST CNN, via a
mostly-low-antisymmetry mechanism** (unlike the project's noise-range breakdown, which
is high-antisymmetry) — not an artifact of axis or hole-filling choice.
**Result — NOT robust across renderings (report with the caveat attached):** the
*magnitude* of the breakdown is rendering-sensitive. Median top eigval/σ²: 8.9
(baseline) → 25.0 (axis=0) → **86.3 (dilate=0)**. Fully-negative-spectrum rate: 16%
(baseline) → 12% (axis=0) → **2% (dilate=0)**. Removing hole-filling (sparser, more
speckled depth maps — further from anything MNIST-like) makes single eigenvalues
much larger and more erratic, but less likely to leave BOTH eigenvalues negative.
Category breakdown at n=50 (baseline): table 2/10, lamp 4/10, chair 7/10, guitar 6/10,
airplane 6/10 — thin/complex shapes break more often than blocky ones, a plausible
secondary story (worth a category-level figure if this goes in the report).
**Conclusion:** enough to write up. Report the qualitative pattern (roughly half of
out-of-domain shapes break, via low-antisym non-PSD, distinct mechanism from the
in-project noise-range breakdown) as the finding; report the numeric severity
(median ratio, fully-negative %) as rendering-choice-dependent, not a fixed constant —
don't quote a single number without naming which render config it came from.
**Next:** category-breakdown figure; pick 3-4 representative shapes across the
severity range for report figures; `ffhq.pt` chase remains optional/deferred.

## 2026-09-04 — Aggregate metrics archived into results/metrics/ (outputs/ was empty)
**Who:** Claude · **Machine:** mac · **Config:** — (no experiment run)
**What:** `outputs/` is gone (disposable scratch, as designed) along with `data/` and
`external/`, so every number not already in `results/` would have cost a ~4.5GB
re-download plus hours of recompute — which silently blocked the remaining Phase-4
analysis (R6 density check, failure-case tables). New `scripts/archive_metrics.py`
makes the aggregates durable in three provenance-labelled tiers: `raw/` (verbatim
`metrics.json` copied from `outputs/` — empty today, populated by any future run),
`derived/` (per-run tables rebuilt from the committed viewer bundles, no compute
needed), `summary/` (medians transcribed from results/README.md and LOG.md for runs
whose raw artifacts are already lost). 79 kB total, git-tracked.
**Result:** 81 region runs + 3 whole-shape rank-24 spectra + 9 2D runs recovered with
per-run eigenvalues, spread, PSD/convergence flags and **local point spacing** (the R6
input). The derived table independently reproduces the LOG's own figures — 51 frozen /
30 naive runs, 11 naive runs with negative eigenvalues, naive median λ₀ 4.0σ² — so the
recovery is faithful. Also added a mode-**concentration** metric (participation ratio,
computed from the stored eigenvectors): it is uncorrelated with eigenvalue spread
(r=0.06), so `build_results.py`'s spread-ranked exemplar selection does not select for
interpretable modes.
**Two things this surfaced, both unresolved:**
- **R6, partially answered at region scale:** corr(relative local spacing, λ₀/σ²) =
  +0.20 over 51 regions (+0.13 @ σ=0.02, +0.21 @ σ=0.03). Weak — density does not look
  like the explanation. Caveat: this is *across regions at fixed σ*, so it does not
  directly test the σ-trend mechanism; the per-shape sweep data needed for that is lost.
- **Tension with the flat-whole-shape claim:** whole-shape top-5 spread @ σ=0.03 is
  10.5% (chair) / 11.0% (airplane) but **50.1% (lamp)**, against the reported
  "whole-shape 1–8%". Different run (`fullspectrum`, k=24) and no convergence data was
  exported for that bundle, so it is not yet a contradiction — but it needs resolving
  before the report repeats the 1–8% figure.
**Next:** report writing is no longer gated on recompute. Open: propagate the
log-concavity/MNIST correction into results/README.md and the published page; re-pick
gallery exemplars on concentration as well as spread; resolve the lamp spread above.

## 2026-08-18 — Viewer finalized: 9 images, draggable/resizable masks everywhere, review-agent audit
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** run_fullspectrum2d.py
**What:** Iterated the viewer to its final form based on user feedback: (a) 2D tab is
one chip per image — 5 MNIST digits *selected for recognizability* (min ‖x̂−x‖ per
class over 400 test digits: labels 0/3/5/7/8) + 4 faces (213/227/34/514), each with a
movable box mask (drag to move, corner handle to resize, saved per image), modes from
rank-k whole-image spectra recomputed client-side; (b) 3D custom mask draggable (✋
Move) with live recomputation; (c) hover tooltips with local per-point/pixel σ
(±% · ×noise); (d) "model confidence" stat tile (calibrated/elevated/ambiguous badge).
An independent review agent audited the page against the requirements: 12 findings
(1 crash, 3 correctness) — all fixed, including a subtle heatmap index misalignment
between separately-sampled clouds (nearest-neighbor align map). A Node DOM-shim
click-sweep test now gates every publish. Fixed along the way: viewer init crash
(tick before build), zsh word-splitting bug that silently dropped digit spectra.
**Result:** published (same artifact URL), 9.3MB page, all data measured
(outputs/fullspectrum2d, outputs/masked-grid*, outputs/fullspectrum).
**Next:** the report.

## 2026-08-18 — Paint-your-own masks in the viewer (rank-24 client-side eigensolver)
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** run_fullspectrum.py
**What:** Free-form mask selection without server compute: precompute the top-24
whole-shape eigenpairs per shape (`scripts/run_fullspectrum.py`, chair/lamp/airplane
@ σ=0.03, int16-quantized in `results/viewer_data_free.json`); for any painted mask M
the masked covariance is exactly (MV)Λ(MV)ᵀ on the rank-24 approximation, so its
modes reduce to a 24×24 Jacobi eigenproblem solved in the browser. Math validated
headlessly in Node against the real bundle (masked μ₀ ≤ whole-shape λ₀ ✓). Viewer:
"✏ custom" region chip → brush-paint points, Compute modes → sweepable like any run,
labeled "rank-24 reconstruction" with a pointer to the exact-modes script.
**Next:** report writing.

## 2026-08-18 — Faces fixed with exact gradients; corrections to two earlier claims
**Who:** Claude (with Rocky) · **Machine:** mac (CPU) · **Config:** run_images2d + bp
**What:** User reported the face tab showed no visible change. Root-caused and fixed.
**Diagnosis chain:** (1) viewer rendering verified correct (replayed its exact math in
Node); (2) face modes were computed with float32 central differences — noisy, and
power iteration amplifies the noise into spiky directions; (3) switching to
`torch.func` autograd silently crashed: guided_diffusion routes every UNet block
through a custom `CheckpointFunction` that functorch can't trace and whose backward
differentiates w.r.t. our frozen params. Fixes: runtime bypass of their checkpointing
(no vendored edits) + new `bp` method in `jacobian.py` (plain reverse-mode `Jᵀv`,
the reference repo's own backprop scheme). 13s/product on CPU.
**Results (exact gradients):** t=400 (σ≈2.0, their demo regime): broad semantic
modes — mouth λ₀=14.5=3.5σ² over ~1250 px (visible smile/expression morphing),
eyes 1.2σ²; antisym ≤0.05; converged. t=100 (σ≈0.34): small localized modes
(λ≈1.2–1.3σ²) — noise level controls uncertainty extent, matching their site.
**Corrections to earlier entries:** (a) an earlier message reported t=100 face modes
as "autograd-confirmed" — that run had crashed and stale finite-difference data was
read; only now are face modes exact. (b) The MNIST "λ=172σ² multimodality" claim:
with exact JVPs the top modes are 1–6 *single-pixel spikes* — a real pathology of
their MNIST checkpoint's Jacobian, not digit-identity uncertainty; structured digit
modes appear in lower modes (k=6 run). The Brascamp–Lieb point (σ² bound only for
log-concave priors) stands mathematically; the t=400 mouth mode (3.5σ², broad,
clean) is now the honest multimodality evidence.
**Viewer:** republished (same URL) — 2D tab now has both noise levels with exact
modes, display gain applies to images too.

## 2026-08-18 — Two-domain comparison: original method reproduced (MNIST + DDPM faces); viewer v2
**Who:** Claude (with Rocky) · **Machine:** mac (CPU) · **Config:** scripts + overrides
**What:** Ran the reference paper's own domain through OUR pipeline (`denoisers2d.py`,
`run_images2d.py`; `spectrum` masks generalized to image shapes): their bundled MNIST
CNN and their DDPM FFHQ denoiser (ffhq.pt, 2.2GB, from ddpm-segmentation). Also ran
the naive-port baseline on the 6-region mask grid, and rebuilt the interactive viewer.
**Results:**
- **MNIST:** digit-identity modes with λ up to 172σ², perfectly converged —
  reproduces the paper's multimodality behavior and **corrects our framing: σ² is a
  bound only for log-concave priors (Brascamp–Lieb); real multimodal data may
  legitimately exceed it.** The 3D 1.4–1.9σ² values may be genuine geometric
  ambiguity, not miscalibration. (One digit showed antisym 0.82 — near-boundary
  pathology, worth a report footnote.)
- **DDPM faces (eyes/mouth masks, t=100):** semantically meaningful modes — eyebrow
  position/thickness, eye shape — the paper's signature figure from our code
  (`outputs/images2d/ffhq/`).
- **Naive-baseline mask grid (30 regions, σ=0.03):** 3/30 converged, 11 with
  negative eigenvals, median λ₀ 4σ² — vs ours 51/60 converged. The frozen-graph
  contribution as a per-region A/B.
- **Viewer v2 (same artifact URL):** two tabs (3D ours / 2D original domain),
  6 movable region masks per shape, estimator toggle (ours vs naive), display-gain
  slider, oscillation on by default, legends + provenance.
**Next:** report writing; fold the log-concavity correction into results/README.

## 2026-08-18 — Comprehensive evidence set complete; results/ folder built
**Who:** Claude (with Rocky) · **Machine:** mac (CPU) · **Config:** local + overrides
**What:** Three background runs to make the report evidence comprehensive, plus a
curated, git-tracked `results/` folder (`scripts/build_results.py` regenerates it:
summary charts, auto-picked exemplar galleries, README with computed tables).
**Results:**
- **Calibration σ-trend (50 shapes each):** λ₀/σ² median 1.06 (σ=0.01) → 1.39
  (0.02) → **1.91 (0.03**, convergence 0.93, 19 neg eigvals — the transitional
  regime approaching the training edge 0.034**)** → breakdown (0.05). A clean
  monotone degradation curve for the report.
- **Unfrozen ablation at 15 shapes (σ=0.02):** median λ₀/σ² 2.91, antisym 0.34,
  convergence 0.18 — vs frozen 1.39 / 0.009 / 1.00. The A/B table is now real
  statistics, not a 5-shape anecdote.
- **Masked gallery, 60 region runs:** σ=0.02: 28/30 converged+PSD, spreads up to
  33%; σ=0.03: 22/30, median spread 12%, up to 31%. Arrow-field figures make modes
  directly interpretable (e.g. chair_0891 r0: mode = coherent shift of the thin
  stretcher rail between the legs).
**Where things stand:** all proposal commitments are demonstrated and quantified;
`results/README.md` is the at-a-glance summary. Remaining work is the write-up
itself (report text, figure selection from results/, discussion of the 1.3–1.9σ²
calibration drift and the training-range boundary).

## 2026-08-17 — Phase-3 sweep complete: calibrated in-distribution, breaks beyond training σ
**Who:** Claude (with Rocky) · **Machine:** mac (CPU) · **Config:** local + overrides
(name=phase3: 50 shapes × σ∈{0.01,0.02,0.05} × 5 eigenpairs, 15 iters, frozen graph)
**What:** Full-scale sweep, 150 runs in ~75 min, `outputs/phase3/run_experiment/`.
**Result (per σ, medians over 50 shapes):**
- **σ=0.01: top eigval 1.06σ² [0.81, 1.59], 0 negative eigvals (of 250), convergence
  0.999, antisym 0.001.** Essentially at the exact-MMSE bound — the method is
  *calibrated* in-distribution.
- **σ=0.02: 1.39σ² [1.12, 2.33], 2 negatives, convergence 0.996, antisym 0.009.**
  Mild inflation, growing with σ.
- **σ=0.05: breakdown** — eigvals scattered [-7σ², +7σ²], 191/250 negative, median
  convergence 0.41. Explanation found in their code: the model was trained with
  σ annealed over **[0.004, 0.034]** (`models/KPconv.py:159`); σ=0.05 is ~50% beyond
  the training range, so the score field (and its Jacobian) is extrapolating.
  Denoising MSE still improves there (1.35×) — the *mean* extrapolates better than
  the *derivative*, a nice report point.
The earlier "~1.3σ² anomaly" is now a clean σ-trend: 1.06 → 1.39 → breakdown as σ
approaches/exceeds the training range. Interpretation: score-Jacobian calibration
degrades near the edge of the amortization range.
**Next:** Report material is essentially complete: calibration-vs-σ table, the
frozen-graph A/B, mode galleries. Optional: rerun σ=0.05 → 0.03 (inside training
range) to show the breakdown boundary; unfrozen ablation slice for the A/B table.

## 2026-08-17 — MPS measured: works, but slower than CPU for the real model
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** —
**What:** Tested whether the real denoiser can use Apple's GPU (MPS) instead of CPU.
Required generalizing the `.cuda()` shim from "no-op" to "redirect to the wrapper's
device" (also fixed a shim bug: second wrapper construction crashed on the spec-less
pykeops stub). No `PYTORCH_ENABLE_MPS_FALLBACK` needed.
**Result (N=2048, 5-forward average):** CPU 244 ms/forward, MPS 344 ms/forward —
MPS is ~1.4× *slower*: this KPConv pipeline is many small irregular kernels
(radius search, gather/scatter, pack-mode reshapes), which MPS launch overhead
dominates; Apple GPUs win on big dense matmuls, not this. Outputs agree to ~3% of
output std (kernel differences). Policy: `device: auto` picks CPU for the real model
(explicit `device: mps` is honored); the analytic/toy path keeps using MPS, where its
dense matmuls do win.
**Next:** —

## 2026-08-17 — First real ModelNet results; graph-rebuild discontinuity found & fixed
**Who:** Claude (with Rocky) · **Machine:** mac (CPU) · **Config:** local + overrides
**What:** Ran the first real experiment locally (ModelNet40 downloaded in ~2 min):
10 shapes (chair/airplane/table/lamp/guitar ×2) × σ∈{0.01,0.02,0.05} × 5 eigenpairs,
~10s per run, `outputs/local/run_experiment/`. Results were bad in an instructive way:
top eigenvalues 4–36× above the σ² MMSE bound, negative eigenvalues, subspace antisym
energy ~0.4–0.5, poor convergence, and **no step-size plateau** (40–80% c-halving
error everywhere). Diagnosis: the model rebuilds its voxel/radius graph every forward,
so perturbed passes flip discrete assignments — finite differences measure O(1) jumps,
not the local derivative.
**Fix:** `Noise2Score3DWrapper.graph_frozen()` — freeze the anchor's graph pyramid
(discrete indices + coarse-level positions), let only input points vary: differentiate
the *smooth branch*. A/B on 5 shapes @ σ=0.02 (`outputs/local-frozen/run_experiment/`):
eigvals all positive, tight 1.29–1.45σ² (was median 7σ², max 32σ²); overlaps ~0.99;
antisym energy 0.001–0.017 (was ~0.4); sweep shows a clean U with plateau at c=1e-3
(now the config default; error 2.2e-3 there). `freeze_graph: true` is the default in
both profiles. Mode figures show localized structure (lamp arm / base as separate
modes). Remaining anomaly: eigvals consistently ~1.3σ², slightly above the exact-MMSE
bound — see PLAN.md open questions.
**Next:** Full gpu.yaml sweep (VM or overnight Mac), quantitative eigval-vs-σ tables
via `scripts/summarize_results.py`, mode-figure gallery for the report.

## 2026-08-17 — GPU-ready: ModelNet40 loader, full run_experiment pipeline, VM runbook
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** configs/local.yaml
**What:** Closed the gaps between "gates pass" and "VM can run the real experiment":
ModelNet40 loader (official zip, pure-torch OFF parse + area-weighted sampling —
dropped the trimesh dep), fully implemented `run_experiment.py` (shapes × σ, spectrum,
per-σ diagnostics incl. new autograd-free step-size sweep, mode figures, incremental
metrics.json), matplotlib viz, VM setup runbook in WORKFLOW.md. requirements.txt alone
now suffices on the VM (no pykeops/pytorch3d).
**Result:** 13 tests pass. Local runs of `run_experiment.py`: analytic+toy PASS (19s,
MPS, eigval overlap 0.9999); real-model smoke PASS mechanically (CPU). Smoke caught and
fixed a real bug: coarse pyramid levels with fewer support points than neighbor_limit
broke the kNN shim (now inf-padded = "no neighbor"). Finding: on the OOD toy blob the
real model's top |eigvals| are *negative* (non-PSD, as the proposal warned) — logged as
an open question to re-check on in-distribution ModelNet shapes.
**Next:** On the VM: WORKFLOW.md runbook top to bottom; first `gpu.yaml` run downloads
ModelNet40 (~2GB). Expect ~10s/(shape·σ) on CPU-scale timing — much less on GPU.

## 2026-08-17 — Phase 2 (nearly) done: real denoiser runs ON THE MAC; equivariance gate passed
**Who:** Claude (with Rocky) · **Machine:** mac (CPU) · **Config:** configs/local.yaml
**What:** Wrote `Noise2Score3DWrapper` + `scripts/check_denoiser.py`. Made the vendored
model run without CUDA/pykeops via runtime shims (no vendored files edited): no-op
`.cuda()` when CUDA is absent; swap their pykeops kNN for exact `cdist`+`topk`; their
`dataloader()`'s hardcoded `.cuda()` replaced device-safely; checkpoint loaded with
`strict=False` guarded to allow only the zero-init conv biases the checkpoint lacks.
**Result (N=2048, σ=0.02, sphere):** forward 0.2s CPU; **equivariance 1.7e-8 — the
proposal's hard gate PASSED**; denoising improves MSE 4.02e-4 → 3.32e-4; 2-eigvec
spectrum in ~6s (finite differences). At N=256 denoising *hurt* MSE — the model needs
training-like density (trained on 10k–50k-pt ModelNet), hence local n_points now 2048.
Two surprises → PLAN.md open questions: `torch.func` autograd cannot trace their graph
ops (finite differences only — sweep needs an autograd-free reference), and estimated
eigenvalues ~1.5σ² exceed the MMSE bound σ². Sanity gate re-verified at N=2048 (PASS,
15s, MPS).
**Next:** ModelNet40 loading (last open Phase-2 item), then Phase-3 sweeps on the VM.

## 2026-08-17 — Noise2Score3D availability confirmed; Phase-2 blocker cleared
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** —
**What:** Verified the primary denoiser exists publicly: official ICCV 2025 code at
github.com/Bobby645/Noise2Score3D with pretrained weights on Hugging Face
(bobby645/Noise2Score3D). Vendored at `external/Noise2Score3D/` (commit `fef67d7`).
**Result:** KPConv-based, trained on ModelNet-40 — exact match to our proposal. Two
caveats: no upstream license file (note in report), and heavy deps (PyTorch3D,
pykeops, CUDA-era pins) → the real denoiser likely runs only on the GPU VM; Mac smoke
runs keep the analytic denoiser. ScoreDenoise fallback no longer needed but stays
documented in SOURCES.md.
**Next:** Download checkpoint, stand up their env on the VM, then the
ordering-preservation gate before trusting any spectra.

## 2026-08-17 — Phase 1 complete: toy pipeline validated, gate passes
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** configs/local.yaml
**What:** Implemented the full Phase-1 machinery: `ToyGaussian` prior with closed-form
posterior, `AnalyticGaussianDenoiser`, JVPs (forward/central/autograd), subspace
iteration with Rayleigh-quotient eigenvalues, and the diagnostics suite. 7 tests pass.
**Result:** `sanity_gaussian.py` at local scale (N=256, k=2, 10 iters, MPS, float32):
PASS in ~3.5s. Eigenvalue rel err 0.2–0.3%, eigenvector |cos| 0.995, antisym energy
1e-7. Step-size sweep on float32 shows error *rising* as c shrinks (1e-2 → 3.6e-4 err;
1e-5 → 0.36) — pure cancellation, since the toy denoiser is linear (no nonlinearity
penalty at large c). Metrics: outputs/local/sanity_gaussian/metrics.json.
**Next:** For *real* (nonlinear) denoisers the sweep will be U-shaped; expect the
float32 cancellation floor to matter and double precision (CPU/CUDA only — MPS has no
float64) to be needed for small c. Phase 2: verify Noise2Score3D availability.

## 2026-08-17 — Project scaffolded
**Who:** Claude (with Rocky) · **Machine:** mac · **Config:** —
**What:** Created the docs system, package skeleton, and scale-profile configs. Vendored
`GaussianDenoisingPosterior` (shallow clone) into `external/`.
**Result:** Structure as described in [CODE_STRUCTURE.md](CODE_STRUCTURE.md). Read of
`moments_calculations.py:get_eigvecs` confirms the reference algorithm is subspace
iteration with forward-difference JVPs (`c=1e-6`, eigenvalue `‖Jv‖σ²/c`, QR
re-orthonormalization) — directly portable to `(B,N,3)` tensors.
**Next:** Phase 1 (toy Gaussian pipeline) per [PLAN.md](PLAN.md); verify Noise2Score3D
code availability early (Phase 2 blocker).
