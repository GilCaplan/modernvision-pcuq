"""The proposal's three challenge checks, as runnable diagnostics.

Planned API (each returns plain numbers for the run's metrics JSON and LOG.md):

- check_equivariance(denoiser, y): does the denoiser preserve point ordering /
  commute with permutations? Apply a random permutation P, compare D(P y) vs P D(y).
  Hard gate for Phase 2 — see docs/PLAN.md.
- sweep_step_size(denoiser, y, v, cs): jvp error vs step size c, against the autograd
  JVP as reference. Picks the plateau where finite differences are trustworthy.
- antisym_energy(denoiser, y, probes): ||J - J^T|| / ||J + J^T|| estimated on random
  probe vectors via jvp/vjp — how non-symmetric (non-PSD) the implied covariance is.
- psd_report(eigvals): negative-eigenvalue mass and the PSD projection we report.
"""
