"""Visualization of uncertainty modes on point clouds.

Planned API (Phase 4, but grown incrementally as results appear):

- plot_modes(x_hat, eigvecs, eigvals, path): per-mode figure — the denoised cloud with
  the eigenvector as a per-point displacement field (arrows / color by magnitude).
- plot_mode_sweep(x_hat, eigvec, eigval, ts, path): x_hat + t*sqrt(eigval)*v for
  t in ts — the 3D analog of the reference repo's image sliders.
- plot_uncertainty_heatmap(x_hat, eigvecs, eigvals, path): per-point marginal
  uncertainty (sum of lambda_i * v_i^2 over modes) as color.

matplotlib for static report figures; plotly for interactive 3D inspection.
"""
