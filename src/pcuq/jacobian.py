"""Jacobian-vector products against a frozen denoiser at anchor y.

All methods approximate J v where J = dD(y)/dy, with v a (N, 3) displacement field
(batched as (k, N, 3)). Planned API:

- jvp(denoiser, y, v, method, c):
    method='forward':  (D(y + c v) - D(y)) / c            — 1 extra forward pass
    method='central':  (D(y + c v) - D(y - c v)) / (2c)   — 2 passes, O(c^2) error
    method='autograd': torch.func.jvp(denoiser, y, v)     — exact, validates the others
  (cf. external/GaussianDenoisingPosterior/moments_calculations.py:_forward_directional)
- vjp(denoiser, y, v): autograd transpose product, for symmetrization.
- sym_jvp(denoiser, y, v, ...): v -> 0.5 * (J v + J^T v). The posterior covariance is
  symmetric by theory; an approximate denoiser's J may not be — spectrum.py works on
  the symmetrized operator, diagnostics.py monitors the antisymmetric energy.
"""
