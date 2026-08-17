"""Top-k eigenpairs of the posterior covariance sigma^2 * J.

Clean re-derivation, for (B, N, 3) point clouds, of the reference subspace iteration
(cf. external/GaussianDenoisingPosterior/moments_calculations.py:get_eigvecs):

  top_eigenpairs(denoiser, y, sigma, k, iters, cfg):
    1. start from k random unit displacement fields v_i (N, 3)
    2. each iteration: v_i <- jvp(denoiser, y, v_i)  (batched, method/c from cfg)
    3. QR re-orthonormalize the k fields (flattened to (3N, k))
    4. eigenvalue_i = ||J v_i|| * sigma^2 ; sort descending
    5. track iteration-to-iteration correlation of the v_i as a convergence signal
  returns (eigvecs (k, N, 3), eigvals (k,), history)

Optional (decide by Phase 3, see docs/PLAN.md): Lanczos on the symmetrized operator.
"""
