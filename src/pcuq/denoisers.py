"""Frozen denoisers behind one tiny interface.

Contract: denoise(y) maps (B, N, 3) -> (B, N, 3), model frozen (eval, no grad state),
point ordering preserved (verified by diagnostics.check_equivariance before trust).

Planned implementations:

- AnalyticGaussianDenoiser (Phase 1): for X ~ N(mu, C), the MMSE denoiser is linear,
    D(y) = mu + C (C + sigma^2 I)^{-1} (y - mu),
  and the posterior covariance sigma^2 * J is known exactly -> ground truth for tests.
- Noise2Score3DWrapper (Phase 2): wraps the vendored pretrained model from external/
  (cf. external/GaussianDenoisingPosterior/models_wrappers/models_wrapper_base.py for
  the wrapper pattern). Fallback: ScoreDenoise wrapper — see docs/SOURCES.md.
"""
