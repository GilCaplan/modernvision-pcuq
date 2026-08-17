# Project: Training-Free Structured Uncertainty from a Frozen Point-Cloud Denoiser

*(Source of truth for **what** we're building and **why**. The "how/when" lives in
[PLAN.md](PLAN.md); the day-to-day process in [WORKFLOW.md](WORKFLOW.md).)*

## Goal

Estimate structured posterior uncertainty directly from the Jacobian of a frozen
point-cloud denoiser, without retraining. We adapt Manor & Michaeli (ICLR 2024) from 2D
images to unstructured 3D point clouds, extracting **global geometric uncertainty modes**
— the top eigenvectors of the posterior covariance.

## The math in one page

For Gaussian denoising `Y = X + σZ`, `Z ~ N(0, I)`, Tweedie's formula ties the posterior
mean to the score:

```
E[X|Y=y] = y + σ² ∇ log p(y)
```

Differentiating once more (Manor & Michaeli's key identity): the **posterior covariance
is σ² times the Jacobian of the posterior mean**:

```
Cov[X|Y=y] = σ² · ∂μ₁(y)/∂y        where μ₁(y) = E[X|Y=y] ≈ D(y)  (the frozen denoiser)
```

So the top eigenvectors of `Cov[X|Y=y]` — the dominant uncertainty directions — can be
found by **power iteration on the denoiser's Jacobian**, where each Jacobian-vector
product is a single extra forward pass via finite differences:

```
J·v ≈ [D(y + c·v) − D(y)] / c        (forward difference, c ≈ 1e-6)
J·v ≈ [D(y + c·v) − D(y − c·v)] / 2c (central difference, more stable)
```

or exactly via autograd (`torch.func.jvp` / double-backward). No training, no sampling.

For a point cloud with N points, `y ∈ R^{N×3}`, so `J ∈ R^{3N×3N}` and eigenvectors are
**per-point 3D displacement fields** — visualizable as arrows on the shape.

## Method (concrete)

1. **Data:** ModelNet40 CAD meshes → sample N points → normalize → add controlled
   Gaussian noise `Y = X + σZ`, *retaining exact point indices* (correspondence).
2. **Denoiser:** pre-trained *Noise2Score3D* (Tweedie-based single-step denoiser →
   a natural MMSE-estimator stand-in). Fallback denoisers listed in [SOURCES.md](SOURCES.md).
3. **Jacobian access:** forward/central finite differences **and** autograd JVPs
   (each validates the other).
4. **Spectrum:** block power iteration (subspace iteration, as in the reference repo)
   or Lanczos → top 3–5 eigenpairs of `σ²·J` (symmetrized: `(J + Jᵀ)/2` via
   `vᵀJv` products or JVP+VJP).
5. **Output:** eigenvalues (uncertainty magnitude) + eigenvector displacement fields
   rendered on the point cloud.

## Our delta over prior work

Manor & Michaeli showed this for images on pixel grids. Point-cloud UQ otherwise needs a
separately-trained uncertainty model or heavy posterior sampling. We deliver
**training-free, test-time UQ from a frozen point-cloud model**, addressing what pixel
grids never had to face:

- **Permutation equivariance** — the denoiser must preserve point ordering for
  correspondence-based differences to be meaningful; we verify empirically.
- **Unstructured geometry** — no grid neighborhood structure; perturbations live in
  R^{N×3} displacement space.

## Challenges & mitigations (from the proposal)

| Challenge | Mitigation |
|---|---|
| Numerical instability: point perturbations cause nonlinear jumps | Step-size (`c`) sweeps; central differences; validate against exact autograd JVPs |
| Point correspondence: denoiser must preserve ordering | Empirical equivariance / ordering-preservation tests before trusting any spectrum |
| Asymmetric Jacobian → non-PSD covariance | Symmetrized products `(J+Jᵀ)/2`; monitor antisymmetric energy `‖J−Jᵀ‖/‖J+Jᵀ‖`; report PSD projections |

## Resources

- One standard GPU (VM, run from the Mac). Local Mac for smoke tests only.
- Feasibility rests on the frozen-model premise: zero training compute.

## Deliverables

- `src/pcuq`: minimal, readable PyTorch implementation.
- Quantitative sanity results (analytic cases where covariance is known in closed form).
- Qualitative results: top uncertainty modes visualized on ModelNet40 shapes across σ.
- Report/figures for the course.
