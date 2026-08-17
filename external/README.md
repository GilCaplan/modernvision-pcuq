# external/ — vendored third-party code

Rules: **read-only.** We never edit, import-and-monkeypatch, or mix our code in here.
Anything we need in modified form is rewritten cleanly in `src/pcuq/` with a comment
pointing back at the origin file. Each vendored repo keeps its own license.

Vendored clones are **gitignored** (they're hundreds of MB); this README is the record
of what to fetch. After a fresh checkout, restore with the clone commands below.

## GaussianDenoisingPosterior/

Manor & Michaeli, ICLR 2024 — official implementation, upstream commit
`402fed92b3566363ab45fa73671dd47a4fa4bb5a` (cloned 2026-08-17).

```bash
git clone --depth 1 https://github.com/HilaManor/GaussianDenoisingPosterior \
    external/GaussianDenoisingPosterior
```

Reference implementation for the method we adapt. What we actually use:

- `moments_calculations.py` — `get_eigvecs` (subspace iteration on the denoiser
  Jacobian via forward differences) and `_forward_directional`. This is the algorithmic
  core we re-derive for point clouds in `src/pcuq/spectrum.py` / `jacobian.py`.
- `models_wrappers/models_wrapper_base.py` — pattern for the frozen-denoiser interface.
- `README.md` / paper — usage of `c`, iteration counts, double-precision notes.

The rest (DDPM_FFHQ, MNIST, pn2v, FMD data, MATLAB scripts, website `docs/`) is
image-domain baggage we don't touch. Note: the clone is ~300MB mostly due to bundled
data/checkpoints; if repo size ever matters, re-clone with sparse checkout of the
`*.py` files only.

## Noise2Score3D/

Wei et al., ICCV 2025 — the frozen point-cloud denoiser under study. Upstream commit
`fef67d75155f0ab75c6f64ce7fc24ae2689186de` (cloned 2026-08-17). No license file
upstream (course research use). Pretrained weights: download separately from
[Hugging Face](https://huggingface.co/bobby645/Noise2Score3D/tree/main) into
`data/checkpoints/` (gitignored).

```bash
git clone --depth 1 https://github.com/Bobby645/Noise2Score3D external/Noise2Score3D
```

What we use: `models/KPconv.py` (the denoiser network), `test.py` (how they load the
checkpoint and run single-step Tweedie inference — the pattern our
`pcuq.denoisers.Noise2Score3DWrapper` follows). Their heavy deps (PyTorch3D, pykeops)
stay confined to GPU-VM environments.
