# Audit Note: Simple 3D Gaussian Splatting Baseline

## Notebook availability
- Requested notebook `/mnt/data/simple-3d-gaussian-splatting-renderer.ipynb` was **not present** in this environment.
- The only artifact available is `simple_3d_gaussian_splatting_renderer.py`, which is a **JSON-encoded notebook** (not a standard `.py` module).

## What exists in the codebase
- A **NumPy-only** renderer that:
  - Stores Gaussians as `Gaussian3D` objects with `position`, `scale`, `rotation`, `color`, `opacity`.
  - Uses **simple perspective projection**: `x = f * x / |z| + cx`, `y = f * y / |z| + cy`.
  - Computes a **2D covariance** via rotated 3D covariance and a scale factor based on depth.
  - Performs **alpha blending** back-to-front (farthest-first).
- No training loop, no torch/autograd path, and no OT loss/optimization support.

## Minimal changes planned/implemented
- **Keep the original NumPy renderer intact** for reference and sanity checks.
- Add a **PyTorch differentiable renderer** that mirrors the same projection and back-to-front blending order.
  - For simplicity and stability, the differentiable path uses **axis-aligned 2D covariances** derived from `scale_x, scale_y` (rotation is ignored in projection).  
  - This is consistent with the sample scene (identity rotations) and is explicitly documented here.
- Add baseline training pipeline, KD-tree init, OT loss, reseed, and evaluation as separate modules.

