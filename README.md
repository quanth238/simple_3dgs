# Simple 3D Gaussian Splatting Baseline

This repository adds a minimal, reproducible baseline for teacher→student compression with a hard Gaussian budget and tile-wise unbalanced OT.

## Setup

The pipeline depends on PyTorch, NumPy, Matplotlib, and PyTest (for the unit test).

```bash
pip install torch numpy matplotlib pytest
```

## Run the full experiment grid

```bash
python eval.py
```

Outputs are written to `outputs/`:

- `outputs/all_metrics.json` with PSNR/SSIM and Gaussian count for each variant.
- `outputs/qualitative.png` with teacher/student/difference comparison.

## Run a single training configuration

```bash
python train.py
```

Adjust hyperparameters by editing the `train_baseline(...)` call in `train.py` (e.g., `student_count`, `beta_ot`, `reseed_period`).

## Run the OT unit test

```bash
python -m pytest -q
```

