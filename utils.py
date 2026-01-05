import json
import math
import os
from typing import Dict

import numpy as np
import torch


def seed_everything(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    if pred.shape[-1] != 3:
        raise ValueError("Expected RGB images for SSIM.")
    pad = window_size // 2
    window = torch.ones(1, 1, window_size, window_size, device=pred.device) / (window_size ** 2)

    pred_gray = pred.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
    target_gray = target.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)

    mu_x = torch.nn.functional.conv2d(pred_gray, window, padding=pad)
    mu_y = torch.nn.functional.conv2d(target_gray, window, padding=pad)
    sigma_x = torch.nn.functional.conv2d(pred_gray ** 2, window, padding=pad) - mu_x ** 2
    sigma_y = torch.nn.functional.conv2d(target_gray ** 2, window, padding=pad) - mu_y ** 2
    sigma_xy = torch.nn.functional.conv2d(pred_gray * target_gray, window, padding=pad) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean()


def save_json(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

