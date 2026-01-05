from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class SinkhornParams:
    epsilon: float = 0.05
    reach: float = 1.0
    iterations: int = 30


def _pairwise_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x ** 2).sum(dim=-1, keepdim=True)
    y2 = (y ** 2).sum(dim=-1, keepdim=True)
    return x2 - 2 * x @ y.T + y2.T


def _unbalanced_sinkhorn_cost(
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    params: SinkhornParams,
) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.tensor(0.0, device=x.device)

    cost = _pairwise_cost(x, y)
    kernel = torch.exp(-cost / params.epsilon)
    kernel = kernel + 1e-8

    tau = params.reach / (params.reach + params.epsilon)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(params.iterations):
        Kv = kernel @ v + 1e-8
        u = (a / Kv) ** tau
        Ku = kernel.T @ u + 1e-8
        v = (b / Ku) ** tau

    transport = u[:, None] * kernel * v[None, :]
    return (transport * cost).sum()


def unbalanced_sinkhorn_divergence(
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    params: SinkhornParams,
) -> torch.Tensor:
    loss_ab = _unbalanced_sinkhorn_cost(a, b, x, y, params)
    loss_aa = _unbalanced_sinkhorn_cost(a, a, x, x, params)
    loss_bb = _unbalanced_sinkhorn_cost(b, b, y, y, params)
    return loss_ab - 0.5 * loss_aa - 0.5 * loss_bb


def tilewise_ot_loss(
    teacher_proj: torch.Tensor,
    student_proj: torch.Tensor,
    teacher_mass: torch.Tensor,
    student_mass: torch.Tensor,
    params: SinkhornParams,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    tiles_y, tiles_x = teacher_mass.shape[1], teacher_mass.shape[2]
    loss = torch.tensor(0.0, device=teacher_proj.device)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            a = teacher_mass[:, ty, tx]
            b = student_mass[:, ty, tx]
            if a.sum() <= 0 and b.sum() <= 0:
                continue

            if top_k is not None:
                if a.numel() > top_k:
                    a_vals, a_idx = torch.topk(a, top_k)
                    a_idx = a_idx[a_vals > 0]
                else:
                    a_idx = torch.nonzero(a > 0, as_tuple=False).squeeze(-1)
                if b.numel() > top_k:
                    b_vals, b_idx = torch.topk(b, top_k)
                    b_idx = b_idx[b_vals > 0]
                else:
                    b_idx = torch.nonzero(b > 0, as_tuple=False).squeeze(-1)
            else:
                a_idx = torch.nonzero(a > 0, as_tuple=False).squeeze(-1)
                b_idx = torch.nonzero(b > 0, as_tuple=False).squeeze(-1)

            if a_idx.numel() == 0 or b_idx.numel() == 0:
                continue

            a_sel = a[a_idx]
            b_sel = b[b_idx]
            x = teacher_proj[a_idx]
            y = student_proj[b_idx]

            loss = loss + unbalanced_sinkhorn_divergence(a_sel, b_sel, x, y, params)

    return loss


def build_proj_features(u: torch.Tensor, s: torch.Tensor, rgb: Optional[torch.Tensor]) -> torch.Tensor:
    if rgb is None:
        return torch.cat([u, s.unsqueeze(-1)], dim=-1)
    return torch.cat([u, s.unsqueeze(-1), rgb], dim=-1)

