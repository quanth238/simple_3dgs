from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


@dataclass
class Cluster:
    indices: np.ndarray
    spread: float


def _cluster_spread(positions: np.ndarray) -> float:
    if positions.shape[0] <= 1:
        return 0.0
    return float(np.max(positions.max(axis=0) - positions.min(axis=0)))


def _split_cluster(indices: np.ndarray, positions: np.ndarray) -> List[np.ndarray]:
    pts = positions[indices]
    spans = pts.max(axis=0) - pts.min(axis=0)
    axis = int(np.argmax(spans))
    sorted_idx = indices[np.argsort(positions[indices, axis])]
    mid = len(sorted_idx) // 2
    return [sorted_idx[:mid], sorted_idx[mid:]]


def init_student_from_teacher_kdtree(teacher: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    positions = teacher["positions"].detach().cpu().numpy()
    colors = teacher["colors"].detach().cpu().numpy()
    scales = teacher["scales"].detach().cpu().numpy()
    opacities = teacher["opacities"].detach().cpu().numpy().squeeze(-1)

    clusters: List[np.ndarray] = [np.arange(positions.shape[0])]
    while len(clusters) < n:
        spreads = [_cluster_spread(positions[idx]) for idx in clusters]
        split_idx = int(np.argmax(spreads))
        if spreads[split_idx] <= 0.0:
            break
        to_split = clusters.pop(split_idx)
        left, right = _split_cluster(to_split, positions)
        if left.size == 0 or right.size == 0:
            clusters.append(to_split)
            break
        clusters.extend([left, right])

    if len(clusters) < n:
        for _ in range(n - len(clusters)):
            clusters.append(clusters[-1])

    student_positions = []
    student_scales = []
    student_colors = []
    student_opacities = []

    for cluster in clusters[:n]:
        weights = opacities[cluster]
        weights = weights / (weights.sum() + 1e-8)

        pos = (positions[cluster] * weights[:, None]).sum(axis=0)
        col = (colors[cluster] * weights[:, None]).sum(axis=0)

        var = (scales[cluster] ** 2 * weights[:, None]).sum(axis=0)
        merged_scale = np.sqrt(np.maximum(var, 1e-6))

        opacity = float(np.clip(weights.sum(), 0.0, 1.0))

        student_positions.append(pos)
        student_colors.append(col)
        student_scales.append(merged_scale)
        student_opacities.append([opacity])

    device = teacher["positions"].device
    return {
        "positions": torch.tensor(student_positions, dtype=torch.float32, device=device),
        "scales": torch.tensor(student_scales, dtype=torch.float32, device=device),
        "colors": torch.tensor(student_colors, dtype=torch.float32, device=device),
        "opacities": torch.tensor(student_opacities, dtype=torch.float32, device=device),
    }

