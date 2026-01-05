import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class Camera:
    position: torch.Tensor
    focal_length: float
    width: int
    height: int
    background_color: torch.Tensor


class GaussianModel(nn.Module):
    def __init__(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        learnable: bool = True,
    ):
        super().__init__()
        self.positions = nn.Parameter(positions, requires_grad=learnable)
        self.log_scales = nn.Parameter(scales.log(), requires_grad=learnable)
        self.color_logits = nn.Parameter(logit(colors), requires_grad=learnable)
        self.opacity_logits = nn.Parameter(logit(opacities), requires_grad=learnable)

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def parameters_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "positions": self.positions,
            "scales": self.scales(),
            "colors": self.colors(),
            "opacities": self.opacities(),
        }

    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def colors(self) -> torch.Tensor:
        return torch.sigmoid(self.color_logits)

    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    @staticmethod
    def from_numpy_gaussians(gaussians: Dict[str, torch.Tensor], learnable: bool) -> "GaussianModel":
        return GaussianModel(
            positions=gaussians["positions"].clone(),
            scales=gaussians["scales"].clone(),
            colors=gaussians["colors"].clone(),
            opacities=gaussians["opacities"].clone(),
            learnable=learnable,
        )


def logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, eps, 1 - eps)
    return torch.log(x / (1 - x))


def project_to_2d(points: torch.Tensor, camera: Camera) -> Tuple[torch.Tensor, torch.Tensor]:
    relative = points - camera.position[None, :]
    z = relative[:, 2].abs().clamp(min=1e-6)
    x_2d = (relative[:, 0] * camera.focal_length) / z + camera.width / 2.0
    y_2d = (relative[:, 1] * camera.focal_length) / z + camera.height / 2.0
    return torch.stack([x_2d, y_2d], dim=-1), z


def render_with_stats(
    model: GaussianModel,
    camera: Camera,
    tile_size: int = 32,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    positions = model.positions
    scales = model.scales()
    colors = model.colors()
    opacities = model.opacities().squeeze(-1)

    device = positions.device
    height, width = camera.height, camera.width
    tiles_y = math.ceil(height / tile_size)
    tiles_x = math.ceil(width / tile_size)

    image = torch.zeros((height, width, 3), device=device)
    alpha_buffer = torch.zeros((height, width), device=device)

    proj, depth = project_to_2d(positions, camera)
    scale_factor = camera.focal_length / depth

    var_x = (scales[:, 0] * scale_factor) ** 2
    var_y = (scales[:, 1] * scale_factor) ** 2
    log_area = 0.5 * torch.log(var_x * var_y + 1e-12)

    # Normalize projected coordinates and footprint scale to keep OT cost well-scaled.
    u_norm = torch.stack(
        [proj[:, 0] / max(width, 1), proj[:, 1] / max(height, 1)],
        dim=-1,
    )
    log_area_norm = log_area - math.log(max(width * height, 1))

    proj_features = {
        "u": u_norm,
        "s": log_area_norm,
        "rgb": colors,
    }

    mass_per_tile = torch.zeros((positions.shape[0], tiles_y, tiles_x), device=device)

    depth_order = torch.norm(positions - camera.position[None, :], dim=-1)
    sorted_indices = torch.argsort(depth_order, descending=True)

    for idx in sorted_indices.tolist():
        x_2d, y_2d = proj[idx]
        if x_2d < 0 or x_2d >= width or y_2d < 0 or y_2d >= height:
            continue

        vx = var_x[idx].clamp(min=1e-8)
        vy = var_y[idx].clamp(min=1e-8)
        max_radius = int(3.0 * torch.sqrt(torch.max(vx, vy)).item()) + 1

        x_min = max(0, int(x_2d.item()) - max_radius)
        x_max = min(width, int(x_2d.item()) + max_radius + 1)
        y_min = max(0, int(y_2d.item()) - max_radius)
        y_max = min(height, int(y_2d.item()) + max_radius + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        ys = torch.arange(y_min, y_max, device=device)
        xs = torch.arange(x_min, x_max, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx = xx - x_2d
        dy = yy - y_2d
        gaussian_val = torch.exp(-0.5 * ((dx ** 2) / vx + (dy ** 2) / vy))

        alpha = (opacities[idx] * gaussian_val).clamp(0.0, 1.0)
        transmittance = 1.0 - alpha_buffer[y_min:y_max, x_min:x_max]
        contribution = alpha * transmittance

        image[y_min:y_max, x_min:x_max, :] += contribution.unsqueeze(-1) * colors[idx]
        alpha_buffer[y_min:y_max, x_min:x_max] += contribution

        tile_y = (yy // tile_size).reshape(-1)
        tile_x = (xx // tile_size).reshape(-1)
        tile_index = tile_y * tiles_x + tile_x
        contrib_flat = contribution.reshape(-1)

        tile_mass = torch.zeros(tiles_y * tiles_x, device=device)
        tile_mass = tile_mass.scatter_add(0, tile_index, contrib_flat)
        mass_per_tile[idx] += tile_mass.reshape(tiles_y, tiles_x)

    background = camera.background_color.to(device).view(1, 1, 3)
    image = image + (1.0 - alpha_buffer).unsqueeze(-1) * background
    return image.clamp(0.0, 1.0), proj_features, mass_per_tile


def create_sample_scene(device: torch.device) -> Dict[str, torch.Tensor]:
    rings = []
    colors = []
    scales = []
    opacities = []
    num_rings = 3
    points_per_ring = 8
    for ring in range(num_rings):
        radius = 0.4 + 0.3 * ring
        z = -4.0 - 0.6 * ring
        for i in range(points_per_ring):
            angle = 2 * math.pi * i / points_per_ring
            rings.append([radius * math.cos(angle), 0.3 * math.sin(angle), z])
            color = [
                0.5 + 0.5 * math.cos(angle),
                0.5 + 0.5 * math.sin(angle),
                0.3 + 0.2 * ring,
            ]
            colors.append(color)
            scales.append([0.18 + 0.05 * ring, 0.18 + 0.04 * ring, 0.25 + 0.03 * ring])
            opacities.append([0.75 - 0.1 * ring])

    positions = torch.tensor(rings, device=device, dtype=torch.float32)
    colors = torch.tensor(colors, device=device, dtype=torch.float32)
    scales = torch.tensor(scales, device=device, dtype=torch.float32)
    opacities = torch.tensor(opacities, device=device, dtype=torch.float32)
    return {
        "positions": positions,
        "scales": scales,
        "colors": colors,
        "opacities": opacities,
    }


def make_camera(
    position: torch.Tensor,
    width: int,
    height: int,
    focal_length: float,
    background_color: Tuple[float, float, float] = (0.05, 0.05, 0.1),
) -> Camera:
    return Camera(
        position=position,
        focal_length=focal_length,
        width=width,
        height=height,
        background_color=torch.tensor(background_color, dtype=torch.float32, device=position.device),
    )
