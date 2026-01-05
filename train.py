import os
import time
from typing import Dict, List, Tuple

import torch

from init_kdtree import init_student_from_teacher_kdtree
from ot_loss import SinkhornParams, build_proj_features, tilewise_ot_loss
from renderer import GaussianModel, create_sample_scene, make_camera, render_with_stats
from utils import psnr, save_json, seed_everything, ssim


def generate_cameras(
    device: torch.device,
    num_views: int,
    radius: float,
    width: int,
    height: int,
    focal_length: float,
) -> List[torch.Tensor]:
    angles = torch.linspace(0, 2 * torch.pi, num_views, device=device)
    cameras = []
    for angle in angles:
        position = torch.tensor(
            [radius * torch.cos(angle), 0.5 * torch.sin(2 * angle), radius * torch.sin(angle)],
            device=device,
        )
        cameras.append(make_camera(position, width, height, focal_length))
    return cameras


def render_teacher_dataset(
    teacher: GaussianModel,
    cameras: List,
    tile_size: int,
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
    images = []
    proj_features = []
    masses = []
    with torch.no_grad():
        for cam in cameras:
            image, proj, mass = render_with_stats(teacher, cam, tile_size=tile_size)
            images.append(image)
            proj_features.append(proj)
            masses.append(mass)
    return images, proj_features, masses


def compute_losses(
    student: GaussianModel,
    teacher_images: List[torch.Tensor],
    cameras: List,
    tile_size: int,
    sinkhorn_params: SinkhornParams,
    lambda_ssim: float,
    beta_ot: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    l_img = torch.tensor(0.0, device=student.device)
    l_ot = torch.tensor(0.0, device=student.device)
    student_images = []
    student_masses = []

    for cam, teacher_img in zip(cameras, teacher_images):
        student_img, student_proj, student_mass = render_with_stats(student, cam, tile_size=tile_size)
        student_images.append(student_img)
        student_masses.append(student_mass)

        l1 = torch.mean(torch.abs(student_img - teacher_img))
        dssim = (1 - ssim(student_img.unsqueeze(0), teacher_img.unsqueeze(0))) / 2
        l_img = l_img + l1 + lambda_ssim * dssim

        if beta_ot > 0:
            teacher_u = cam.cached_proj["u"]
            teacher_s = cam.cached_proj["s"]
            teacher_rgb = cam.cached_proj["rgb"]
            teacher_proj = build_proj_features(teacher_u, teacher_s, teacher_rgb)
            student_proj = build_proj_features(student_proj["u"], student_proj["s"], student_proj["rgb"])

            l_ot = l_ot + tilewise_ot_loss(
                teacher_proj,
                student_proj,
                cam.cached_mass,
                student_mass,
                params=sinkhorn_params,
                top_k=top_k,
            )

    l_img = l_img / len(cameras)
    l_ot = l_ot / len(cameras)
    return l_img, l_ot, student_images, student_masses


def reseed_student(
    student: GaussianModel,
    teacher: GaussianModel,
    mass_history: List[torch.Tensor],
    residual_history: List[torch.Tensor],
    teacher_mass_per_tile: List[torch.Tensor],
    replace_count: int,
) -> None:
    if not mass_history or replace_count == 0:
        return

    avg_mass = torch.stack([m.sum(dim=(1, 2)) for m in mass_history]).mean(dim=0)
    _, worst_idx = torch.topk(avg_mass, k=replace_count, largest=False)

    residual = torch.stack(residual_history).mean(dim=0)
    residual_score = residual.mean(dim=-1)

    with torch.no_grad():
        teacher_positions = teacher.positions
        distances = torch.cdist(teacher_positions, student.positions)
        closest_student = distances.argmin(dim=1)
        score = torch.zeros_like(avg_mass)

        residual_tiles = []
        for cam_idx, mass in enumerate(teacher_mass_per_tile):
            tiles_y, tiles_x = mass.shape[1:]
            tile_size_y = residual_score.shape[0] // tiles_y
            tile_size_x = residual_score.shape[1] // tiles_x
            tile_residual = torch.zeros((tiles_y, tiles_x), device=student.device)
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    y0 = ty * tile_size_y
                    x0 = tx * tile_size_x
                    tile_residual[ty, tx] = residual_score[y0 : y0 + tile_size_y, x0 : x0 + tile_size_x].mean()
            residual_tiles.append(tile_residual)

        teacher_scores = torch.zeros((teacher_positions.shape[0],), device=student.device)
        for mass, tile_residual in zip(teacher_mass_per_tile, residual_tiles):
            teacher_scores += (mass * tile_residual[None, :, :]).sum(dim=(1, 2))

        for t_idx, s_idx in enumerate(closest_student):
            score[s_idx] += teacher_scores[t_idx]

        candidates = torch.topk(score, k=replace_count, largest=True).indices
        student.positions[worst_idx] = teacher.positions[candidates] + 0.01 * torch.randn_like(
            teacher.positions[candidates]
        )
        student.log_scales[worst_idx] = teacher.log_scales[candidates]
        student.color_logits[worst_idx] = teacher.color_logits[candidates]
        student.opacity_logits[worst_idx] = teacher.opacity_logits[candidates]


def train_baseline(
    output_dir: str = "outputs",
    student_count: int = 4,
    steps: int = 30,
    batch_views: int = 3,
    tile_size: int = 32,
    beta_ot: float = 0.1,
    lambda_ssim: float = 0.2,
    gamma_reg: float = 1e-3,
    reseed_period: int = 10,
    reseed_count: int = 1,
    top_k: int = 20,
    return_model: bool = False,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    teacher_gaussians = create_sample_scene(device)
    teacher_model = GaussianModel.from_numpy_gaussians(teacher_gaussians, learnable=False).to(device)

    student_init = init_student_from_teacher_kdtree(teacher_gaussians, student_count)
    student_model = GaussianModel.from_numpy_gaussians(student_init, learnable=True).to(device)

    cameras = generate_cameras(device, num_views=6, radius=3.0, width=128, height=96, focal_length=300.0)
    train_cams = cameras[:4]
    val_cams = cameras[4:]

    teacher_train_images, teacher_proj, teacher_mass = render_teacher_dataset(
        teacher_model, train_cams, tile_size
    )
    teacher_val_images, _, _ = render_teacher_dataset(teacher_model, val_cams, tile_size)

    for cam, proj, mass in zip(train_cams, teacher_proj, teacher_mass):
        cam.cached_proj = proj
        cam.cached_mass = mass

    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-2)
    sinkhorn_params = SinkhornParams()
    mass_history: List[torch.Tensor] = []
    residual_history: List[torch.Tensor] = []

    for step in range(steps):
        optimizer.zero_grad()
        batch_cams = train_cams[:batch_views]
        batch_teacher_images = teacher_train_images[:batch_views]

        start = time.time()
        l_img, l_ot, student_images, student_masses = compute_losses(
            student_model,
            batch_teacher_images,
            batch_cams,
            tile_size,
            sinkhorn_params,
            lambda_ssim,
            beta_ot,
            top_k,
        )
        reg = student_model.scales().mean() + student_model.opacities().mean()
        total = l_img + beta_ot * l_ot + gamma_reg * reg
        total.backward()
        optimizer.step()
        elapsed = time.time() - start

        print(
            f"step {step:03d} | loss {total.item():.4f} | img {l_img.item():.4f} | ot {l_ot.item():.4f} | {elapsed:.3f}s"
        )

        mass_history.append(torch.stack([m for m in student_masses]).mean(dim=0).detach())
        residual_history.append(torch.stack([abs(s - t) for s, t in zip(student_images, batch_teacher_images)]).mean(dim=0))

        if (step + 1) % reseed_period == 0:
            reseed_student(
                student_model,
                teacher_model,
                mass_history,
                residual_history,
                [cam.cached_mass for cam in batch_cams],
                reseed_count,
            )
            mass_history.clear()
            residual_history.clear()

    metrics = evaluate(student_model, val_cams, teacher_val_images)
    os.makedirs(output_dir, exist_ok=True)
    save_json(metrics, os.path.join(output_dir, "metrics.json"))
    if return_model:
        return metrics, student_model
    return metrics


def evaluate(student: GaussianModel, cameras: List, targets: List[torch.Tensor]) -> Dict[str, float]:
    psnrs = []
    ssims = []
    with torch.no_grad():
        for cam, target in zip(cameras, targets):
            image, _, _ = render_with_stats(student, cam, tile_size=32)
            psnrs.append(psnr(image, target).item())
            ssims.append(ssim(image.unsqueeze(0), target.unsqueeze(0)).item())
    return {
        "psnr_mean": float(torch.tensor(psnrs).mean().item()),
        "psnr_std": float(torch.tensor(psnrs).std().item()),
        "ssim_mean": float(torch.tensor(ssims).mean().item()),
        "ssim_std": float(torch.tensor(ssims).std().item()),
        "gaussians": int(student.positions.shape[0]),
    }


if __name__ == "__main__":
    train_baseline()
