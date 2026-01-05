import os
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from init_kdtree import init_student_from_teacher_kdtree
from renderer import GaussianModel, create_sample_scene, make_camera, render_with_stats
from train import evaluate, generate_cameras, render_teacher_dataset, train_baseline
from utils import save_json, seed_everything


def save_image_grid(
    images: List[torch.Tensor],
    titles: List[str],
    path: str,
) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.cpu().numpy())
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_experiments(output_dir: str = "outputs", student_count: int = 4) -> Dict[str, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    teacher_gaussians = create_sample_scene(device)
    teacher_model = GaussianModel.from_numpy_gaussians(teacher_gaussians, learnable=False).to(device)

    cameras = generate_cameras(device, num_views=6, radius=3.0, width=128, height=96, focal_length=300.0)
    train_cams = cameras[:4]
    val_cams = cameras[4:]

    teacher_train_images, _, _ = render_teacher_dataset(teacher_model, train_cams, tile_size=32)
    teacher_val_images, _, _ = render_teacher_dataset(teacher_model, val_cams, tile_size=32)

    results: Dict[str, Dict[str, float]] = {}
    os.makedirs(output_dir, exist_ok=True)

    results["teacher"] = evaluate(teacher_model, val_cams, teacher_val_images)

    kd_init = init_student_from_teacher_kdtree(teacher_gaussians, student_count)
    kd_student = GaussianModel.from_numpy_gaussians(kd_init, learnable=False).to(device)
    results["kdtree_init"] = evaluate(kd_student, val_cams, teacher_val_images)

    distill_metrics = train_baseline(
        output_dir=os.path.join(output_dir, "distill_only"),
        student_count=student_count,
        beta_ot=0.0,
        reseed_count=0,
    )
    results["distill_only"] = distill_metrics

    full_metrics, full_model = train_baseline(
        output_dir=os.path.join(output_dir, "full_baseline"),
        student_count=student_count,
        beta_ot=0.1,
        reseed_count=1,
        return_model=True,
    )
    results["full_baseline"] = full_metrics

    save_json(results, os.path.join(output_dir, "all_metrics.json"))

    with torch.no_grad():
        cam = val_cams[0]
        teacher_img, _, _ = render_with_stats(teacher_model, cam, tile_size=32)
        student_img, _, _ = render_with_stats(full_model, cam, tile_size=32)
        diff_img = torch.abs(student_img - teacher_img)
    save_image_grid(
        [teacher_img, student_img, diff_img],
        ["Teacher", "Student", "Abs Diff"],
        os.path.join(output_dir, "qualitative.png"),
    )

    print("Variant metrics:")
    for name, metrics in results.items():
        print(f"{name:>12} | PSNR {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} | "
              f"SSIM {metrics['ssim_mean']:.3f} ± {metrics['ssim_std']:.3f} | "
              f"Gaussians {metrics['gaussians']}")

    return results


if __name__ == "__main__":
    run_experiments()
