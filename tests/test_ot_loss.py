import torch

from ot_loss import SinkhornParams, build_proj_features, tilewise_ot_loss


def test_ot_loss_finite_and_grad():
    device = torch.device("cpu")
    teacher_u = torch.tensor([[10.0, 10.0], [20.0, 20.0]], device=device)
    student_u = torch.tensor([[12.0, 10.0], [18.0, 22.0]], device=device, requires_grad=True)
    teacher_s = torch.tensor([0.1, 0.1], device=device)
    student_s = torch.tensor([0.1, 0.1], device=device, requires_grad=True)
    teacher_rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    student_rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, requires_grad=True)

    teacher_proj = build_proj_features(teacher_u, teacher_s, teacher_rgb)
    student_proj = build_proj_features(student_u, student_s, student_rgb)

    teacher_mass = torch.tensor([[1.0], [0.5]], device=device).view(2, 1, 1)
    student_mass = torch.tensor([[0.8], [0.3]], device=device).view(2, 1, 1)

    params = SinkhornParams(epsilon=0.5, reach=1.0, iterations=10)
    loss = tilewise_ot_loss(teacher_proj, student_proj, teacher_mass, student_mass, params=params)
    assert torch.isfinite(loss)

    loss.backward()
    assert student_u.grad.abs().sum() > 0
    assert student_s.grad.abs().sum() > 0
    assert student_rgb.grad.abs().sum() > 0

