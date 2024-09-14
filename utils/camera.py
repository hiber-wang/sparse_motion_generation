from typing import Optional

import torch


def perspective_projection(points: torch.Tensor, K: torch.Tensor, rotation: Optional[torch.Tensor] = None, translation: Optional[torch.Tensor] = None) -> torch.Tensor:
    dtype, device = points.dtype, points.device
    K = K.to(dtype=dtype, device=device)
    if rotation is not None:
        points = torch.einsum('...ij,...kj->...ki', rotation.to(device=device, dtype=dtype), points)
    if translation is not None:
        points = points + translation.to(dtype=dtype, device=device).unsqueeze(dim=-2)

    projected_points = points / points[..., -1].unsqueeze(dim=-1)
    projected_points = torch.einsum('...ij,...kj->...ki', K, projected_points)    
    return projected_points[..., :-1]


def normalize_coordinate(points: torch.Tensor, K: torch.Tensor):
    w, h = K[..., 0, [2]] * 2, K[..., 1, [2]] * 2
    scale = torch.cat([w, h], dim=-1).reshape(*K.shape[:-2], *(1 for _ in points.shape[2:-1]), -1)
    return points / scale
