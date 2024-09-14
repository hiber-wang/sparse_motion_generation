from typing import Dict

import torch
from torch import nn

from model import SMPLX
from utils.camera import perspective_projection, normalize_coordinate
from utils.representation import recover_from_vector
from utils.keypoints_conversion import smpl_to_openpose
from utils.rotation_conversion import (rotation_6d_to_matrix, 
                                       matrix_to_axis_angle)


def mean_flat(t: torch.Tensor) -> torch.Tensor:
    return t.mean(dim=list(range(1, len(t.shape))))


def sum_flat(t: torch.Tensor) -> torch.Tensor:
    return t.sum(dim=list(range(1, len(t.shape))))


def geman_mcclure(t: torch.Tensor, sigma: float) -> torch.Tensor:
    t_squared = t ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * t_squared) / (sigma_squared + t_squared)


def geodesic(x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean', eps: float = 1e-7) -> torch.Tensor:
        R_diffs = x @ y.transpose(-1, -2)
        # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))
        if reduction == "mean":
            return dists.mean()
        elif reduction == "sum":
            return dists.sum()
        else:
            return dists
        

def jitter(x: torch.Tensor) -> torch.Tensor:
    # Assuming tensor has shape (B, T, ...)
    return torch.linalg.norm(x[:, 2:] + x[:, :-2] - 2 * x[:, 1:-1], dim=-1)


def masked_l2(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    l2 = (a - b).norm(dim=-1)
    if len(l2.shape) > 2:
        # reduce to (B, T) by mean
        l2 = l2.mean(dim=list(range(2, len(l2.shape))))
    m_l2 = sum_flat(l2 * mask)
    num_entries = sum_flat(mask)
    return m_l2 / num_entries



r6d_to_aa = lambda x: matrix_to_axis_angle(rotation_6d_to_matrix(x))


class VelocityLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = meta['padding_mask']
        vel_gt = target[:, 1:, :] - target[:, :-1, :]
        vel_pred = x[:, 1:, :] - x[:, :-1, :]
        return masked_l2(vel_pred, vel_gt, mask[..., 1:].unsqueeze(dim=-1))
    


class ReprojectionLoss(nn.Module):
    def __init__(self, smpl_dir: str) -> None:
        super().__init__()
        self.smpl = SMPLX(smpl_dir)
    
    def forward(self, x: torch.Tensor, target: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        # K, R, T = meta['K'], meta['R'], meta['T']
        K = meta['K']
        mask = meta['padding_mask']
        kp2d_gt = normalize_coordinate(meta['kp2d'], K)
        data_dict = recover_from_vector(x)
        pose, trans = data_dict['rotation'], data_dict['trans']
        pose = r6d_to_aa(pose)

        kp2d_pred = perspective_projection(self.smpl.forward_kinematics(pose, transl=trans), K=K)
        kp2d_pred = normalize_coordinate(smpl_to_openpose(kp2d_pred), K)

        return sum_flat((kp2d_pred - kp2d_gt).norm(dim=-1).mean(dim=-1) * mask) / sum_flat(mask)



class FlatMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor, target: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = meta['padding_mask']
        return masked_l2(x, target, mask)
    


class GeometricLoss(nn.Module):
    def __init__(self, smpl_dir: str) -> None:
        super().__init__()
        self.smpl = SMPLX(smpl_dir)

    def forward(self, x: torch.Tensor, target: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = meta['padding_mask']
        pred = recover_from_vector(x)
        gt = recover_from_vector(target)
        kp3d_pred = self.smpl.full_body_joints(self.smpl.forward_kinematics(r6d_to_aa(pred['rotation']), transl=pred['trans']), flat=False)
        kp3d_gt = self.smpl.full_body_joints(self.smpl.forward_kinematics(r6d_to_aa(gt['rotation']), transl=gt['trans']), flat=False)
        return masked_l2(kp3d_pred, kp3d_gt, mask)
    

class WeightedMSELoss(nn.Module):
    def __init__(self, rot_scale: float = 1.0, pos_scale: float = 1.0, trans_scale: float = 1.0) -> None:
        super().__init__()
        self.rot_scale = rot_scale
        self.pos_scale = pos_scale
        self.trans_scale = trans_scale

    def forward(self, x: torch.Tensor, target: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = meta['padding_mask']
        pred = recover_from_vector(x)
        gt = recover_from_vector(target)

        loss = self.rot_scale * masked_l2(pred['rotation'], gt['rotation'], mask) \
               + self.pos_scale * masked_l2(pred['position'], gt['position'], mask) \
               + self.trans_scale * masked_l2(pred['trans'], gt['trans'], mask)
        
        return loss
    

class WeightedVelocityLoss(nn.Module):
    def __init__(self, rot_scale: float = 1.0, pos_scale: float = 1.0, trans_scale: float = 1.0) -> None:
        super().__init__()
        self.rot_scale = rot_scale
        self.pos_scale = pos_scale
        self.trans_scale = trans_scale

    def forward(self, x: torch.Tensor, target: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = meta['padding_mask']
        pred = recover_from_vector(x[:, 1:] - x[:, :-1])
        gt = recover_from_vector(target[:, 1:] - target[:, :-1])

        loss = self.rot_scale * masked_l2(pred['rotation'], gt['rotation'], mask[:, 1:]) \
               + self.pos_scale * masked_l2(pred['position'], gt['position'], mask[:, 1:]) \
               + self.trans_scale * masked_l2(pred['trans'], gt['trans'], mask[:, 1:])

        return loss
    

class TranslationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor, target: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = meta['padding_mask']
        pred = recover_from_vector(x)['trans']
        gt = recover_from_vector(target)['trans']

        l1 = (pred - gt).abs().mean(dim=-1)
        return sum_flat(l1 * mask) / sum_flat(mask)


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, target: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass



class GeodesicLoss(nn.Module):
    r"""Creates a criterion that measures the distance between rotation matrices, which is
    useful for pose estimation problems.
    The distance ranges from 0 to :math:`pi`.
    See: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices and:
    "Metrics for 3D Rotations: Comparison and Analysis" (https://link.springer.com/article/10.1007/s10851-009-0161-2).

    Both `input` and `target` consist of rotation matrices, i.e., they have to be Tensors
    of size :math:`(minibatch, 3, 3)`.

    The loss can be described as:

    .. math::
        \text{loss}(R_{S}, R_{T}) = \arccos\left(\frac{\text{tr} (R_{S} R_{T}^{T}) - 1}{2}\right)

    Args:
        eps (float, optional): term to improve numerical stability (default: 1e-7). See:
            https://github.com/pytorch/pytorch/issues/8069.

        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``

    Shape:
        - Input: Shape :math:`(N, 3, 3)`.
        - Target: Shape :math:`(N, 3, 3)`.
        - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`. Otherwise, scalar.
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return geodesic(input, target, self.reduction, self.eps)