from typing import List, Dict, Any
from torch import nn

from .smpl import SMPLX
from .dit import MotionDiT, MaskDiT
from .controlnet import ControlNet
from .loss import (FlatMSELoss, 
                   ReprojectionLoss, 
                   VelocityLoss, 
                   GeometricLoss, 
                   WeightedMSELoss,
                   WeightedVelocityLoss, 
                   TranslationLoss,
                   GeodesicLoss,)


_loss_fns = {
    'flat_mse': FlatMSELoss,
    'reprojection': ReprojectionLoss,
    'velocity': VelocityLoss,
    'geometric': GeometricLoss,
    'weighted_mse': WeightedMSELoss,
    'weighted_velocity': WeightedVelocityLoss,
    'translation': TranslationLoss,
    'geodesic': GeodesicLoss,
}

_denoisers = {
    'dit': MotionDiT,
    'maskdit': MaskDiT,
}



class CombinedLoss(nn.Module):
    def __init__(self, loss_fns: List[nn.Module], scales: List[float]):
        super().__init__()
        assert len(loss_fns) == len(scales)
        self.loss_fns = nn.ModuleList(loss_fns)
        self.scales = scales

    def forward(self, x, target, meta):
        loss = 0
        for loss_fn, scale in zip(self.loss_fns, self.scales):
            loss += scale * loss_fn(x, target, meta)
        return loss


def build_loss_fns(loss_fn_cfgs: List[Dict[str, Any]]) -> nn.Module:
    loss_fns = []
    scales = []
    for loss_fn_cfg in loss_fn_cfgs:
        args = loss_fn_cfg.get('args', {})
        loss_fns.append(_loss_fns[loss_fn_cfg['loss_fn']](**args))
        scales.append(loss_fn_cfg['scale'])
    return CombinedLoss(loss_fns, scales)


def build_denoiser(denoiser_cfg: Dict[str, Any]):
    return _denoisers[denoiser_cfg['arch']](
        **denoiser_cfg
    )