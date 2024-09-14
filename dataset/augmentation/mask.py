from typing import Callable, List, Union, Tuple, Any, Iterable
from functools import reduce

import torch
import torch.utils


def to_tuple(x: Any) -> Tuple:
    if isinstance(x, str) or not isinstance(x, Iterable):
        return (x,)
    return tuple(x)


def _get_mask_scheme(name: str) -> Callable:
    masks = {
        'random_frames': random_frames,
        'random_joints': random_joints,
        'random_keyframes': random_keyframes,
        'random_keyjoints': random_keyjoints,
        'spaced_frames': spaced_frames,
        'specified_joints': specified_joints,
    }
    
    return masks[name]



def _compose(*mask_fns):
    def compose2(f, g):
        return lambda *a, **kw: f(*a, **kw) * g(*a, **kw)
    return reduce(compose2, mask_fns)



def masked(schemes: Union[str, Iterable[str]] = 'random_frames', *a, **kw) -> Callable:
    schemes = to_tuple(schemes)
    mask_fn = _compose(*tuple(_get_mask_scheme(s) for s in schemes))
    def _decorator(klass: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
        class MaskedDataset(klass):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __getitem__(self, idx):
                res = super().__getitem__(idx)
                return res, {'obs_mask': mask_fn(res, *a, **kw), 'clean': res}

        return MaskedDataset
    return _decorator



def random_frames(x: torch.Tensor, mask_prob: float = 0.1, **kw) -> torch.BoolTensor:
    """Randomly mask out frames with probability `prob`

    Args:
        x (torch.Tensor): the tensor to be masked
        mask_prob (float): masking probability

    Returns:
        torch.BoolTensor: the mask for tensor `x`
    """
    b, t, *d = x.shape
    mask = torch.ones_like(x, dtype=torch.bool)
    obs = torch.bernoulli(torch.ones((b, t, *(1 for _ in d)), device=x.device) * mask_prob).bool()
    return mask * ~obs


def random_keyframes(x: torch.Tensor, num_keyframes: int = 20, **kw) -> torch.BoolTensor:
    """Randomly select frames to be keyframe

    Args:
        x (torch.Tensor): the tensor to be masked
        num_keyframes (int, optional): number of keyframes. Defaults to 20.

    Returns:
        torch.BoolTensor: the mask for tensor `x`
    """
    _, t, *_ = x.shape
    mask = torch.zeros_like(x, dtype=torch.bool)
    for sub in mask:
        idxs = torch.randperm(t)[:num_keyframes]
        sub[idxs] = True
    return mask.bool()


def random_joints(x: torch.Tensor, mask_prob: float = 0.1, num_joints: int = 52, **kw) -> torch.BoolTensor:
    """Randomly mask out joints with probability `prob`

    Args:
        x (torch.Tensor): the tensor to be masked
        mask_prob (float): masking probability
        num_joints (int): the number of joints

    Returns:
        torch.BoolTensor: the mask for tensor `x`
    """
    b, t, *d = x.shape
    mask = torch.ones_like(x.view(b, t, num_joints, d[0] // num_joints, *d[1:]), dtype=torch.bool)
    obs = torch.bernoulli(torch.ones((b, 1, num_joints, *(1 for _ in d)), device=x.device) * mask_prob).bool()
    return (mask * ~obs).reshape(b, t, *d)


def random_keyjoints(x: torch.Tensor, num_joints: int = 52, num_keyjoints: int = 4, **kw) -> torch.BoolTensor:
    """Randomly select joints within each frame as keyjoints, and mask all other joints

    Args:
        x (torch.Tensor): the tensor to be masked
        num_joints (int, optional): the total number of joints. Defaults to 52.
        num_keyjoints (int, optional): the number of keyjoints to be selected. Defaults to 4.

    Returns:
        torch.BoolTensor: the mask for tensor `x`
    """

    ## TODO: avoid nested loop
    b, t, *d = x.shape
    mask = torch.zeros_like(x.view(b, t, num_joints, d[0] // num_joints, *d[1:]), dtype=torch.bool)
    for sub in mask:
        for frame in sub:
            idxs = torch.randperm(num_keyjoints)
            frame[idxs] = True
    return mask


def spaced_frames(x: torch.Tensor, every: int = 10, **kw) -> torch.BoolTensor:
    """Mask all frames except evenly spaced ones

    Args:
        x (torch.Tensor): the tensor to be masked
        every (int, optional): visible frame. Defaults to 10.

    Returns:
        torch.BoolTensor: the mask for tensor `x`
    """
    _, t, *_ = x.shape
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, range(0, t, every)] = True
    return mask



def specified_joints(x: torch.Tensor, num_joints: int = 52, joint_idxs: List[int] = list(), *kw) -> torch.BoolTensor:
    b, t, *d = x.shape
    mask = torch.zeros_like(x.view(b, t, num_joints, d[0] // num_joints, *d[1:]), dtype=torch.bool)
    mask[:, :, joint_idxs] = True
    return mask.reshape((b, t, *d))
