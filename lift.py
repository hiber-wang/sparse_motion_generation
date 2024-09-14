from typing import Tuple, List

import argparse
import os
import math
import torch
from torch.func import grad
import numpy as np
from safetensors.torch import load_model
from tqdm.auto import tqdm

from utils.script_util import (default_config,
                               parse_yaml,
                               merge_config,
                               create_model_and_diffusion
                               )
from utils.representation import (to_blender_npy,
                                  recover_from_vector, 
                                  to_vector
                                  )
from utils.rotation_conversion import axis_angle_to_matrix, matrix_to_rotation_6d
from model.loss import jitter

aa_to_r6d = lambda x: matrix_to_rotation_6d(axis_angle_to_matrix(x))


def read_data(data: np.lib.npyio.NpzFile) -> Tuple[torch.Tensor, np.ndarray]:
    # Rotate along x-axis for 180 degrees
    rot = torch.tensor([[
        [1.,  0.,  0.],
        [0., -1.,  0.,],
        [0.,  0., -1.]
    ]])
    
    trans = torch.from_numpy(data['transl'])
    trans = (rot @ trans.unsqueeze(dim=-1)).squeeze(dim=-1)
    trans -= trans[[0]]
    # trans[:, 2] = 0.0
    global_orient = axis_angle_to_matrix(torch.from_numpy(data['global_orient']))
    global_orient = matrix_to_rotation_6d(rot @ global_orient )
    poses = torch.cat([     
        global_orient,
        aa_to_r6d(torch.from_numpy(data['body_pose'])).flatten(start_dim=1),
        aa_to_r6d(torch.from_numpy(data['left_hand_pose'])).flatten(start_dim=1),
        aa_to_r6d(torch.from_numpy(data['right_hand_pose'])).flatten(start_dim=1),
    ], dim=-1)

    return torch.cat([poses, trans], dim=-1).unsqueeze(dim=0), data['betas']


def get_timesteps(start_idx: int, num_timesteps: int, end_idx: int = 0) -> List[int]:
    step_size = math.ceil((start_idx - end_idx) / num_timesteps)
    return list(
        range(end_idx, start_idx, step_size)
    )[::-1]


def jitter_score(x, t, **kw):
    scale = kw.pop('scale', 1.0)
    trans = recover_from_vector(x)['trans']
    return scale * jitter(trans).mean()


def smooth_score(x, t, **kw):
    scale = kw.pop('scale', 1.0)
    trans = recover_from_vector(x)['trans']
    score = (trans[:, 1:] - trans[:, :-1]).norm(dim=-1, p=1)
    return scale * score.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--num_frames', type=int, default=196)
    parser.add_argument('--input', type=str)
    parser.add_argument('--inference_step', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    cfgs = parse_yaml(args.config)
    cfgs = merge_config(default_config(), cfgs)

    model, diffusion = create_model_and_diffusion(cfgs['denoiser'], cfgs['diffusion'])
    data = np.load(args.input)
    x, beta = read_data(data)
    x = x.to(device=args.device)

    checkpoint = cfgs['training']['save_dir']
    load_model(model, os.path.join(checkpoint, 'model.safetensors'))
    model = model.to(args.device)
    model.eval()

    # Step 1. using inpainting to fix depth ambiguity
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :, -1] = True
    sample = diffusion.p_sample_loop(
        model.to(args.device),
        x.shape,
        device=args.device,
        progress=True,
        clip_denoised=False,
        model_kwargs={'x_clean': x, 'obs_mask': mask}
    )

    # Step 2. using DDIM inversion to smooth data
    latent = diffusion.invert(model, sample, args.inference_step, clip_denoised=False)
    sample = diffusion.ddim_sample_loop_shortened(
        model,
        x.shape,
        noise=latent,
        num_inference_steps=args.inference_step,
        clip_denoised=False,
        cond_fn=smooth_score,
        model_kwargs={'scale': 5.0}
    )
    np.savez('sample', **to_blender_npy(sample, betas=beta, include_pos=False))


if __name__ == '__main__':
    main()