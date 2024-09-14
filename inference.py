import argparse
import os
import torch
import numpy as np
from safetensors.torch import load_model

from utils.script_util import (default_config,
                               parse_yaml,
                               merge_config,
                               create_model_and_diffusion,
                               create_controlnet)
from utils.representation import to_blender_npy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--condition', type=str, required=False)
    parser.add_argument('--num_frames', type=int, default=196)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    cfgs = parse_yaml(args.config)
    cfgs = merge_config(default_config(), cfgs)

    condition = torch.load(args.condition) if not args.condition is None else None
    if not condition is None:
        for k, v in condition.items():
            condition[k] = v.to(device=args.device)

    model, diffusion = create_model_and_diffusion(cfgs['denoiser'], cfgs['diffusion'])
    if cfgs['controlnet'] is None:
        controlnet = None
    else:
        # Overwrite denoiser configs with ControlNet configs
        controlnet_cfg = merge_config(cfgs['denoiser'], cfgs['controlnet'])
        controlnet = create_controlnet(controlnet_cfg)

    checkpoint = cfgs['training']['save_dir']
    load_model(model, os.path.join(checkpoint, 'model.safetensors'))
    model.eval()
    if not controlnet is None:
        load_model(controlnet, os.path.join(checkpoint, 'model_1.safetensors'))
        controlnet = controlnet.to(args.device)
        controlnet.eval()

    
    sample = diffusion.p_sample_loop(
        model.to(args.device),
        (1, args.num_frames, model.input_dim),
        device=args.device,
        progress=True,
        clip_denoised=False,
        model_kwargs=condition,
        controlnet=controlnet,
    )

    np.savez('sample', **to_blender_npy(sample))


if __name__ == '__main__':
    main()