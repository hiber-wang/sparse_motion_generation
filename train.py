import argparse

from utils.script_util import (create_model_and_diffusion,
                               create_controlnet,
                               create_dataloader,
                               parse_yaml,
                               default_config,
                               merge_config)

from utils.training_loop import TrainLoop
from model import build_loss_fns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    cfgs = parse_yaml(args.config)
    cfgs = merge_config(default_config(), cfgs)

    model, diffusion = create_model_and_diffusion(cfgs['denoiser'], cfgs['diffusion'])
    if cfgs['controlnet'] is None:
        controlnet = None
    else:
        # Overwrite denoiser configs with ControlNet configs
        controlnet_cfg = merge_config(cfgs['denoiser'], cfgs['controlnet'])
        controlnet = create_controlnet(controlnet_cfg)
    data = create_dataloader(cfgs['data'])
    
    loss_fn = build_loss_fns(cfgs['losses'])
    TrainLoop(
        model=model,
        diffusion=diffusion,
        controlnet=controlnet,
        data=data,
        loss_fn=loss_fn,
        **cfgs['training']
    ).run_loop()


if __name__ == '__main__':
    main()
