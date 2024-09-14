from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import yaml
from pathlib import Path
from yaml.loader import Reader, Scanner, Parser, Composer, SafeConstructor, Resolver # type: ignore
from torch.utils.data import DataLoader

from diffusion import create_diffusion
from model import build_denoiser, ControlNet
from dataset import build_dataset



@dataclass
class DiffusionConfig:
    timestep_respacing: str = ''
    noise_schedule: str = 'linear'
    use_kl: bool = False
    sigma_small: bool = False
    predict_xstart: bool = False
    learn_sigma: bool = False
    rescale_learned_sigmas: bool = False
    diffusion_steps: int = 1000


@dataclass
class DenoiserConfig:
    arch: str = 'dit'
    input_dim: int = 471
    hidden_dim: int = 512
    num_layers: int = 16
    mlp_ratio: float = 4.0
    cond_dim: int = 130


class ControlNetConfig(DenoiserConfig):
    pass


@dataclass
class TrainingConfig:
    num_epochs: int = 1000_0000
    lr: float = 1e-4
    ema_rate: float = 0.9999
    save_dir: str = ''
    log_interval: int = 100
    save_interval: int = 100
    mixed_precision: str = 'no'
    resume_checkpoint: str = ''
    weight_decay: float = 0.0


@dataclass
class Config:
    denoiser: DenoiserConfig
    diffusion: DiffusionConfig
    controlnet: Optional[ControlNetConfig] = None
    data: Optional[Dict[str, Any]] = None
    losses: Optional[Dict[str, Any]] = None
    training: Optional[TrainingConfig] = None
    

default_config = lambda: asdict(Config(
    DenoiserConfig(),
    DiffusionConfig(),
    None,
    None,
    None,
    TrainingConfig()
))



def create_model_and_diffusion(model_cfg: Dict[str, Any], diffusion_cfg: Dict[str, Any]):
    model = build_denoiser(model_cfg)
    diffusion = create_diffusion(**diffusion_cfg)
    return model, diffusion



def create_controlnet(model_cfg: Dict[str, Any]):
    controlnet = ControlNet(**model_cfg)
    return controlnet



def create_dataloader(data_configs: Dict[str, Any]) -> DataLoader:
    dataset = build_dataset(data_configs)
    dataloader = DataLoader(dataset, 
                            batch_size=data_configs['batch_size'])
    
    return dataloader


class StrictBoolSafeResolver(Resolver):
    pass

# remove resolver entries for On/Off/Yes/No
for ch in "OoYyNn":
    if len(StrictBoolSafeResolver.yaml_implicit_resolvers[ch]) == 1:
        del StrictBoolSafeResolver.yaml_implicit_resolvers[ch]
    else:
        StrictBoolSafeResolver.yaml_implicit_resolvers[ch] = [x for x in
                StrictBoolSafeResolver.yaml_implicit_resolvers[ch] if x[0] != 'tag:yaml.org,2002:bool']

class StrictBoolSafeLoader(Reader, Scanner, Parser, Composer, SafeConstructor, StrictBoolSafeResolver):
    def __init__(self, stream):
        Reader.__init__(self, stream)
        Scanner.__init__(self)
        Parser.__init__(self)
        Composer.__init__(self)
        SafeConstructor.__init__(self)
        StrictBoolSafeResolver.__init__(self)


def parse_yaml(fpath):
    """ Parse stream using StrictBoolSafeLoader. """
    return yaml.load(Path(fpath).read_text(), Loader=StrictBoolSafeLoader)


def merge_config(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in new.items():
        if isinstance(value, dict) and key in old and isinstance(old[key], dict):
            old[key] = merge_config(old[key], value)
        else:
            old[key] = value

    return old