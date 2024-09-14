from typing import Dict, Any

from torch.utils.data import ConcatDataset, Dataset, DataLoader

from .amass import AMASS
from .beat2 import BEAT2
from .augmentation import masked, RandomCameraProjection, RandomRotation


# A registry for all supported datasets
_datasets = {
    'amass': AMASS,
    'beat2': BEAT2
}

_augmentations = {
    'random_projection': RandomCameraProjection,
    'random_rotation': RandomRotation,
    'mask': masked,
}


def build_dataset(data_configs: Dict[str, Any]) -> Dataset:
    if 'augmentation' in data_configs:
        augmentation = [_augmentations[config['type']](**config['args']) if 'args' in config
                        else _augmentations[config['type']]()
                        for config in data_configs['augmentation']]
    else:
        augmentation = None
    datasets = [
        _datasets[config['dataset']](**config['args'], augmentors=augmentation)
        for config in data_configs['datasets']
    ]

    return ConcatDataset(datasets)


def create_dataloader(data_configs: Dict[str, Any]) -> DataLoader:
    dataset = build_dataset(data_configs)
    dataloader = DataLoader(dataset, 
                            batch_size=data_configs['batch_size'])
    
    return dataloader