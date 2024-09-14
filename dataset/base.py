from typing import Dict, Tuple, Any, Iterable, Callable, Optional

import torch
from torch.utils.data import Dataset
from glob import glob
import os

from utils.representation import to_vector

# Names for motion related fields
_motion = ['rotation', 'position', 'trans']
# Names that should not be padded
_nopad = ['K', 'R', 'T', 'scale', 'offset']


def pad_sequence(data: torch.Tensor, padded_len: int) -> torch.Tensor:
    return torch.cat([
        data,
        torch.zeros((padded_len - data.shape[0],) + data.shape[1:], dtype=data.dtype)
    ], dim=0)


class BaseMotionDataset(Dataset):
    def __init__(self, data_dir: str, num_frames: int, include_pos: bool = True, augmentors: Optional[Iterable[Callable]] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_frames = num_frames
        self.files = dict(enumerate(glob(os.path.abspath(os.path.join(data_dir, '*.pt')))))
        self.cache = {k: None for k in self.files.keys()}
        self.include_pos = include_pos
        self.augmentors = augmentors

    
    def __len__(self) -> int:
        return len(self.files)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.cache[idx] is None:
            data = torch.load(self.files[idx])
            self.cache[idx] = data
        data = self.cache[idx]
        data = self.process_data(data) # type: ignore
        # Augment data
        if not self.augmentors is None:
            for augmentor in self.augmentors:
                data = augmentor(data)
        # Pad short sequences
        for k, v in data.items():
            if not k in _nopad:
                data[k] = pad_sequence(v, self.num_frames)
        # Extract motion data and form vector representation
        motion, cond = {k: v for k, v in data.items() if k in _motion}, {k: v for k, v in data.items() if not k in _motion}
        motion = to_vector(motion, include_pos=self.include_pos)
        return motion, cond


    def process_data(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError('Children classes suppose to implement this method')
