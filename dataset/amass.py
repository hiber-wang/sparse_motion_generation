from typing import Dict
import torch
import numpy as np

from .base import BaseMotionDataset



class AMASS(BaseMotionDataset):
    
    def process_data(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l = data['rotation'].shape[0]
        if l < self.num_frames:
            # pad sequence
            time_slice = slice(None)
            mask = torch.ones((l,), dtype=torch.bool)
        else:
            start = np.random.randint(0, l - self.num_frames + 1)
            time_slice = slice(start, start+self.num_frames)
            mask = torch.ones((self.num_frames,), dtype=torch.bool)

        rot = data['rotation'][time_slice]
        pos = data['position'][time_slice]
        trans = data['trans'][time_slice]
        offset = trans[[0]]
        return {
            'rotation': rot,
            'position': pos - offset.unsqueeze(dim=1),
            'trans': trans - offset,
            'padding_mask': mask,
            'offset': offset,
        }