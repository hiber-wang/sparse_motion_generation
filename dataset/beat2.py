from typing import Dict

import torch
import numpy as np

from .base import BaseMotionDataset



class BEAT2(BaseMotionDataset):

    def process_data(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l = data['rotation'].shape[0]
        start = np.random.randint(0, l - self.num_frames + 1)
        rot = data['rotation'][start:start+self.num_frames]
        pos = data['position'][start:start+self.num_frames]
        trans = data['trans'][start:start+self.num_frames]
        mask = torch.ones((self.num_frames,), dtype=torch.bool)
        offset = trans[[0]]
        
        return {
            'rotation': rot,
            'position': pos - offset.unsqueeze(dim=1),
            'trans': trans - offset,
            'padding_mask': mask,
            'offset': offset,
        }