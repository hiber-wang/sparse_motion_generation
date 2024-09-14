from typing import Dict
import torch
import numpy as np

from utils.rotation_conversion import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix


class RandomRotation:
    # Randomly rotate motion along y-axis
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        angle_y = torch.rand(1) * 2 * np.pi
        perturb = axis_angle_to_matrix(torch.tensor([[0., angle_y, 0.]]))
        glob_orient = matrix_to_rotation_6d(perturb @ rotation_6d_to_matrix(data['rotation'][:, 0]))
        data['rotation'][:, 0] = glob_orient
        trans = data['trans']
        trans = (perturb @ trans.unsqueeze(dim=-1)).squeeze(dim=-1)
        data['trans'] = trans

        return data