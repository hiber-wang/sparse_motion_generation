from typing import Dict, Union

import torch
import numpy as np

from utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle


def to_vector(data_dict: Dict[str, torch.Tensor], include_pos: bool = True) -> torch.Tensor:
    rot = data_dict['rotation'].flatten(start_dim=-2)
    trans = data_dict['trans']    
    if include_pos:
        pos = data_dict['position'][..., list(range(0, 22)) + list(range(25, 55)), :].flatten(start_dim=-2)
        return torch.cat([rot, trans, pos], dim=-1)
    else:
        return torch.cat([rot, trans], dim=-1)



def recover_from_vector(vec: torch.Tensor, num_joints: int = 52, include_pos: bool = True) -> Dict[str, torch.Tensor]:
    rot = vec[..., :num_joints * 6].reshape(*vec.shape[:-1], -1, 6)
    trans = vec[..., num_joints * 6 : num_joints * 6 + 3]
    if include_pos:
        pos = vec[..., num_joints * 6 + 3:].reshape(*vec.shape[:-1], -1, 3)

        return {
            'rotation': rot,
            'position': pos,
            'trans': trans,
        }
    
    else:
        return {
            'rotation': rot,
            'trans': trans,
        }



BLENDER_JOINTS = 55

def to_blender_npy(vec: torch.Tensor, 
                   num_joints: int = 52,
                   include_pos: bool = True,
                   fps: int = 30, 
                   gender: str = 'neutral',
                   betas: np.ndarray = np.zeros((10,))) -> Dict[str, Union[np.ndarray, str]]:
    r6d_to_aa = lambda x: matrix_to_axis_angle(rotation_6d_to_matrix(x))
    data_dict = recover_from_vector(vec, num_joints=num_joints, include_pos=include_pos)
    for k, v in data_dict.items():
        data_dict[k] = v.squeeze()
    
    assert len(data_dict['rotation'].shape) == 3, 'to_blender_npy only supports converting single motion sequence to blender SMPL-X format'
    poses = r6d_to_aa(data_dict['rotation'])
    num_inserted = BLENDER_JOINTS - num_joints
    poses = torch.cat([poses[:, :22], torch.zeros((poses.shape[0], num_inserted, 3), device=poses.device, dtype=poses.dtype), poses[:, 22:]], dim=1).flatten(start_dim=1).detach().cpu().numpy()
    trans = data_dict['trans'].detach().cpu().numpy()
    
    return {
        'poses': poses,
        'trans': trans,
        'gender': gender,
        'mocap_framerate': np.array(fps),
        'betas': betas,
    }



def euclidean_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    bol_a = (a[:, 0] != 0).to(dtype=torch.int)
    bol_b = (b[:, 0] != 0).to(dtype=torch.int)
    dist = torch.norm(a - b, dim=1)
    return((dist*bol_a*bol_b).reshape(a.shape[0],1))


def normalize_keypoints(X):
    shape = X.shape
    num_sample = X.shape[0]
    X = X.reshape((num_sample, -1))
    # Keypoints
    Nose = X[:,0*2:0*2+2]
    Neck = X[:,1*2:1*2+2]
    RShoulder = X[:,2*2:2*2+2]
    RElbow = X[:,3*2:3*2+2]
    RWrist = X[:,4*2:4*2+2]
    LShoulder = X[:,5*2:5*2+2]
    LElbow = X[:,6*2:6*2+2]
    LWrist = X[:,7*2:7*2+2]
    RHip = X[:,8*2:8*2+2]
    RKnee = X[:,9*2:9*2+2]
    RAnkle = X[:,10*2:10*2+2]
    LHip = X[:,11*2:11*2+2]
    LKnee = X[:,12*2:12*2+2]
    LAnkle = X[:,13*2:13*2+2]
    REye = X[:,14*2:14*2+2]
    LEye = X[:,15*2:15*2+2]
    REar = X[:,16*2:16*2+2]
    LEar = X[:,17*2:17*2+2]

    # Length of head
    length_Neck_LEar = euclidean_dist(Neck, LEar)
    length_Neck_REar = euclidean_dist(Neck, REar)
    length_Neck_LEye = euclidean_dist(Neck, LEye)
    length_Neck_REye = euclidean_dist(Neck, REye)
    length_Nose_LEar = euclidean_dist(Nose, LEar)
    length_Nose_REar = euclidean_dist(Nose, REar)
    length_Nose_LEye = euclidean_dist(Nose, LEye)
    length_Nose_REye = euclidean_dist(Nose, REye)

    length_head = torch.max(torch.cat([
        length_Neck_LEar,
        length_Neck_REar,
        length_Neck_LEye, 
        length_Neck_REye,
        length_Nose_LEar, 
        length_Nose_REar,
        length_Nose_LEye,
        length_Nose_REye
    ], dim=1), dim=1)[0]

    length_Neck_LHip = euclidean_dist(Neck, LHip)
    length_Neck_RHip = euclidean_dist(Neck, RHip)
    length_torso = torch.max(torch.cat([
        length_Neck_LHip, length_Neck_RHip
    ], dim=1), dim=1)[0]

    # Length of right leg
    length_leg_right = euclidean_dist(RHip, RKnee) + euclidean_dist(RKnee, RAnkle)

    # Length of left leg
    length_leg_left = euclidean_dist(LHip, LKnee) + euclidean_dist(LKnee, LAnkle)

    # Length of leg
    length_leg = torch.max(torch.cat([
        length_leg_left, length_leg_right
    ], dim=1), dim=1)[0]

    # Length of body
    length_body = length_head + length_torso + length_leg
    length_body = length_body.unsqueeze(dim=-1)
    
    # Check all samples have length_body of 0
    length_chk = (length_body > 0).to(dtype=torch.int)
    
    # Check keypoints at origin
    keypoints_chk = (X > 0).to(dtype=torch.int)
    
    chk = length_chk * keypoints_chk
    
    # Set all length_body of 0 to 1 (to avoid division by 0)
    length_body[length_body.abs() < 1e-3] = 1

    # The center of gravity
    # number of point OpenPose locates:
    num_pts = (X[:, 0::2] != 0).sum(1).reshape(num_sample,1)
    centr_x = (X[:, 0::2] / num_pts).sum(1).reshape(num_sample, 1)
    centr_y = (X[:, 1::2] / num_pts).sum(1).reshape(num_sample, 1)
    # centr_x = X[:, 0::2].sum(1).reshape(num_sample,1) / num_pts
    # centr_y = X[:, 1::2].sum(1).reshape(num_sample,1) / num_pts

    # The  coordinates  are  normalized relative to the length of the body and the center of gravity
    X_norm_x = (X[:, 0::2] - centr_x) / length_body
    X_norm_y = (X[:, 1::2] - centr_y) / length_body

    # Stack 1st element x and y together
    X_norm = torch.cat((X_norm_x[:, :1], X_norm_y[:, :1]), dim=1)

        
    for i in range(1, X.shape[1]//2):
        X_norm = torch.cat((X_norm, X_norm_x[:, i:i+1], X_norm_y[:, i:i+1]), dim=1)

    # Set all samples have length_body of 0 to origin (0, 0)
    X_norm = X_norm * chk
    
    return X_norm.reshape(shape)
