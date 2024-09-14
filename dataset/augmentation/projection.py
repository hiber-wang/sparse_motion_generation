from typing import Dict, Tuple, Optional

import torch
import numpy as np

from utils.rotation_conversion import (
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)
from utils.keypoints_conversion import smpl_to_openpose
from utils.representation import normalize_keypoints
from utils.camera import perspective_projection, normalize_coordinate


class RandomCameraProjection:
    rx_factor = np.pi / 16
    ry_factor = np.pi / 4
    rz_factor = np.pi / 16
    
    pitch_std = np.pi / 12
    pitch_mean = np.pi / 36
    roll_std = np.pi / 24
    t_factor = 1
    
    tx_std = 0.25
    ty_std = 0.05
    ty_mean = 0.1
    tz_scale = 3
    tz_min = 3
    
    # motion_prob = 0.75
    motion_prob = 0.
    interp_noise = 0.2

    def __init__(self, h: int, w: int, f: Optional[float] = None, normalize_kp: bool = True) -> None:
        f = f or (h * h + w * w) ** 0.5
        self.w, self.h, self.f = h, w, f
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = self.f # type: ignore
        K[1, 1] = self.f # type: ignore
        K[0, 2] = self.w / 2
        K[1, 2] = self.h / 2
        self.K = K.unsqueeze(dim=0)

        self.fov_tol = 1.2 * (0.5 ** 0.5)
        self.normalize_kp = normalize_kp


    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        kp3d = data['position']
        transl = data['trans']
        rotation = data['rotation']
        kp3d -= transl.unsqueeze(dim=-2)

        l = kp3d.shape[0]
        R, T = self.create_camera()
        R, T = R.to(dtype=transl.dtype, device=transl.device), T.to(dtype=transl.dtype, device=transl.device)

        if np.random.rand() < self.motion_prob:
            R = self.create_rotation_move(R, l)
            T = self.create_translation_move(T, l)

        # Translate global orientation, joint positions, and translation by camera matrix [R|T]
        transl = torch.matmul(R, transl.unsqueeze(dim=-1)).squeeze(dim=-1)
        T = self.normalize_camera_placement(transl, T)
        kp3d = torch.einsum('...ij,...kj->...ki', R, kp3d)
        kp3d = kp3d + (T + transl).unsqueeze(dim=-2)
        global_orient = matrix_to_rotation_6d(R @ rotation_6d_to_matrix(rotation[:, 0]))
        rotation[:, 0] = global_orient
        # Update motion data
        data['position'] = kp3d
        data['trans'] = T + transl
        data['rotation'] = rotation

        # Get 2D projected keypoints
        kp2d = perspective_projection(kp3d, K=self.K)
        kp2d = smpl_to_openpose(kp2d)

        data['kp2d'] = kp2d
        data['K'] = self.K
        data['R'] = R
        data['T'] = T

        # Projected 2D points + camera pose to constitute the condition
        if self.normalize_kp:
            data['condition'] = normalize_keypoints(kp2d).flatten(start_dim=1)
        else:
            data['condition'] = normalize_coordinate(kp2d, self.K).flatten(start_dim=1)
        return data

    
    def create_camera(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # yaw = np.random.rand() * 2 * np.pi
        yaw = np.random.randn() * (np.pi / 6)
        pitch = np.random.normal(scale=self.pitch_std) + self.pitch_mean
        roll = np.random.normal(scale=self.roll_std)
        
        yaw_rm = axis_angle_to_matrix(torch.tensor([[0, yaw, 0]]))
        pitch_rm = axis_angle_to_matrix(torch.tensor([[pitch, 0, 0]]))
        roll_rm = axis_angle_to_matrix(torch.tensor([[0, 0, roll]]))
        R = (roll_rm @ pitch_rm @ yaw_rm)

        # Place people in the scene
        tz = np.random.rand() * self.tz_scale + self.tz_min
        max_d = self.w * tz / self.f / 2 # type: ignore
        tx = np.random.normal(scale=self.tx_std) * max_d
        ty = np.random.normal(scale=self.ty_std) * max_d + self.ty_mean
        T = torch.tensor([[tx, ty, tz]])

        return R, T


    def create_rotation_move(self, R: torch.Tensor, l: int) -> torch.Tensor:
        """Create rotational move for the camera"""
        
        # Create final camera pose
        rx = np.random.normal(scale=self.rx_factor)
        ry = np.random.normal(scale=self.ry_factor)
        rz = np.random.normal(scale=self.rz_factor)
        Rf = R[0] @ axis_angle_to_matrix(torch.tensor([rx, ry, rz], dtype=R.dtype))
        
        # Inbetweening two poses
        Rs = torch.stack((R[0], Rf))
        rs = matrix_to_rotation_6d(Rs).numpy() 
        rs_move = self.noisy_interpolation(rs, l)
        R_move = rotation_6d_to_matrix(torch.from_numpy(rs_move).to(dtype=R.dtype))
        return R_move
    
    
    def create_translation_move(self, T: torch.Tensor, l: int) -> torch.Tensor:
        """Create translational move for the camera"""
        
        # Create final camera position
        tx = np.random.normal(scale=self.t_factor)
        ty = np.random.normal(scale=self.t_factor)
        tz = np.random.normal(scale=self.t_factor)
        Ts = np.array([[0, 0, 0], [tx, ty, tz]])
        
        T_move = self.noisy_interpolation(Ts, l)
        T_move = torch.from_numpy(T_move).to(dtype=T.dtype)
        return T_move + T
    

    def noisy_interpolation(self, data: np.ndarray, l: int) -> np.ndarray:
        """Non-linear interpolation with noise"""
        
        dim = data.shape[-1]
        output = np.zeros((l, dim))
        
        linspace = np.stack([np.linspace(0, 1, l) for _ in range(dim)])
        noise = (linspace[0, 1] - linspace[0, 0]) * self.interp_noise
        space_noise = np.stack([np.random.uniform(-noise, noise, l - 2) for _ in range(dim)])
        
        linspace[:, 1:-1] = linspace[:, 1:-1] + space_noise
        for i in range(dim):
            output[:, i] = np.interp(linspace[i], np.array([0., 1.,]), data[:, i])
        return output
    

    def normalize_camera_placement(self, transl_human: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        # Human translation in camera coordinate
        transl = transl_human + T
        if transl[..., 2].min() < 1.0:
            T[..., 2] = T[..., 2] + (1.0 - transl[..., 2].min())
            transl = T + transl_human
        
        # If the subject is away from the FoV, put the camera behind
        fov = torch.div(transl[..., :2], transl[..., 2:].abs())
        if fov.max() > self.fov_tol:
            t_max = transl[fov.max(1)[0].max(0)[1].item()]  # type: ignore
            z_trg = t_max[:2].abs().max(0)[0] / self.fov_tol
            pad = z_trg - t_max[2]
            T = T + pad
        
        return T
