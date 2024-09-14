from typing import Union, Tuple, Callable
import torch

from model import SMPLX
from model.loss import geman_mcclure, jitter
from utils.representation import recover_from_vector
from utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
from utils.camera import perspective_projection, normalize_coordinate
from utils.keypoints_conversion import smpl_to_openpose


r6d_to_aa = lambda x: matrix_to_axis_angle(rotation_6d_to_matrix(x))


def reprojection(smpl: SMPLX, 
                 x: torch.Tensor, 
                 beta: torch.Tensor, 
                 cam_rotation: torch.Tensor,
                 cam_translation: torch.Tensor, 
                 cam_intrinsics: torch.Tensor,
                 ) -> torch.Tensor:
    data_dict = recover_from_vector(x)
    poses, trans = data_dict['rotation'], data_dict['trans']
    poses = r6d_to_aa(poses)
    kp2d = perspective_projection(smpl.forward_kinematics(poses, transl=trans, betas=beta), K=cam_intrinsics, rotation=cam_rotation, translation=cam_translation)
    kp2d = smpl_to_openpose(kp2d)
    return normalize_coordinate(kp2d, cam_intrinsics)


def camera_fitting(kp2d_gt: torch.Tensor,
                   smpl: SMPLX, 
                   x: torch.Tensor,
                   beta: torch.Tensor,
                   cam_rotation: torch.Tensor,
                   cam_translation: torch.Tensor,
                   cam_intrinsics: torch.Tensor,
                   sigma: float,
                   lam_reproj: float, 
                   ) -> torch.Tensor:
    kp2d_est = reprojection(smpl, x, beta, cam_rotation, cam_translation, cam_intrinsics)
    reproj_error = lam_reproj * geman_mcclure(kp2d_est - kp2d_gt, sigma).mean()
    
    return reproj_error


def pose_fitting(kp2d_gt: torch.Tensor,
                 smpl: SMPLX,
                 x: torch.Tensor,
                 beta: torch.Tensor,
                 cam_rotation: torch.Tensor,
                 cam_translation: torch.Tensor,
                 cam_intrinsics: torch.Tensor,
                 sigma: float,
                 lam_reproj: float,
                 lam_shape: float,
                 lam_jitter: float,
                 ) -> torch.Tensor:
    kp2d_est = reprojection(smpl, x, beta, cam_rotation, cam_translation, cam_intrinsics)
    reproj_error = lam_reproj * geman_mcclure(kp2d_est - kp2d_gt, sigma).mean()
    shape_loss = lam_shape * (beta ** 2).sum(dim=-1).mean()
    data_dict = recover_from_vector(x)
    poses, trans = data_dict['rotation'], data_dict['trans']
    smooth_loss = lam_jitter * (jitter(poses).mean() + jitter(trans).mean())

    return reproj_error + shape_loss + smooth_loss


def create_closure(optimizer: torch.optim.Optimizer,
                   loss_fn: Callable,
                   params: Tuple,
                   ) -> Callable:
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(*params)
        loss.backward()
        return loss

    return closure



class TemporalSMPLify:
    def __init__(self, 
                 smpl: SMPLX,
                 lr: float = 1e-2,
                 num_iter: int = 5,
                 num_steps: int = 10,
                 sigma: float = 100.0,
                 lam_reproj: float = 100.0,
                 lam_shape: float = 1.0,
                 lam_jitter: float = 20.0,
                 device: Union[str, torch.device] = 'cuda'):
        self.smpl = smpl.to(device=device)
        self.device = device
        self.lr = lr
        self.num_iter = num_iter
        self.num_steps = num_steps
        self.sigma = sigma
        self.lam_reproj = lam_reproj
        self.lam_shape = lam_shape
        self.lam_jitter = lam_jitter

    def __call__(self,
                 kp2d_gt: torch.Tensor,
                 smpl: SMPLX,
                 x_init: torch.Tensor,
                 beta_init: torch.Tensor,
                 cam_rotation_init: torch.Tensor,
                 cam_translation_init: torch.Tensor,
                 cam_intrinsics: torch.Tensor,
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Initial esitmations
        x_init = x_init.detach()
        beta = beta_init.detach()
        cam_rotation_init = cam_rotation_init.detach()
        cam_translation_init = cam_translation_init.detach()
        kp2d_gt = normalize_coordinate(kp2d_gt, cam_intrinsics)  # normalize coordinate to range [0, 1]
        # Parmameters to optimize
        x = x_init.clone()
        cam_rotation = cam_rotation_init.clone()
        cam_translation = cam_translation_init.clone()
        # Stage 1. Optimize camera parameters
        optimizer = torch.optim.LBFGS(
            [cam_rotation, cam_translation],
            lr=self.lr,
            max_iter=self.num_iter,
            line_search_fn='strong_wolfe'
        )
        
        cam_closure = create_closure(
            optimizer,
            camera_fitting,
            (kp2d_gt,
             smpl,
             x,
             beta,
             cam_rotation,
             cam_translation,
             cam_intrinsics,
             self.sigma,
             self.lam_reproj,
            )
        )

        for _ in range(self.num_steps):
            optimizer.zero_grad()
            optimizer.step(cam_closure)

        # Stage 2. Optimize pose parameters
        optimizer = torch.optim.LBFGS(
            [x, beta],
            lr=self.lr,
            max_iter=self.num_iter,
            line_search_fn='strong_wolfe'
        )

        pose_closure = create_closure(
            optimizer,
            pose_fitting,
            (kp2d_gt,
             smpl,
             x,
             beta,
             cam_rotation,
             cam_translation,
             cam_intrinsics,
             self.sigma,
             self.lam_reproj,
             self.lam_shape,
             self.lam_jitter
            )
        )

        for _ in range(self.num_steps):
            optimizer.zero_grad()
            optimizer.step(pose_closure)

        return (x, beta, cam_rotation, cam_translation)