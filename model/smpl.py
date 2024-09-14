from typing import List, Optional

from functools import partial
import torch
from model.smplx import SMPLX as _SMPLX


class SMPLX(_SMPLX):
    """A thin wrapper layer of SMPLX
    """
    # Constant definitions

    ## Joints indices
    ROOT_IDX = [0]
    BODY_IDXS = list(range(1,22))
    LEFT_HAND_IDXS = list(range(25, 40))
    RIGHT_HAND_IDXS = list(range(40, 55))
    HANDS_IDXS = LEFT_HAND_IDXS + RIGHT_HAND_IDXS

    ## Kinematic chains
    BODY_CHAIN = [
        [0, 2, 5, 8, 11],      # pelvis to right leg
        [0, 1, 4, 7, 10],      # pelvis to left leg
        [0, 3, 6, 9, 12, 15],  # pelvis to head
        [9, 14, 17, 19, 21],   # spine3 to right hand
        [9, 13, 16, 18, 20],   # spine3 to left hand
    ]

    LEFT_HAND_CHAIN = [
        [20, 22, 23, 24],      # left wrist to left index3
        [20, 34, 25, 36],      # left wrist to left thumb3
        [20, 25, 26, 27],      # left wrist to left middle3
        [20, 31, 32, 33],      # left wrist to left ring3
        [20, 28, 29, 30],      # left wrist to left pinky3
    ]

    RIGHT_HAND_CHAIN = [
        [21, 43, 44, 45],      # right wrist to right index3
        [21, 46, 47, 48],      # right wrist to right thumb3
        [21, 40, 41, 42],      # right wrist to right middle3
        [21, 37, 38, 39],      # right wrist to right ring3
        [21, 49, 50, 21],      # right wrist to right pinky3
    ]

    HANDS_CHAIN = LEFT_HAND_CHAIN + RIGHT_HAND_CHAIN
    BODY_HANDS_CHAIN = BODY_CHAIN + HANDS_CHAIN

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('requires_grad', False)      # disable this by default to save memory
        kwargs.setdefault('create_expression', False)  # disable this by default to save memory
        kwargs.setdefault('use_pca', False)
        super().__init__(*args, **kwargs)


    def forward_kinematics(self, 
                           pose: torch.Tensor, 
                           transl: Optional[torch.Tensor] = None,
                           betas: Optional[torch.Tensor] = None,
                           *args, 
                           **kwargs
                           ) -> torch.Tensor:
        # The input pose consists of only 22 SMPL body joints + 30 MANO hand joints
        kwargs['return_verts'] = False
        shape = pose.shape
        flat = shape[-1] != 3   # whether the `pose` tensor is a flattened representation of joint poses or not
        if not flat:
            pose = pose.flatten(start_dim=-2)
        pose = pose.flatten(end_dim=-2)    # reshape `pose` into (B', D')
        if not transl is None:
            transl = transl.flatten(end_dim=-2)
        if not betas is None:
            # betas has shape (B, D), assuming consistent across temporal horizon
            betas = betas.unsqueeze(dim=1).repeat(1, shape[1], 1).flatten(end_dim=-2)
        global_orient = pose[..., :3]
        body_pose = pose[..., 3 : 3 * (1 + len(SMPLX.BODY_IDXS))]
        left_hand_pose = pose[..., 3 * (1 + len(SMPLX.BODY_IDXS)) : 3 * (1 + len(SMPLX.BODY_IDXS + SMPLX.LEFT_HAND_IDXS))]
        right_hand_pose = pose[..., 3 * (1 + len(SMPLX.BODY_IDXS + SMPLX.LEFT_HAND_IDXS)):]

        jnts = self.forward(global_orient=global_orient,
                            body_pose=body_pose,
                            left_hand_pose=left_hand_pose,
                            right_hand_pose=right_hand_pose,
                            transl=transl,
                            betas=betas,
                            *args,
                            **kwargs).joints
                
        return jnts.reshape(shape[:-1] + jnts.shape[-2:]
                            if flat
                            else shape[:-2] + jnts.shape[-2:])


    @staticmethod
    def retrieve_body_parts(x: torch.Tensor, idxs: List[int], flat: bool = True) -> torch.Tensor:
        if not flat:
            return x[..., idxs, :]
        else:
            idxs = [i * 3 + j for i in idxs for j in range(3)]
            return x[..., idxs]
        
        
    @staticmethod
    def fit_ground(joints: torch.Tensor) -> torch.Tensor:
        y_min = joints[..., 1].min()
        return y_min
    

    root_joint = partial(retrieve_body_parts.__func__, idxs=ROOT_IDX)
    body_joints = partial(retrieve_body_parts.__func__, idxs=BODY_IDXS)
    hand_joints = partial(retrieve_body_parts.__func__, idxs=HANDS_IDXS)
    left_hand_joints = partial(retrieve_body_parts.__func__, idxs=LEFT_HAND_IDXS)
    right_hand_joints = partial(retrieve_body_parts.__func__, idxs=RIGHT_HAND_IDXS)
    full_body_joints = partial(retrieve_body_parts.__func__, idxs=ROOT_IDX + BODY_IDXS + HANDS_IDXS)
