# This code is borrowed from MMHuman3D.
# We modified the keypoints definition to only convert body and hand keypoints.
from typing import Dict, Tuple, List

import torch

# A cache holding precomputed keypoint mappings
__CACHE__: Dict[Tuple[str, str], Tuple[List[int], List[int]]] = {}


# NOTE: this is not standard SMPL joints definition.
SMPL_KEYPOINTS = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine_1',
    'left_knee',
    'right_knee',
    'spine_2',
    'left_ankle',
    'right_ankle',
    'spine_3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eyeball',
    'right_eyeball',
    'left_index_1',
    'left_index_2',
    'left_index_3',
    'left_middle_1',
    'left_middle_2',
    'left_middle_3',
    'left_pinky_1',
    'left_pinky_2',
    'left_pinky_3',
    'left_ring_1',
    'left_ring_2',
    'left_ring_3',
    'left_thumb_1',
    'left_thumb_2',
    'left_thumb_3',
    'right_index_1',
    'right_index_2',
    'right_index_3',
    'right_middle_1',
    'right_middle_2',
    'right_middle_3',
    'right_pinky_1',
    'right_pinky_2',
    'right_pinky_3',
    'right_ring_1',
    'right_ring_2',
    'right_ring_3',
    'right_thumb_1',
    'right_thumb_2',
    'right_thumb_3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_bigtoe',
    'left_smalltoe',
    'left_heel',
    'right_bigtoe',
    'right_smalltoe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
]


# NOTE: this is not standard OpenPose-135 keypoints, 
# we only include 25 body keypoints and 40 hand keypoints.
OPENPOSE_KEYPOINTS = [
    # Body joints
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'neck',  # upper_neck
    'head',
    'left_bigtoe',
    'left_smalltoe',
    'left_heel',
    'right_bigtoe',
    'right_smalltoe',
    'right_heel',
    # Hand joints
    'left_thumb_1',
    'left_thumb_2',
    'left_thumb_3',
    'left_thumb',
    'left_index_1',
    'left_index_2',
    'left_index_3',
    'left_index',
    'left_middle_1',
    'left_middle_2',
    'left_middle_3',
    'left_middle',
    'left_ring_1',
    'left_ring_2',
    'left_ring_3',
    'left_ring',
    'left_pinky_1',
    'left_pinky_2',
    'left_pinky_3',
    'left_pinky',
    'right_thumb_1',
    'right_thumb_2',
    'right_thumb_3',
    'right_thumb',
    'right_index_1',
    'right_index_2',
    'right_index_3',
    'right_index',
    'right_middle_1',
    'right_middle_2',
    'right_middle_3',
    'right_middle',
    'right_ring_1',
    'right_ring_2',
    'right_ring_3',
    'right_ring',
    'right_pinky_1',
    'right_pinky_2',
    'right_pinky_3',
    'right_pinky',
]


_KEYPOINTS = {
    'openpose': OPENPOSE_KEYPOINTS,
    'smpl': SMPL_KEYPOINTS,
}



# We assume the mapping between src keypoints to dst keypoints is surjective.
def get_mapping(src: str, dst: str) -> Tuple[List[int], List[int]]:
    if (src, dst) in __CACHE__:
        return __CACHE__[(src, dst)]
    src_names = _KEYPOINTS[src]
    dst_names = _KEYPOINTS[dst]

    src_idxs, dst_idxs = [], []
    for dst_idx, dst_name in enumerate(dst_names):
        src_idx = src_names.index(dst_name)
        src_idxs.append(src_idx)
        dst_idxs.append(dst_idx)

    __CACHE__[(src, dst)] = (src_idxs, dst_idxs)
    
    return src_idxs, dst_idxs


def smpl_to_openpose(keypoints: torch.Tensor) -> torch.Tensor:
    src_idxs, dst_idxs = get_mapping('smpl', 'openpose')
    out_shape = keypoints.shape[:-2] + (len(_KEYPOINTS['openpose']),) + (keypoints.shape[-1],)
    out_keypoints = torch.zeros(out_shape, device=keypoints.device, dtype=keypoints.dtype)
    out_keypoints[..., dst_idxs, :] = keypoints[..., src_idxs, :]
    return out_keypoints