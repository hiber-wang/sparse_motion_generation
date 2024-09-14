import torch
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from math import ceil, sqrt

from model import SMPLX
from utils.rotation_conversion import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d


class OnlineMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 0

    def __call__(self, x):
        self.count += 1
        old_mean = self.mean
        mean += (x - old_mean) / self.count
        
    def get_std_and_var(self):
        return self.mean, sqrt(self.m2 / self.count)



def transform_amass(poses, trans):
    # AMASS processing: rotate along x-axis for -90 degrees
    correct = torch.tensor([[
        [1., 0., 0.],
        [0., 0., 1.],
        [0., -1, 0.]
    ]], dtype=poses.dtype, device=poses.device)
    poses = poses.reshape((poses.shape[0], -1, 3))
    glob_rot = axis_angle_to_matrix(poses[:, 0])
    glob_rot = matrix_to_axis_angle(correct @ glob_rot)
    poses[:, 0] = glob_rot
    trans = (correct @ trans.unsqueeze(dim=-1)).view_as(trans)
    return poses.flatten(start_dim=1), trans
    


def transform_beat2(poses, trans):
    # BEAT2 processing: discard poses for jaw and L&R eyes
    poses = poses.reshape((poses.shape[0], -1, 3))
    poses = poses[:, list(range(0, 22)) + list(range(25, 55))].flatten(start_dim=1)
    return poses, trans


transform_fns = {
    'amass': transform_amass,
    'beat2': transform_beat2,
}


def process_sequence(smpl, data, dataset, chunk_size=32, fps=30, device='cpu', dtype=torch.float32):
    smpl = smpl.to(dtype=dtype)
    axis_angle_to_rotation_6d = lambda aa: matrix_to_rotation_6d(axis_angle_to_matrix(aa))

    try:
        poses = torch.from_numpy(data['poses']).to(device=device, dtype=dtype)
        transl = torch.from_numpy(data['trans']).to(device=device, dtype=dtype)
    except KeyError:
        return

    # Downsample motion sequence
    framerate = data.get('mocap_framerate', None)
    framerate = data.get('mocap_frame_rate', framerate)
    if framerate is None:
        return
    
    seqlen = poses.shape[0]
    num_samples = int(float(fps) / framerate * seqlen)
    idxs = torch.linspace(0, seqlen-1, num_samples, dtype=torch.int32)
    if len(idxs) / fps < 2.0:    # discard short sequences
        return
    poses, transl = poses[idxs], transl[idxs]

    poses, transl = transform_fns[dataset](poses, transl)
    
    num_chunks = ceil(poses.shape[0] / chunk_size)
    chunks = zip(poses.split(num_chunks), transl.split(num_chunks))

    rot_list, joint_list = [], []
    for chunk in chunks:
        p, t = chunk
        jnts = smpl.forward_kinematics(p, transl=t)
        joint_list.append(jnts.detach().cpu())
        rot6d = axis_angle_to_rotation_6d(p.reshape(p.shape[0], -1, 3))
        rot_list.append(rot6d.detach().cpu())


    rot = torch.cat(rot_list, dim=0)
    pos = torch.cat(joint_list, dim=0)
    # Put feet on ground
    ground = smpl.fit_ground(pos)
    pos[..., 1] -= ground
    transl -= ground
    
    return {'rotation': rot, 'position': pos, 'trans': transl.detach().cpu(), 'mocap_framerate': fps, 'gender': 'neutral'}



if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl_dir', type=str)
    parser.add_argument('--source_dir', type=str)
    parser.add_argument('--target_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    smpl = SMPLX(args.smpl_dir).to(device=args.device)
    files = glob(os.path.join(args.source_dir, '*.npz'))
    os.makedirs(args.target_dir, exist_ok=True)

    for f in tqdm(files):
        data = np.load(f)
        processed = process_sequence(smpl, data, args.dataset, args.chunk, args.fps, args.device)
        if processed is None:
            print(f'Sequence {f} is discarded during processing')
            continue
        torch.save(processed, os.path.join(args.target_dir, f.split('/')[-1].replace('.npz', '.pt')))