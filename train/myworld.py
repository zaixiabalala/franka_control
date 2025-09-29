import os
import json
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from dataset.constants import *
from dataset.projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform


class MyWorldDataset(Dataset):
    """
    myworld Dataset.
    """
    def __init__(
        self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_action = 20, 
        voxel_size = 0.005,
        cam_ids = ['1234567890'],
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
        vis = False
    ):
        assert split in ['train', 'val', 'all']

        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        self.vis = vis
        
        self.all_demos = sorted(os.listdir(self.data_path))
        self.num_demos = len(self.all_demos)

        self.data_paths = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        
        for i in range(self.num_demos):
            demo_path = os.path.join(self.data_path, self.all_demos[i])
            if not os.path.isdir(demo_path):
                continue
                
            cloud_dir = os.path.join(demo_path, "pointclouds")
            angles_dir = os.path.join(demo_path, "angles")
            
            if not (os.path.exists(cloud_dir) and os.path.exists(angles_dir)):
                continue
                
            # 获取该demo中的所有文件
            demo_cloud_files = sorted(glob.glob(os.path.join(cloud_dir, "*.pcd")))
            demo_angles_files = sorted(glob.glob(os.path.join(angles_dir, "*.npy")))
            
            if len(demo_cloud_files) != len(demo_angles_files):
                print(f"Warning: Demo {self.all_demos[i]} has mismatched cloud ({len(demo_cloud_files)}) and angles ({len(demo_angles_files)}) files")
                continue
                
            # 提取帧ID（从文件名中提取数字）
            frame_ids = []
            for cloud_file in demo_cloud_files:
                frame_name = os.path.splitext(os.path.basename(cloud_file))[0]
                try:
                    frame_id = int(frame_name)
                    frame_ids.append(frame_id)
                except ValueError:
                    # 如果文件名不是纯数字，使用索引
                    frame_ids.append(len(frame_ids))
            
            frame_ids = sorted(frame_ids)
            
            # 生成训练样本
            obs_frame_ids_list = []
            action_frame_ids_list = []
            
            for cur_idx in range(len(frame_ids) - 1):
                obs_pad_before = max(0, num_obs - cur_idx - 1)
                action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                frame_begin = max(0, cur_idx - num_obs + 1)
                frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                obs_frame_ids_list.append(obs_frame_ids)
                action_frame_ids_list.append(action_frame_ids)
            
            self.data_paths += [demo_path] * len(obs_frame_ids_list)
            self.obs_frame_ids += obs_frame_ids_list
            self.action_frame_ids += action_frame_ids_list
        
        print(f"Found {len(self.obs_frame_ids)} training samples across {self.num_demos} demos")
        
    def __len__(self):
        return len(self.obs_frame_ids)

    def _augmentation(self, clouds, states):
        """对点云和状态应用相同的数据增强"""
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # 从度转换为弧度
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        center = clouds[-1][..., :3].mean(axis=0)

        for i in range(len(clouds)):
            clouds[i][..., :3] -= center
            clouds[i] = apply_mat_to_pcd(clouds[i], aug_mat)
            clouds[i][..., :3] += center
            
        return clouds, states

    def _normalize_state(self, state_list):
        """归一化状态数据"""
        # state_list: [T, 8] - 7个关节角度 + 1个夹爪宽度
        state_list = state_list.copy()
        # 归一化关节角度 (前7个)
        state_list[:, :7] = state_list[:, :7] / np.pi
        # 归一化夹爪宽度 (最后一个)
        state_list[:, 7] = (state_list[:, 7] - 0.0) / 0.08 * 2 - 1
        return state_list

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]
        
        # 目录路径
        cloud_dir = os.path.join(data_path, "pointclouds")
        angles_dir = os.path.join(data_path, "angles")
        
        # 加载观察帧的点云
        clouds = []
        for frame_id in obs_frame_ids:
            cloud_file = os.path.join(cloud_dir, f"{frame_id}.pcd")
            if not os.path.exists(cloud_file):
                # 如果文件不存在，尝试其他可能的命名方式
                cloud_files = glob.glob(os.path.join(cloud_dir, "*.pcd"))
                if cloud_files:
                    cloud_file = sorted(cloud_files)[frame_id % len(cloud_files)]
                else:
                    raise FileNotFoundError(f"Cloud file not found for frame {frame_id}")
            
            cloud = o3d.io.read_point_cloud(cloud_file)
            points = np.array(cloud.points)
            colors = np.array(cloud.colors)
            
            # 应用工作空间裁剪
            x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
            y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
            z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
            mask = (x_mask & y_mask & z_mask)
            points = points[mask]
            colors = colors[mask]
            
            # 应用ImageNet归一化
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud = np.concatenate([points, colors], axis=-1)
            clouds.append(cloud)
        
        # 加载动作帧的状态
        action_states = []
        for frame_id in action_frame_ids:
            state_file = os.path.join(angles_dir, f"{frame_id}.npy")
            if not os.path.exists(state_file):
                state_files = glob.glob(os.path.join(angles_dir, "*.npy"))
                if state_files:
                    state_file = sorted(state_files)[frame_id % len(state_files)]
                else:
                    raise FileNotFoundError(f"State file not found for frame {frame_id}")
            
            state = np.load(state_file)
            assert state.shape == (8,), f"Expected state shape (8,), got {state.shape}"
            action_states.append(state)
        
        action_states = np.stack(action_states)
        
        # 数据增强
        if self.split == 'train' and self.aug:
            clouds, action_states = self._augmentation(clouds, action_states)
        
        # 可视化
        if self.vis:
            print(f"Demo: {os.path.basename(data_path)}")
            print(f"Obs frames: {obs_frame_ids}, Action frames: {action_frame_ids}")
            print(f"Action states shape: {action_states.shape}")
            print(f"Point cloud range: {clouds[-1][..., :3].min(axis=0)} to {clouds[-1][..., :3].max(axis=0)}")
            print(f"Point cloud size: {len(clouds[-1])} points")
            
            # 显示点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(clouds[-1][..., :3])
            pcd.colors = o3d.utility.Vector3dVector(clouds[-1][..., 3:6] * IMG_STD + IMG_MEAN)
            
            # 显示工作空间边界
            bbox3d = o3d.geometry.AxisAlignedBoundingBox(WORKSPACE_MIN, WORKSPACE_MAX)
            bbox3d.color = [1, 0, 0]
            
            o3d.visualization.draw_geometries([pcd, bbox3d])
        
        # 准备体素输入
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype=np.int32)
            feats = cloud.astype(np.float32)
            input_coords_list.append(coords)
            input_feats_list.append(feats)
        
        # 状态归一化
        actions_normalized = self._normalize_state(action_states.copy())
        
        # 转换为torch tensor
        actions = torch.from_numpy(action_states).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()
        
        ret_dict = {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'action': actions,
            'action_normalized': actions_normalized
        }
        
        if self.with_cloud:
            ret_dict["clouds_list"] = clouds
        
        return ret_dict
        

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict['input_coords_list']
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


