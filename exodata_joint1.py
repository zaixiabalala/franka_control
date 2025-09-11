import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import glob

class Dataset60(Dataset):
    def __init__(self, data_root, chunk_size, joint_dim=8, image_size=(224, 224), provide_eih=True):
        """
        外骨骼数据集 - 支持多视角渲染
        
        Args:
            data_root: 数据根目录，包含 records_easy 和 scene_* 文件夹
            chunk_size: 动作序列长度
            joint_dim: 关节角度维度
            image_size: 图像尺寸
        """
        self.data_root = data_root
        self.chunk_size = chunk_size
        self.joint_dim = joint_dim
        self.image_size = image_size
        self.provide_eih = provide_eih
        
        # 构建样本索引：支持多视角扩展
        self.samples = self._build_sample_indices()
        
        self.stats = self._compute_real_stats()
    
    def _compute_real_stats(self):
        """计算数据集的真实统计信息"""
        print("[ExoDataset] 计算真实统计信息...")
        
        all_states = []
        all_actions = []
        
        # 采样一部分数据来计算统计信息
        sample_indices = np.linspace(0, len(self.samples)-1, min(1000, len(self.samples)), dtype=int)
        
        for idx in sample_indices:
            sample = self.samples[idx]
            
            # 加载当前状态
            current_joints = self._load_joint_data(sample["current_joint_file"])
            all_states.append(current_joints.numpy())
            
            # 加载动作序列
            for joint_file in sample["future_joint_files"]:
                joints = self._load_joint_data(joint_file)
                all_actions.append(joints.numpy())
        
        # 计算统计信息
        all_states = np.array(all_states)  # (N, joint_dim)
        all_actions = np.array(all_actions)  # (M, joint_dim)
        
        state_min = torch.tensor(np.min(all_states, axis=0))
        state_max = torch.tensor(np.max(all_states, axis=0))
        action_min = torch.tensor(np.min(all_actions, axis=0))
        action_max = torch.tensor(np.max(all_actions, axis=0))
        
        print(f"[ExoDataset] 状态范围: min={state_min}, max={state_max}")
        print(f"[ExoDataset] 动作范围: min={action_min}, max={action_max}")
        
        return {
            "observation.image.color": {"min": torch.zeros(3,1,1), "max": torch.ones(3,1,1)*255},
            "observation.image.eih": {"min": torch.zeros(3,1,1), "max": torch.ones(3,1,1)*255},
            "observation.state": {"min": state_min, "max": state_max},
            "action": {"min": action_min, "max": action_max}
        }
        
    def _build_sample_indices(self):
        """构建样本索引列表（支持多视角扩展）"""
        samples = []
        
        print(f"[ExoDataset] data_root = {self.data_root}")
        
        # 1) 找到所有 record 文件夹
        records_easy_path = os.path.join(self.data_root, "records_easy")
        if not os.path.exists(records_easy_path):
            raise ValueError(f"records_easy 目录不存在: {records_easy_path}")
        
        record_folders = [f for f in os.listdir(records_easy_path) 
                         if f.startswith("record_") and os.path.isdir(os.path.join(records_easy_path, f))]
        record_folders.sort()
        print(f"[ExoDataset] 找到 {len(record_folders)} 个 record 文件夹")
        
        # 2) 找到所有 scene 文件夹
        scene_folders = [f for f in os.listdir(self.data_root) 
                        if f.startswith("scene_") and os.path.isdir(os.path.join(self.data_root, f))]
        scene_folders.sort()
        print(f"[ExoDataset] 找到 {len(scene_folders)} 个 scene 文件夹: {scene_folders}")
        
        # 3) 为每个 record 构建多视角样本
        total_samples = 0
        for record_folder in record_folders:
            record_path = os.path.join(records_easy_path, record_folder)
            
            # 检查关节数据
            angles_path = os.path.join(record_path, "angles")
            if not os.path.exists(angles_path):
                print(f"[ExoDataset][WARN] 关节数据目录不存在: {angles_path}")
                continue
            
            # 获取关节数据文件
            joint_files = sorted(glob.glob(os.path.join(angles_path, "angle_cam0_*.npy")))
            if len(joint_files) == 0:
                print(f"[ExoDataset][WARN] 未找到关节数据文件: {angles_path}")
                continue
            
            num_frames = len(joint_files)
            print(f"[ExoDataset] {record_folder}: {num_frames} 帧关节数据")
            
            # 检查是否有足够的帧数
            if num_frames < self.chunk_size + 1:
                print(f"[ExoDataset][WARN] {record_folder} 帧数不足: {num_frames} < {self.chunk_size + 1}")
                continue
            
            # 为每个 scene 创建样本
            for scene_folder in scene_folders:
                # 构建 scene 下的 record 文件夹路径，添加 _30000 后缀
                scene_record_folder = f"{record_folder}_30000"
                scene_path = os.path.join(self.data_root, scene_folder, "test", scene_record_folder, "renders")
                
                if not os.path.exists(scene_path):
                    print(f"[ExoDataset][WARN] 渲染图片目录不存在: {scene_path}")
                    continue
                
                # 获取渲染图片文件
                render_files = sorted(glob.glob(os.path.join(scene_path, "*.png")))
                if len(render_files) == 0:
                    print(f"[ExoDataset][WARN] 未找到渲染图片: {scene_path}")
                    continue
                
                # 检查图片数量是否匹配关节数据
                if len(render_files) != num_frames:
                    print(f"[ExoDataset][WARN] {record_folder} + {scene_folder}: 图片数量({len(render_files)}) != 关节数据({num_frames})")
                    continue
                
                print(f"[ExoDataset] {record_folder} + {scene_folder}: {len(render_files)} 张图片")
                
                # 从第61份数据开始构建样本（索引60，因为从0开始计数）
                start_index = 60  # 第61份数据的索引
                for i in range(start_index, num_frames - self.chunk_size):
                    sample = {
                        "record_folder": record_folder,
                        "scene_folder": scene_folder,
                        "current_time": i,
                        "future_times": list(range(i + 1, i + 1 + self.chunk_size)),
                        "current_joint_file": joint_files[i],
                        "future_joint_files": joint_files[i + 1:i + 1 + self.chunk_size],
                        "current_image_file": render_files[i],
                        "future_image_files": render_files[i + 1:i + 1 + self.chunk_size],
                    }
                    samples.append(sample)
                    total_samples += 1
        
        print(f"[ExoDataset] 总共创建了 {total_samples} 个训练样本")
        print(f"[ExoDataset] 原始数据: {len(record_folders)} 条")
        print(f"[ExoDataset] 视角数量: {len(scene_folders)} 个")
        if len(record_folders) > 0:
            print(f"[ExoDataset] 平均每条数据扩展为: {total_samples / len(record_folders):.1f} 个样本")
        
        return samples
    
    def _load_and_preprocess_image(self, img_path):
        """加载并预处理图像"""
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()  # (3, H, W)
            img = img/255.0
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回黑色图像作为 fallback
            return torch.zeros(3, *self.image_size, dtype=torch.float32)
    
    def _load_joint_data(self, joint_file):
        """加载关节角度数据"""
        try:
            joints = np.load(joint_file)
            return torch.tensor(joints, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading joint data {joint_file}: {e}")
            return torch.zeros(self.joint_dim, dtype=torch.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 加载当前视角的图像
        current_image = self._load_and_preprocess_image(sample["current_image_file"])
        # 可选：为满足默认 ACT 配置的第二路图像键，返回零张量占位（形状与 color 一致）
        if self.provide_eih:
            eih_image = torch.zeros_like(current_image)
        
        # 2. 当前关节角度
        current_joints = self._load_joint_data(sample["current_joint_file"])
        
        # 3. 未来关节角度序列
        future_joint_files = sample["future_joint_files"]
        actions = torch.zeros(self.chunk_size, self.joint_dim, dtype=torch.float32)
        mask = torch.ones(self.chunk_size, dtype=torch.bool)
        
        n_steps = min(len(future_joint_files), self.chunk_size)
        for i in range(n_steps):
            joints = self._load_joint_data(future_joint_files[i])
            actions[i] = joints
            mask[i] = False
        
        batch = {
            "observation.image.color": current_image,  # 单视角图像
            "observation.state": current_joints,      # 当前关节角度
            "action": actions,                        # 未来关节角度序列
            "action_is_pad": mask,                   # padding 掩码
            # 添加元信息用于调试
            "record_folder": sample["record_folder"],
            "scene_folder": sample["scene_folder"],
            "current_time": sample["current_time"],
        }
        if self.provide_eih:
            batch["observation.image.eih"] = eih_image
        
        return batch
    
    def get_sample_info(self, idx):
        """获取样本的详细信息（用于调试）"""
        if idx >= len(self.samples):
            return None
        
        sample = self.samples[idx]
        return {
            "record_folder": sample["record_folder"],
            "scene_folder": sample["scene_folder"],
            "current_time": sample["current_time"],
            "current_joint_file": sample["current_joint_file"],
            "current_image_file": sample["current_image_file"],
            "total_frames": len(sample["future_joint_files"]) + 1,
        } 