#!/usr/bin/env python3
"""
基于RISE模型的实时推理脚本 - Franka机器人部署

快速启动:
  python inference_with_policy_interface_rise.py

默认参数:
  - 测试模式: True (不连接真实机器人)
  - 推理频率: 5.0 Hz
  - 计算设备: cuda
  - 最大步数: 1000
  - 模型路径: 自动搜索 RISE/logs/my_task/policy_last.ckpt
  - 配置文件: 自动搜索 franka_control/config/robot_config.yaml

功能:
  - 支持RealSense D415深度相机
  - 基于点云的RISE策略推理
  - 实时机器人控制（可选）
  - 智能路径搜索和错误处理
"""

import os
from shlex import join
import numpy as np
import torch
import time
from pathlib import Path
import argparse
from PIL import Image
import cv2
import math
from safetensors.torch import load_file
import sys
import yaml
from common.gripper_util import convert_gripper_width_to_encoder


# 添加项目路径到sys.path，确保优先使用项目中的lerobot库
project_dir = Path(__file__).parent.parent
model_lerobot_path = project_dir / "model" / "lerobot" / "src"
sys.path.insert(0, str(model_lerobot_path))
sys.path.insert(0, str(project_dir))  # 添加项目根目录到路径

# 导入 RISE 策略（基于 my_train.py 和 eval.py）
import sys
rise_path = Path(__file__).parent.parent.parent / "RISE"
sys.path.insert(0, str(rise_path))

# 导入必要的库
try:
    import open3d as o3d
    import MinkowskiEngine as ME
    from policy import RISE
    from dataset.projector import Projector
    RISE_AVAILABLE = True
except ImportError as e:
    print(f"警告: RISE相关库导入失败: {e}")
    print("请确保已安装 open3d, MinkowskiEngine 等依赖")
    RISE_AVAILABLE = False

# 导入PolicyInterface
from policy_interface import create_policy_interface

# 导入precise_wait
from common.precise_sleep import precise_wait

# 相机相关导入
import pyrealsense2 as rs
from r3kit.devices.camera.realsense import config as rs_cfg
from r3kit.devices.camera.realsense.d415 import D415
R3KIT_RS_AVAILABLE = True

# D415 相机配置（与采集脚本保持一致）
FPS = 30
D415_CAMERAS = {   
    "cam4": "327322062498",  # 固定机位视角
    "eih": "038522062288",   # eye-in-hand视角（需要根据实际序列号修改）
}

class CameraSystem:
    """相机系统接口"""
    
    def __init__(self):
        self.cameras = {}
        self.camera_names = ["cam4", "eih"]  # 支持双视角
        self.use_realsense = True
        
        # 流配置
        rs_cfg.D415_STREAMS = [
            (rs.stream.depth, 640,480, rs.format.z16, FPS),
            (rs.stream.color, 640,480, rs.format.bgr8, FPS),
        ]
        for name in self.camera_names:
            serial = D415_CAMERAS.get(name)
            if serial is None:
                print(f"{name} 缺少序列号，跳过")
                continue
            try:
                cam = D415(id=serial, depth=True, name=name)
                self.cameras[name] = cam
                print(f"成功初始化相机 {name} (序列号: {serial})")
            except Exception as e:
                print(f"初始化相机 {name} 失败: {e}")
                continue
                
        if len(self.cameras) > 0:
            self.use_realsense = True
            print(f"使用 RealSense D415，相机数量: {len(self.cameras)}")
            print(f"可用相机: {list(self.cameras.keys())}")
        else:
            print("警告: 没有成功初始化任何相机")
    
    def get_image(self, cam_name):
        """获取指定相机的图像"""
        if cam_name not in self.cameras:
            return None
        
        try:
            if self.use_realsense:
                # r3kit D415 接口
                color, depth = self.cameras[cam_name].get()
                if color is None:
                    return None
                # 转 RGB
                frame_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                return frame_rgb
            else:
                # OpenCV 摄像头
                ret, frame = self.cameras[cam_name].read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return frame_rgb
                return None
        except Exception as e:
            print(f"获取 {cam_name} 图像失败: {e}")
            return None
    
    def get_depth(self, cam_name):
        """获取指定相机的深度"""
        if cam_name not in self.cameras:
            return None
        
        try:
            if self.use_realsense:
                # r3kit D415 接口
                color, depth = self.cameras[cam_name].get()
                return depth
        except Exception as e:
            print(f"获取 {cam_name} 深度失败: {e}")
            return None

    def get_all_images(self):
        """获取所有相机的图像"""
        images = {}
        for cam_name in self.camera_names:
            image = self.get_image(cam_name)
            if image is not None:
                images[cam_name] = image
            else:
                # 生成模拟图像作为 fallback
                images[cam_name] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                print(f"警告: {cam_name} 相机图像获取失败，使用模拟图像")
        
        return images
    
    def get_image_and_depth(self, cam_name):
        """获取指定相机的图像和深度"""
        if cam_name not in self.cameras:
            return None, None
        
        return self.get_image(cam_name), self.get_depth(cam_name)
    
    def close(self):
        """关闭所有相机"""
        for cam_name, cap in self.cameras.items():
            try:
                if self.use_realsense:
                    # D415 类可能没有 stop 方法，使用 __del__ 或者不做任何操作
                    if hasattr(cap, 'stop'):
                        cap.stop()
                    elif hasattr(cap, 'close'):
                        cap.close()
                    # 对于 r3kit D415，通常由析构函数自动处理
                else:
                    cap.release()
                print(f"{cam_name} 已关闭")
            except Exception as e:
                print(f"关闭 {cam_name} 失败: {e}")


class RISEPolicyWrapper:
    """RISE策略包装器 - 基于点云和MinkowskiEngine的策略"""
    
    def __init__(self, model_path, device="cpu", camera_system=None, debug_image=False):
        if not RISE_AVAILABLE:
            raise ImportError("RISE相关库未正确安装，请检查依赖")
            
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.camera_system = camera_system
        self.debug_image = debug_image
        
        # 配置参数 - 基于 RISE 模型
        self.camera_names = ["cam4", "eih"]  # 支持双视角
        self.joint_dim = 7  # 7个关节角度（弧度）  
        self.gripper_dim = 1  # 1个夹爪开合值  
        self.action_dim = self.joint_dim + self.gripper_dim  # 总共8维  
        
        # RISE 模型参数
        self.num_action = 20  # 动作序列长度
        self.voxel_size = 0.005  # 体素大小
        self.obs_feature_dim = 512  # 观测特征维度
        self.hidden_dim = 512  # 隐藏层维度
        self.nheads = 8  # 注意力头数
        self.num_encoder_layers = 4  # 编码器层数
        self.num_decoder_layers = 1  # 解码器层数
        self.dropout = 0.1  # dropout率
        self.action_queue = []
        
        # 加载模型
        self.policy = self._load_policy()
        
        print(f"RISE策略初始化完成: {model_path}")
        print(f"使用设备: {self.device}")
        print(f"支持双视角输入: 固定机位(cam4) + eye-in-hand(eih)")
        print(f"相机系统状态: {len(self.camera_system.cameras) if self.camera_system else 0} 个相机已初始化")
    
    def _load_policy(self):
        """加载训练好的RISE策略模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        policy = RISE(
            num_action=self.num_action,
            input_dim=6,  # 点云特征维度：3D坐标 + 3D颜色
            obs_feature_dim=self.obs_feature_dim,
            action_dim=8,  # 7个关节 + 1个夹爪
            hidden_dim=self.hidden_dim,
            nheads=self.nheads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout
        ).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        policy.load_state_dict(checkpoint, strict=False)
        
        print(f"RISE模型加载成功:")
        print(f" 模型类型: RISE (基于点云的策略)")
        print(f" 设备: {next(policy.parameters()).device}")
        print(f" 动作序列长度: {self.num_action}")
        print(f" 体素大小: {self.voxel_size}")
        print(f" 观测特征维度: {self.obs_feature_dim}")
        print(f" 隐藏层维度: {self.hidden_dim}")
        print(f" 注意力头数: {self.nheads}")
        print(f" 编码器层数: {self.num_encoder_layers}")
        print(f" 解码器层数: {self.num_decoder_layers}")
        
        return policy
    
    def create_point_cloud(self, color_image, depth_image, cam_intrinsics):
        """
        从RGB-D图像创建点云（基于 eval.py）
        """
        h, w = depth_image.shape
        fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
        cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

        colors = o3d.geometry.Image(color_image.astype(np.uint8))
        depths = o3d.geometry.Image(depth_image.astype(np.float32))

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, depth_scale=1.0, convert_rgb_to_intensity=False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points).astype(np.float32)
        colors = np.array(cloud.colors).astype(np.float32)

        # 工作空间裁剪
        x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
        y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
        z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
        mask = (x_mask & y_mask & z_mask)
        points = points[mask]
        colors = colors[mask]
        
        # ImageNet归一化
        colors = (colors - IMG_MEAN) / IMG_STD
        
        # 合并点和颜色
        cloud_final = np.concatenate([points, colors], axis=-1).astype(np.float32)
        return cloud_final
    
    def create_batch(self, coords, feats):
        coords_batch = [coords]
        feats_batch = [feats]
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        return coords_batch, feats_batch
    
    def create_input(self, color_image, depth_image, cam_intrinsics):
        """
        从RGB-D图像创建输入（基于 eval.py）
        """
        cloud = self.create_point_cloud(color_image, depth_image, cam_intrinsics)
        coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype=np.int32)
        coords_batch, feats_batch = self.create_batch(coords, cloud)
        return coords_batch, feats_batch, cloud
    
    def unnormalize_action(self, action):
        """
        反归一化动作（基于 myworld.py 的训练时归一化方式）
        
        Args:
            action: 归一化的动作 (..., 8) - 前7维关节角度，第8维夹爪宽度
        
        Returns:
            action: 反归一化后的动作
                - 前7维：关节角度（弧度），范围 [-π, π]
                - 第8维：夹爪宽度（米），范围 [0, 0.08]
        """
        action = action.copy()
        
        # 反归一化关节角度：从 [-1, 1] 恢复到 [-π, π]
        action[..., :7] = action[..., :7] * np.pi
        
        # 反归一化夹爪宽度：从 [-1, 1] 恢复到 [0, 0.08]
        # 训练时：gripper_norm = (gripper - 0.0) / 0.08 * 2 - 1
        # 反推：gripper = (gripper_norm + 1) * 0.08 / 2
        action[..., 7] = (action[..., 7] + 1) * 0.08 / 2
        
        return action
    
    def get_current_state_with_gripper(self, obs):
        """从观测中获取当前状态（8维）"""
        joints_rad = obs['robot0_joint_pos']  # (7,)
        
        if 'robot0_gripper_width' in obs:
            gripper_width = obs['robot0_gripper_width']
            if isinstance(gripper_width, np.ndarray):
                gripper_width = gripper_width[0] if len(gripper_width) > 0 else 0.04
        else:
            gripper_width = 0.04  # 默认夹爪宽度（米）
        
        # 返回8维状态：7个关节角度（弧度） + 1个夹爪宽度（米）
        return np.concatenate([joints_rad, [gripper_width]])
    
    def preprocess_image(self, image, depth):
        """
        根据RGB图像对齐裁剪深度图
        
        Args:
            image: RGB图像 (480, 640, 3) uint8
            depth: 深度图 (480, 640) uint16 毫米单位
        
        Returns:
            cropped_rgb: 裁剪后的RGB图像
            cropped_depth: 裁剪后的深度图
        """
        # 确保输入是numpy数组
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(depth, Image.Image):
            depth = np.array(depth)
        
        # 获取原始图像尺寸
        h, w = image.shape[:2]
        start_w = 200
        end_w = 560
        start_h = 0
        end_h = 360

        # 裁剪深度图和RGB图
        cropped_depth = depth[start_h:end_h, start_w:end_w]
        cropped_rgb = image[start_h:end_h, start_w:end_w]
        
        return cropped_rgb, cropped_depth

    
    def predict_single_action(self, images, current_state, cam_intrinsics):
        """
        单步预测动作（使用 RISE 策略）。
        返回: (8,) numpy 数组，前7维为关节(弧度)，第8维为夹爪(米)。
        """
        # 获取RGB和深度图像
        if "cam4" in images:
            color_img, depth_img = self.camera_system.get_image_and_depth("cam4")
        else:
            # 生成模拟数据
            color_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth_img = np.ones((480, 640), dtype=np.float32) * 0.5
            print("警告: 固定机位视角图像获取失败，使用模拟数据")
        
        if color_img is None or depth_img is None:
            color_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth_img = np.ones((480, 640), dtype=np.float32) * 0.5
            print("警告: 图像获取失败，使用模拟数据")
        
        color_img, depth_img = self.preprocess_image(color_img, depth_img)
        
        # 创建点云输入
        coords_batch, feats_batch, cloud = self.create_input(color_img, depth_img, cam_intrinsics)
        feats_batch, coords_batch = feats_batch.to(self.device), coords_batch.to(self.device)
        cloud_data = ME.SparseTensor(feats_batch, coords_batch)

        if len(self.action_queue) == 0:
            with torch.no_grad():
                # 使用RISE策略进行预测
                pred_raw_actions = self.policy(cloud_data, actions=None, batch_size=1).squeeze(0).cpu().numpy()
                
                # 反归一化动作
                actions = self.unnormalize_action(pred_raw_actions)
                
                for action in actions:
                    self.action_queue.append(action)

        action = self.action_queue.pop(0)
            
        return action
    
    
    def __call__(self, obs):
        """
        策略函数 - PolicyInterface兼容接口
        
        Args:
            obs: 观测字典，包含robot0_joint_pos等
            
        Returns:
            action: 8维动作 [j1, j2, j3, j4, j5, j6, j7, gripper] (弧度, 米)
        """
        # 获取当前图像
        current_images = self.camera_system.get_all_images()
        
        # 获取当前状态
        current_state = self.get_current_state_with_gripper(obs)
        
        # 相机内参
        cam_intrinsics = np.array([[606.268127441406, 0, 319.728454589844, 0],
                              [0, 605.743286132812, 234.524749755859, 0],
                              [0, 0, 1, 0]])
        
        # 单步预测动作
        full_action = self.predict_single_action(current_images, current_state, cam_intrinsics)
        
        joint_action = full_action[:self.joint_dim]
        gripper_width = full_action[self.joint_dim]  # 夹爪宽度（米）

        cur_action = np.concatenate([joint_action, [gripper_width]])
        return cur_action
    
    def check_camera_status(self):
        """检查相机状态"""
        if not self.camera_system:
            print("相机系统未初始化")
            return False
        
        print("相机状态检查:")
        for cam_name in self.camera_names:
            if cam_name in self.camera_system.cameras:
                print(f"  ✅ {cam_name}: 已初始化")
            else:
                print(f"  ❌ {cam_name}: 未初始化")
        
        return len(self.camera_system.cameras) > 0


class RISEInferenceRunner:
    """RISE推理运行器 """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 device: str = "cuda",
                 max_steps: int = 1000,
                 test_mode: bool = False,
                 frequency: float = 20.0,
                 debug_image: bool = False):
        """
        初始化ACT推理运行器
        
        Args:
            model_path: 模型路径
            config_path: 配置文件路径
            device: 计算设备
            max_steps: 最大运行步数
            test_mode: 测试模式
            frequency: 推理频率 (Hz)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.max_steps = max_steps
        self.test_mode = test_mode
        self.frequency = frequency
        self.debug_image = debug_image
        self.dt = 1.0 / frequency  # 时间间隔
        
        # 创建相机系统
        self.camera_system = CameraSystem()
        
        # 创建RISE策略
        self.policy = RISEPolicyWrapper(
            model_path=model_path,
            device=device,
            camera_system=self.camera_system,
            debug_image=self.debug_image
        )
        
        print(f"RISE推理运行器初始化完成")
        print(f"模型路径: {model_path}")
        print(f"配置文件: {config_path}")
        print(f"设备: {device}")
        print(f"测试模式: {test_mode}")
        print(f"推理频率: {frequency} Hz")
        
        # 检查相机状态
        self.policy.check_camera_status()
    
    def run(self):
        """执行推理"""
        if self.test_mode:
            print("使用测试模式")
            self._run_test_mode()
        else:
            print("使用实时推理模式")
            self._run_real_time_mode()
    
    def _run_test_mode(self):
        """测试模式：运行几次推理"""
        print("开始测试推理...")
        for i in range(3):
            print(f"\n=== 测试推理 {i + 1} ===")
            # 模拟观测数据
            obs = {
                'robot0_joint_pos': np.random.uniform(-1, 1, 7),
                'robot0_joint_vel': np.random.uniform(-0.1, 0.1, 7),
                'robot0_eef_pos': np.random.uniform(0.3, 0.7, 3),
                'robot0_eef_rot_axis_angle': np.random.uniform(-1, 1, 3),
                'robot0_gripper_width': np.random.uniform(0.0, 0.08, 1),  # 添加gripper宽度
                'timestamp': time.monotonic()
            }
            
            # 执行策略
            cur_action = self.policy(obs)
            joint_action = cur_action[:self.policy.joint_dim]
            gripper_action = cur_action[self.policy.joint_dim]
            
            print(f"预测的关节动作（7维）: {joint_action}")
            print(f"预测的夹爪动作（1维）: {gripper_action}")
            print(f"预测的完整动作（8维）: {cur_action}")
            print(f"预测的夹爪动作: {gripper_action}")
            
            time.sleep(2)
    
    def _run_real_time_mode(self):
        """实时推理模式"""
        try:
            # 创建策略接口
            interface = create_policy_interface(self.config_path, self.policy)
            
            print("启动策略接口...")
            interface.start()
            print("策略接口已启动!")
            
            # 获取初始观测
            obs = interface.get_observation()
            print(f"初始关节位置: {obs['robot0_joint_pos']}")
            print(f"初始Gripper宽度: {obs['robot0_gripper_width']}")
            
            # 运行策略
            print(f"\n开始运行策略...")
            print(f"推理频率: {self.frequency} Hz (dt = {self.dt:.3f}s)")
            print("按 Ctrl+C 停止")
            
            # 初始化时间控制
            t_start = time.monotonic()
            step = 0
            
            # 超时降级策略相关变量
            last_joint_action = None
            last_gripper_action = None
            inference_times = []
            max_inference_time = 0.18  # 最大允许推理时间 (180ms) - 针对130ms推理时间优化
            timeout_count = 0
            
            while True:
                if self.max_steps is not None and step >= self.max_steps:
                    print(f"达到最大步数 {self.max_steps}，停止运行")
                    break
                
                # 计算当前周期结束时间
                t_cycle_end = t_start + (step + 1) * self.dt
                t_cycle_start = time.monotonic()
                
                # 获取观测
                obs = interface.get_observation()
                
                # 执行策略 - 添加超时检查
                t_inference_start = time.monotonic()
                try:
                    cur_action = self.policy(obs)
                    joint_action = cur_action[:self.policy.joint_dim]
                    gripper_action = cur_action[self.policy.joint_dim]
                    t_inference_end = time.monotonic()
                    inference_time = t_inference_end - t_inference_start
                    inference_times.append(inference_time)
                    
                    # 更新最后有效的动作
                    last_joint_action = joint_action.copy()
                    last_gripper_action = gripper_action.copy()
                    timeout_count = 0
                    
                except Exception as e:
                    print(f"推理失败: {e}")
                    t_inference_end = time.monotonic()
                    inference_time = t_inference_end - t_inference_start
                    inference_times.append(inference_time)
                    timeout_count += 1
                
                # 检查是否超时
                current_time = time.monotonic()
                elapsed_time = current_time - t_cycle_start
                remaining_time = t_cycle_end - current_time
                
                # 如果推理时间过长或剩余时间不足，使用降级策略
                if (inference_time > max_inference_time or 
                    remaining_time < 0.01): # 剩余时间少于10ms):
                    
                    if last_joint_action is not None and last_gripper_action is not None:
                        # 使用上次的有效动作
                        joint_action = last_joint_action
                        gripper_action = last_gripper_action
                        print(f"⚠️  使用降级策略: 推理时间={inference_time:.3f}s, 剩余时间={remaining_time:.3f}s")
                    else:
                        joint_action = obs['robot0_joint_pos'] + np.random.normal(0, 0.001, 7)
                        print(f"⚠️  使用随机扰动动作，等待有效推理: 推理时间={inference_time:.3f}s")
                        continue 
                
                interface.execute_action(joint_action)
                interface.execute_gripper_action(gripper_action)
                
                if step % 10 == 0:
                    current_time = time.monotonic() - t_start
                    avg_inference_time = np.mean(inference_times[-10:]) if len(inference_times) >= 10 else np.mean(inference_times)
                    print(f"Step {step}: 时间={current_time:.2f}s, 推理时间={inference_time:.3f}s (平均={avg_inference_time:.3f}s)")
                    print(f"  关节动作: {joint_action}")
                    print(f"  Gripper动作: {gripper_action}")
                    if timeout_count > 0:
                        print(f"  超时次数: {timeout_count}")
                
                step += 1
                
                # 等待到下一个周期
                precise_wait(t_cycle_end)
                
        except KeyboardInterrupt:
            print("\n用户中断，停止策略...")
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 停止策略接口
            if 'interface' in locals():
                print("停止策略接口...")
                interface.stop()
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'camera_system'):
            self.camera_system.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="基于RISE模型的实时推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 设置默认路径（相对于当前脚本位置）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    parser.add_argument("--model_path", type=str, 
                       default="policy_last.ckpt",
                       help="训练好的RISE模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备 (cpu/cuda)")
    parser.add_argument("--config_path", type=str,
                       default=str(script_dir.parent / "config" / "robot_config.yaml"),
                       help="机器人配置文件路径")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="最大运行步数")
    parser.add_argument("--test_mode", action="store_true", default=True,
                       help="测试模式（不连接真实机器人）")
    parser.add_argument("--frequency", type=float, default=5.0,
                       help="推理频率 (Hz) - RISE模型推理较慢，建议5Hz")
    parser.add_argument("--debug_image", action="store_true", default=False,
                       help="显示图像处理调试信息")
    
    args = parser.parse_args()
    
    # 检查并设置模型路径
    if not os.path.exists(args.model_path):
        print(f"⚠️  默认模型路径不存在: {args.model_path}")
        return 1
    
    # 检查并设置配置文件路径
    if not os.path.exists(args.config_path):
        print(f"⚠️  默认配置文件不存在: {args.config_path}")
        return 1
    
    print(f"📁 使用模型路径: {args.model_path}")
    print(f"📁 使用配置文件: {args.config_path}")
    print(f"🎯 测试模式: {args.test_mode}")
    print(f"⚡ 推理频率: {args.frequency} Hz")
    print(f"🖥️  计算设备: {args.device}")
    print("-" * 50)
    
    print("🚀 启动RISE推理系统...")
    if args.test_mode:
        print("📝 运行在测试模式 - 不会连接真实机器人")
    else:
        print("⚠️  运行在实时模式 - 将连接真实机器人!")
        print("   请确保机器人已正确连接并处于安全状态")
    print("-" * 50)
    
    # 创建并运行RISE推理运行器
    try:
        runner = RISEInferenceRunner(
            model_path=args.model_path,
            config_path=args.config_path,
            device=args.device,
            max_steps=args.max_steps,
            test_mode=args.test_mode,
            frequency=args.frequency,
            debug_image=args.debug_image
        )
        
        # 执行推理
        runner.run()
        
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # 清理资源
        if 'runner' in locals():
            runner.cleanup()
    
    print("推理脚本执行完成")
    return 0


if __name__ == "__main__":
    exit(main())
