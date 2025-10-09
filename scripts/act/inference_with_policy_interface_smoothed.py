#!/usr/bin/env python3
"""
基于相机和ACT模型的实时推理脚本 - 更新版本
适配最新版本的lerobot库
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
from collections import deque

# 添加项目路径到sys.path，确保优先使用项目中的lerobot库
project_dir = Path(__file__).parent.parent
model_lerobot_path = project_dir / "model" / "lerobot" / "src"
sys.path.insert(0, str(model_lerobot_path))
sys.path.insert(0, str(project_dir))  # 添加项目根目录到路径

# 在添加路径后导入项目模块
from common.gripper_util import convert_gripper_width_to_encoder

# 导入最新版本的lerobot库
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.constants import OBS_IMAGES, ACTION, OBS_STATE

# 导入PolicyInterface
from policy_interface import create_policy_interface

# 导入precise_wait
from common.precise_sleep import precise_wait

# 相机相关导入
import pyrealsense2 as rs
from r3kit.devices.camera.realsense import config as rs_cfg
from r3kit.devices.camera.realsense.d415 import D415
R3KIT_RS_AVAILABLE = True

class ActionSmoother:
    """动作平滑器 - 检测突变并平滑动作"""
    
    def __init__(self, mutation_threshold=0.1,history_size=10):
        """
        初始化动作平滑器
        
        Args:
            mutation_threshold: 突变阈值 (rad)
            history_size: 历史动作存储大小
        """
        self.mutation_threshold = mutation_threshold
        self.history_size = history_size
        
        # 存储历史动作（用于突变检测）
        self.joint_history = deque(maxlen=history_size)
        self.step_count = 0
        
        # 统计信息
        self.mutation_count = 0
        self.total_steps = 0
        
    def smooth_action(self, joint_action):
        """
        平滑关节动作 - 修复版本
        
        Args:
            joint_action: 7维关节动作数组
            
        Returns:
            smoothed_action: 平滑后的7维关节动作数组
        """
        self.total_steps += 1
        self.step_count += 1
        joint_action = np.array(joint_action)
        
        # 如果没有足够的历史数据，直接返回原始动作并存储
        if len(self.joint_history) < 1:
            self.joint_history.append(joint_action.copy())
            return joint_action
        
        # 计算动作变化率（与历史记录中的最后一个动作比较）
        prev_joint = self.joint_history[-1]  # 使用历史记录中的最后一个动作
        curr_joint = joint_action
        change_vector = curr_joint - prev_joint
        change_rate = np.linalg.norm(change_vector)
        
        # 检查是否发生突变
        if change_rate > self.mutation_threshold:
            self.mutation_count += 1
            
            # 计算缩放因子，使变化率等于阈值
            scale_factor = self.mutation_threshold / change_rate
            
            # 缩放变化向量，保持方向不变
            smoothed_change = change_vector * scale_factor
            smoothed_action = prev_joint + smoothed_change
            
            print(f"🚨 检测到突变! 步骤: {self.step_count}")
            print(f"  原始变化率: {change_rate:.6f} rad")
            print(f"  缩放因子: {scale_factor:.4f}")
            print(f"  平滑后变化率: {np.linalg.norm(smoothed_change):.6f} rad")
            print(f"  主要变化关节: {self._find_max_change_joint(change_vector)}")
            print("-" * 40)
            
            # 存储平滑后的动作到历史记录
            self.joint_history.append(smoothed_action.copy())
            return smoothed_action
        else:
            # 没有突变，存储原始动作并返回
            self.joint_history.append(joint_action.copy())
            return joint_action
    
    def _find_max_change_joint(self, change_vector):
        """找到变化最大的关节"""
        change_vector = np.array(change_vector)
        max_joint_idx = np.argmax(np.abs(change_vector))
        return f"关节{max_joint_idx+1} (变化: {change_vector[max_joint_idx]:.4f} rad)"
    
    def get_statistics(self):
        """获取统计信息"""
        if self.total_steps == 0:
            return {
                'total_steps': 0,
                'mutation_count': 0,
                'mutation_rate': 0.0
            }
        
        return {
            'total_steps': self.total_steps,
            'mutation_count': self.mutation_count,
            'mutation_rate': self.mutation_count / self.total_steps
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print(f"\n=== 动作平滑统计 ===")
        print(f"总步数: {stats['total_steps']}")
        print(f"突变次数: {stats['mutation_count']}")
        print(f"突变比例: {stats['mutation_rate']:.2%}")
        print(f"突变阈值: {self.mutation_threshold} rad")

# D415 相机配置（与采集脚本保持一致）
FPS = 30
D415_CAMERAS = {   
    "cam4": "327322062498",  # 固定机位视角
    "eih": "038522062288",   # eye-in-hand视角（需要根据实际序列号修改）
}

class CameraSystem:
    """相机系统接口 - 从inference_poly1复用"""
    
    def __init__(self):
        self.cameras = {}
        self.camera_names = ["cam4", "eih"]  # 支持双视角
        self.use_realsense = True
        
        # 与采集脚本保持一致的流配置
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
                # 转 RGB（下游预处理默认以 RGB 处理）
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


class ACTPolicyWrapper:
    """ACT策略包装器 - 适配最新版本的lerobot库"""
    
    def __init__(self, model_path, device="cpu", camera_system=None, debug_image=False,use_eih=True):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.camera_system = camera_system
        self.debug_image = debug_image
        self.use_eih = use_eih
        # 配置参数
        self.image_size = (224, 224)
        self.camera_names = ["cam4"]  # 默认只有固定机位视角
        if self.use_eih:
            self.camera_names.append("eih")  # 如果需要eih，添加到相机列表
        self.joint_dim = 7  # 7个关节角度（弧度）  
        self.gripper_dim = 1  # 1个夹爪开合值  
        self.action_dim = self.joint_dim + self.gripper_dim  # 总共8维  
        self.chunk_size = 100  # ACT模型的chunk大小
        
        # 加载模型
        self.policy = self._load_policy()
        
        print(f"ACT策略初始化完成: {model_path}")
        print(f"使用设备: {self.device}")
        print(f"支持视角: 固定机位(cam4)" + (" + eye-in-hand(eih)" if self.use_eih else ""))
        print(f"相机系统状态: {len(self.camera_system.cameras) if self.camera_system else 0} 个相机已初始化")
    
    def _load_policy(self):
        """加载训练好的策略模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        # 使用from_pretrained加载模型(推荐方式)
        policy = ACTPolicy.from_pretrained(
            pretrained_name_or_path=str(self.model_path)
        )
        
        # 移动到指定设备
        policy.to(self.device)
        
        # 设置执行部署
        policy.config.n_action_steps = 50

        # 打印配置信息
        print(f"模型加载成功:")
        print(f" 策略类型: {policy.config.type}")
        print(f" 设备: {next(policy.parameters()).device}")
        print(f" 时间集成系数: {policy.config.temporal_ensemble_coeff}")
        print(f" 动作步数: {policy.config.n_action_steps}")
        print(f" 块大小: {policy.config.chunk_size}")
        
        return policy
    
    def preprocess_image(self, image, debug=False):
        """预处理图像 - 与训练时保持一致：先裁剪成正方形，再缩放到目标尺寸"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 获取原始图像尺寸
        width, height = image.size
        if debug:
            print(f"原始图像尺寸: {width}x{height}")
        
        # 按照训练时的处理方式裁剪
        if width == 640 and height == 480:
            # 640*480尺寸：从特定位置裁剪到360*360
            left = 200
            right = 560
            top = 0
            bottom = 360
            if debug:
                print(f"640x480图片，裁剪区域: ({left}, {top}, {right}, {bottom})")
        else:
            # 其他尺寸：按比例裁剪成正方形
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            right = left + min_dim
            top = (height - min_dim) // 2
            bottom = top + min_dim
            if debug:
                print(f"其他尺寸图片，裁剪成正方形: ({left}, {top}, {right}, {bottom})")
        
        # 裁剪
        image_cropped = image.crop((left, top, right, bottom))
        if debug:
            print(f"裁剪后尺寸: {image_cropped.size}")
        
        # 缩放到目标尺寸
        image_resized = image_cropped.resize(self.image_size, Image.Resampling.LANCZOS)
        if debug:
            print(f"缩放后尺寸: {image_resized.size}")
        
        # 转换为tensor并归一化
        image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float()  # (3, H, W)
        image_tensor = image_tensor / 255.0
        
        return image_tensor
    
    def get_current_state_with_gripper(self, obs):
        """从观测中获取当前状态（8维）"""
        # 从观测中提取关节位置（弧度）
        joints_rad = obs['robot0_joint_pos']  # (7,)
        
        # 获取夹爪宽度（从观测中获取，如果没有则使用默认值）
        if 'robot0_gripper_width' in obs:
            gripper_width = obs['robot0_gripper_width']
            if isinstance(gripper_width, np.ndarray):
                gripper_width = gripper_width[0] if len(gripper_width) > 0 else 0.04
        else:
            gripper_width = 0.04  # 默认夹爪宽度（米）
        
        # 返回8维状态：7个关节角度（弧度） + 1个夹爪宽度（米）
        return np.concatenate([joints_rad, [gripper_width]])
    
    def predict_single_action(self, images, current_state):
        """
        单步预测动作（使用 ACTPolicy.select_action）。
        返回: (8,) numpy 数组，前7维为关节(弧度)，第8维为夹爪(米)。
        """
        # 预处理固定机位视角图像
        if "cam4" in images:
            color_img_tensor = self.preprocess_image(images["cam4"], debug=self.debug_image)
        else:
            # 随机图像回退
            fake = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            color_img_tensor = self.preprocess_image(fake, debug=self.debug_image)
            print("警告: 固定机位视角图像获取失败，使用模拟图像")
        
        # 构建batch - 根据是否使用eih来决定输入格式
        if self.use_eih:
            # 预处理eye-in-hand视角图像
            if "eih" in images:
                eih_img_tensor = self.preprocess_image(images["eih"], debug=self.debug_image)
            else:
                # 随机图像回退
                fake = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                eih_img_tensor = self.preprocess_image(fake, debug=self.debug_image)
                print("警告: eye-in-hand视角图像获取失败，使用模拟图像")
            
            batch = {
                "observation.images.cam": color_img_tensor.unsqueeze(0).to(self.device),
                "observation.images.eih": eih_img_tensor.unsqueeze(0).to(self.device),
                "observation.state": torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(self.device),
            }
        else:
            # 只使用固定机位视角
            batch = {
                "observation.images.cam": color_img_tensor.unsqueeze(0).to(self.device),
                "observation.state": torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(self.device),
            }
        
        with torch.no_grad():
            # 使用select_action进行单步预测
            action = self.policy.select_action(batch)  # (1, action_dim)，已反归一化
            action = action.squeeze(0).detach().cpu().numpy()  # (8,)
        
        # 直接返回模型输出，所有单位都是弧度（前7维）和米（第8维）
        return action
    
    
    def __call__(self, obs):
        """
        策略函数 - PolicyInterface兼容接口
        
        Args:
            obs: 观测字典，包含robot0_joint_pos等
            
        Returns:
            action: 7维关节动作 [j1, j2, j3, j4, j5, j6, j7] (弧度)
        """
        # 获取当前图像
        current_images = self.camera_system.get_all_images()
        
        # 获取当前状态
        current_state = self.get_current_state_with_gripper(obs)
        
        # 单步预测动作
        full_action = self.predict_single_action(current_images, current_state)
        #print(f"预测的完整动作（8维）: {full_action}")
        
        joint_action = full_action[:self.joint_dim]
        # 获取gripper动作（第8维）
        gripper_width = full_action[self.joint_dim] #+ 0.05 # 夹爪宽度（米）

        # if gripper_width > 0.035:
        #     gripper_width *= 1.5
        # elif gripper_width < 0.025:
        #     gripper_width *= 0.7
        #gripper_width = 0.08

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


class ACTInferenceRunner:
    """ACT推理运行器 - 使用与replay_trajectory相同的接口形式"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 device: str = "cuda",
                 max_steps: int = 1000,
                 test_mode: bool = False,
                 frequency: float = 20.0,
                 debug_image: bool = False,
                 use_eih: bool = True):
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
        self.use_eih = use_eih
        self.dt = 1.0 / frequency  # 时间间隔
        
        # 创建相机系统
        self.camera_system = CameraSystem()
        
        # 创建ACT策略
        self.policy = ACTPolicyWrapper(
            model_path=model_path,
            device=device,
            camera_system=self.camera_system,
            debug_image=self.debug_image,
            use_eih=self.use_eih
        )
        
        # 创建动作平滑器
        self.action_smoother = ActionSmoother(mutation_threshold=0.01,history_size=10)
        
        print(f"ACT推理运行器初始化完成")
        print(f"模型路径: {model_path}")
        print(f"配置文件: {config_path}")
        print(f"设备: {device}")
        print(f"测试模式: {test_mode}")
        print(f"推理频率: {frequency} Hz")
        print(f"使用eih: {self.use_eih}")
        
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
                    last_joint_action = joint_action.copy()  # 保存7维关节动作
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
                        # 不要使用当前位置，而是跳过这次执行
                        joint_action = obs['robot0_joint_pos'] + np.random.normal(0, 0.001, 7)
                        print(f"⚠️  使用随机扰动动作，等待有效推理: 推理时间={inference_time:.3f}s")
                        continue  # 跳过这次循环
                
                # 动作平滑处理
                smoothed_joint_action = self.action_smoother.smooth_action(joint_action)
                
                # 执行动作
                interface.execute_action(smoothed_joint_action)
                interface.execute_gripper_action(gripper_action)
                
                # 每10步打印一次详细信息
                if step % 10 == 0:
                    current_time = time.monotonic() - t_start
                    avg_inference_time = np.mean(inference_times[-10:]) if len(inference_times) >= 10 else np.mean(inference_times)
                    print(f"Step {step}: 时间={current_time:.2f}s, 推理时间={inference_time:.3f}s (平均={avg_inference_time:.3f}s)")
                    print(f"  关节动作: {joint_action}")
                    print(f"  Gripper动作: {gripper_action}")
                    if timeout_count > 0:
                        print(f"  超时次数: {timeout_count}")
                
                step += 1
                
                # 使用precise_wait等待到下一个周期
                precise_wait(t_cycle_end)
                
        except KeyboardInterrupt:
            print("\n用户中断，停止策略...")
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 打印动作平滑统计信息
            self.action_smoother.print_statistics()
            
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
    parser = argparse.ArgumentParser(description="基于相机和ACT模型的实时推理脚本 - 更新版本")
    parser.add_argument("--model_path", type=str, 
                       default="/home/robotflow/Downloads/060000/pretrained_model",
                       help="训练好的模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备 (cpu/cuda)")
    parser.add_argument("--config_path", type=str,
                       default="/home/robotflow/my_code/other_codes/franka_control_final/config/robot_config.yaml",
                       help="机器人配置文件路径")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="最大运行步数")
    parser.add_argument("--test_mode", action="store_true", default=False,
                       help="测试模式（不连接真实机器人）")
    parser.add_argument("--frequency", type=float, default=10.0,
                       help="推理频率 (Hz) - 针对130ms推理时间优化")
    parser.add_argument("--debug_image", action="store_true", default=False,
                       help="显示图像处理调试信息")
    parser.add_argument("--use_eih", action="store_true", default=False,  # 新增
                       help="使用eye-in-hand视角作为输入")
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config_path):
        print(f"错误: 配置文件不存在: {args.config_path}")
        return 1
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return 1
    
    # 创建并运行ACT推理运行器
    try:
        runner = ACTInferenceRunner(
            model_path=args.model_path,
            config_path=args.config_path,
            device=args.device,
            max_steps=args.max_steps,
            test_mode=args.test_mode,
            frequency=args.frequency,
            debug_image=args.debug_image,
            use_eih=args.use_eih
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
