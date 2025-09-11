#!/usr/bin/env python3
"""
基于相机和ACT模型的实时推理脚本 - 重构版本
使用与replay_trajectory相同的接口形式
"""

import os
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

# 添加项目路径到sys.path
project_dir = Path(__file__).parent.parent
sys.path.append(str(project_dir))

# 导入训练好的模型
from model.lerobot.common.policies.act.configuration_act import ACTConfig
from model.lerobot.common.policies.act.modeling_act import ACTPolicy

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
    "cam4": "327322062498",  
}

class CameraSystem:
    """相机系统接口 - 从inference_poly1复用"""
    
    def __init__(self):
        self.cameras = {}
        self.camera_names = ["cam4"]
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
            cam = D415(id=serial, depth=True, name=name)
            self.cameras[name] = cam
        if len(self.cameras) > 0:
            self.use_realsense = True
            print(f"使用 RealSense D415，相机数量: {len(self.cameras)}")
    
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
    """ACT策略包装器 - 将ACT模型包装为PolicyInterface兼容的策略"""
    
    def __init__(self, model_path, device="cpu", camera_system=None):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.camera_system = camera_system
        
        # 配置参数
        self.image_size = (224, 224)
        self.camera_names = ["cam4"]
        self.joint_dim = 7  # 7个关节角度（弧度）  
        self.gripper_dim = 1  # 1个夹爪开合值  
        self.action_dim = self.joint_dim + self.gripper_dim  # 总共8维  
        self.chunk_size = 32  # ACT模型的chunk大小
        
        # 单步预测模式，不需要chunk相关参数
        
        # 加载模型
        self.policy = self._load_policy()
        
        print(f"ACT策略初始化完成: {model_path}")
        print(f"使用设备: {self.device}")
    
    def _load_policy(self):
        """加载训练好的策略模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        # 加载配置
        config = ACTConfig(
            input_shapes={
                "observation.image.color": [3, 224, 224],
                "observation.state": [self.action_dim],
            },
            output_shapes={"action": [self.action_dim]},
            chunk_size=self.chunk_size,
            n_action_steps=self.chunk_size,
            input_normalization_modes={
                "observation.image.color": "min_max",
                "observation.state": "min_max",
            },
            output_normalization_modes={"action": "min_max"},
        )
        
        # 需要提供统计信息用于归一化
        dataset_stats = {
            "observation.image.color": {"min": torch.zeros(3, 1, 1), "max": torch.ones(3, 1, 1)},
            "observation.state": {  
                # 前7维是关节角度(-π到π弧度)，第8维是夹爪宽度(0到0.08米)  
                "min": torch.tensor([-3.14] * self.joint_dim + [0.0]),  
                "max": torch.tensor([3.14] * self.joint_dim + [0.08]),  
            },  
            "action": {  
                # 前7维是关节角度(-π到π弧度)，第8维是夹爪宽度(0到0.08米)  
                "min": torch.tensor([-3.14] * self.joint_dim + [0.0]),  
                "max": torch.tensor([3.14] * self.joint_dim + [0.08]),  
            },  
        }
        
        policy = ACTPolicy(config, dataset_stats=dataset_stats)
        
        try:
            # 使用 safetensors 加载模型权重
            state_dict = load_file(self.model_path / "model.safetensors")
            policy.load_state_dict(state_dict)
            print("模型权重加载成功")
            
            # 打印加载的统计信息用于调试
            if hasattr(policy, 'buffer_observation_state'):
                print(f"观测状态统计信息:")
                print(f"  min: {policy.buffer_observation_state['min']}")
                print(f"  max: {policy.buffer_observation_state['max']}")
            if hasattr(policy, 'buffer_action'):
                print(f"动作统计信息:")
                print(f"  min: {policy.buffer_action['min']}")
                print(f"  max: {policy.buffer_action['max']}")
                
        except Exception as e:
            print(f"模型权重加载失败: {e}")
            print("使用随机初始化")
            
        policy.to(self.device)
        policy.eval()
        return policy
    
    def preprocess_image(self, image):
        """预处理图像"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 调整尺寸到 224x224
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # 转换为 tensor 格式 (3, H, W)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        
        # 归一化到 [0, 1]
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
        # 预处理图像
        if self.camera_names[0] in images:
            img_tensor = self.preprocess_image(images[self.camera_names[0]])
        else:
            # 随机图像回退
            fake = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img_tensor = self.preprocess_image(fake)
        
        # 构建batch
        batch = {
            "observation.image.color": img_tensor.unsqueeze(0).to(self.device),
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
        
        # 返回关节动作（前7维）
        joint_action = full_action[:self.joint_dim]
        
        return joint_action
    
    def get_gripper_action(self, obs):
        """
        获取gripper动作
        
        Args:
            obs: 观测字典
            
        Returns:
            gripper_encoder: gripper编码器值 (0-255)
        """
        # 获取当前图像
        current_images = self.camera_system.get_all_images()
        
        # 获取当前状态
        current_state = self.get_current_state_with_gripper(obs)
        
        # 单步预测动作
        full_action = self.predict_single_action(current_images, current_state)
        
        # 获取gripper动作（第8维）
        gripper_width = full_action[self.joint_dim] - 0.005 # 夹爪宽度（米）
        
        # 将gripper宽度转换为编码器值
        # gripper宽度范围是0-0.08米，编码器范围是0-255
        gripper_encoder = int(gripper_width * 255 / 0.08)
        gripper_encoder = np.clip(gripper_encoder, 0, 255)
        
        return gripper_encoder


class ACTInferenceRunner:
    """ACT推理运行器 - 使用与replay_trajectory相同的接口形式"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 device: str = "cuda",
                 max_steps: int = 1000,
                 test_mode: bool = False,
                 frequency: float = 20.0):
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
        self.dt = 1.0 / frequency  # 时间间隔
        
        # 创建相机系统
        self.camera_system = CameraSystem()
        
        # 创建ACT策略
        self.policy = ACTPolicyWrapper(
            model_path=model_path,
            device=device,
            camera_system=self.camera_system
        )
        
        print(f"ACT推理运行器初始化完成")
        print(f"模型路径: {model_path}")
        print(f"配置文件: {config_path}")
        print(f"设备: {device}")
        print(f"测试模式: {test_mode}")
        print(f"推理频率: {frequency} Hz")
    
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
                'timestamp': time.time()
            }
            
            # 执行策略
            joint_action = self.policy(obs)
            gripper_action = self.policy.get_gripper_action(obs)
            
            print(f"预测的关节动作（7维）: {joint_action}")
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
            
            while True:
                if self.max_steps is not None and step >= self.max_steps:
                    print(f"达到最大步数 {self.max_steps}，停止运行")
                    break
                
                # 计算当前周期结束时间
                t_cycle_end = t_start + (step + 1) * self.dt
                
                # 获取观测
                obs = interface.get_observation()
                
                # 执行策略
                joint_action = self.policy(obs)
                gripper_action = self.policy.get_gripper_action(obs)
                
                # 执行动作
                interface.execute_action(joint_action)
                interface.execute_gripper_action(gripper_action)
                
                # 每50步打印一次
                if step % 5 == 0:
                    current_time = time.monotonic() - t_start
                    print(f"Step {step}: 时间={current_time:.2f}s, 关节位置 {obs['robot0_joint_pos']}, 动作 {joint_action}")
                    print(f"  Gripper动作: {gripper_action}")
                
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
    parser = argparse.ArgumentParser(description="基于相机和ACT模型的实时推理脚本 - 重构版本")
    parser.add_argument("--model_path", type=str, 
                       default="/home/robotflow/lerobot-main/src/temp/outputs/train/exo_act4/checkpoints/checkpoint_step_47500.safetensors",
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
    parser.add_argument("--frequency", type=float, default=20.0,
                       help="推理频率 (Hz)")
    
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
            frequency=args.frequency
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
