#!/usr/bin/env python3
"""
基于相机和ACT模型的实时推理脚本 - 调试版本
适配最新版本的lerobot库，集成推理数据保存功能
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

# 导入debug_logger
from debug_logger import InferenceLogger, AnomalyDetector


# 添加项目路径到sys.path，确保优先使用项目中的lerobot库
project_dir = Path(__file__).parent.parent
model_lerobot_path = project_dir / "model" / "lerobot" / "src"
sys.path.insert(0, str(model_lerobot_path))
sys.path.insert(0, str(project_dir))  # 添加项目根目录到路径

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
    
    def __init__(self, model_path, device="cpu", camera_system=None, debug_image=False):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.camera_system = camera_system
        self.debug_image = debug_image
        
        # 配置参数
        self.image_size = (224, 224)
        self.camera_names = ["cam4", "eih"]  # 支持双视角
        self.joint_dim = 7  # 7个关节角度（弧度）  
        self.gripper_dim = 1  # 1个夹爪开合值  
        self.action_dim = self.joint_dim + self.gripper_dim  # 总共8维  
        self.chunk_size = 32  # ACT模型的chunk大小
        
        # 加载模型
        self.policy = self._load_policy()
        
        print(f"ACT策略初始化完成: {model_path}")
        print(f"使用设备: {self.device}")
        print(f"支持双视角输入: 固定机位(cam4) + eye-in-hand(eih)")
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
        
        # 先裁剪成正方形（取较小的边作为边长）
        if width > height:
            # 宽度大于高度，从中心裁剪
            left = (width - height) // 2
            right = left + height
            top = 0
            bottom = height
        else:
            # 高度大于等于宽度，从中心裁剪
            top = (height - width) // 2
            bottom = top + width
            left = 0
            right = width
        
        # 裁剪成正方形
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
        
        # 预处理eye-in-hand视角图像
        if "eih" in images:
            eih_img_tensor = self.preprocess_image(images["eih"], debug=self.debug_image)
        else:
            # 随机图像回退
            fake = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            eih_img_tensor = self.preprocess_image(fake, debug=self.debug_image)
            print("警告: eye-in-hand视角图像获取失败，使用模拟图像")
        
        # 构建batch - 使用新版本的格式
        batch = {
            "observation.image.color": color_img_tensor.unsqueeze(0).to(self.device),
            "observation.image.eih": eih_img_tensor.unsqueeze(0).to(self.device),
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
        
        joint_action = full_action[:self.joint_dim]
        # 获取gripper动作（第8维）
        gripper_width = full_action[self.joint_dim] - 0.005 # 夹爪宽度（米）

        gripper_encoder = convert_gripper_width_to_encoder(gripper_width)

        cur_action = np.concatenate([joint_action, [gripper_encoder]])
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
                 debug_log_dir: str = "debug_logs",
                 save_frequency: int = 1):
        """
        初始化ACT推理运行器
        
        Args:
            model_path: 模型路径
            config_path: 配置文件路径
            device: 计算设备
            max_steps: 最大运行步数
            test_mode: 测试模式
            frequency: 推理频率 (Hz)
            debug_image: 是否显示图像处理调试信息
            debug_log_dir: 调试日志保存目录
            save_frequency: 数据保存频率（每N步保存一次）
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
        
        # 创建ACT策略
        self.policy = ACTPolicyWrapper(
            model_path=model_path,
            device=device,
            camera_system=self.camera_system,
            debug_image=self.debug_image
        )
        
        # 初始化调试记录器
        self.logger = InferenceLogger(
            log_dir=debug_log_dir,
            save_frequency=save_frequency,
            save_images=True,
            max_logs=10000
        )
        
        # 初始化异常检测器
        self.detector = AnomalyDetector(
            action_threshold=0.5,
            inference_time_threshold=0.1,
            gripper_threshold=10
        )
        
        print(f"ACT推理运行器初始化完成")
        print(f"模型路径: {model_path}")
        print(f"配置文件: {config_path}")
        print(f"设备: {device}")
        print(f"测试模式: {test_mode}")
        print(f"推理频率: {frequency} Hz")
        print(f"调试日志目录: {debug_log_dir}")
        print(f"保存频率: 每{save_frequency}步")
        
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
            
            # 获取图像数据
            current_images = self.camera_system.get_all_images()
            
            # 执行策略
            t_start = time.monotonic()
            cur_action = self.policy(obs)
            t_end = time.monotonic()
            
            joint_action = cur_action[:self.policy.joint_dim]
            gripper_action = cur_action[self.policy.joint_dim]
            
            print(f"预测的关节动作（7维）: {joint_action}")
            print(f"预测的夹爪动作（1维）: {gripper_action}")
            print(f"预测的完整动作（8维）: {cur_action}")
            
            # 记录调试数据
            input_data = {
                "cam_image": current_images.get("cam4"),
                "eih_image": current_images.get("eih"),
                "robot_state": obs['robot0_joint_pos'],
                "gripper_state": obs['robot0_gripper_width'][0]
            }
            
            output_data = {
                "joint_action": joint_action,
                "gripper_action": gripper_action,
                "gripper_width": obs['robot0_gripper_width'][0],
                "full_action": cur_action
            }
            
            metadata = {
                "inference_time": t_end - t_start,
                "step": i,
                "test_mode": True
            }
            
            # 保存记录
            record_id = self.logger.log_inference(input_data, output_data, metadata)
            print(f"调试记录已保存: {record_id}")
            
            # 异常检测
            anomalies = self.detector.detect_anomalies(input_data, output_data, metadata)
            if anomalies:
                print(f"⚠️  检测到异常: {anomalies}")
            
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
                
                # 获取当前图像
                current_images = self.camera_system.get_all_images()
                
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
                    last_joint_action = cur_action.copy()
                    last_gripper_action = cur_action[self.policy.joint_dim]
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
                    remaining_time < 0.01 or  # 剩余时间少于10ms
                    timeout_count > 0):
                    
                    if last_joint_action is not None and last_gripper_action is not None:
                        # 使用上次的有效动作
                        joint_action = last_joint_action
                        gripper_action = last_gripper_action
                        print(f"⚠️  使用降级策略: 推理时间={inference_time:.3f}s, 剩余时间={remaining_time:.3f}s")
                    else:
                        # 如果没有任何有效动作，使用当前位置
                        joint_action = obs['robot0_joint_pos']
                        gripper_action = 128  # 默认gripper位置
                        print(f"⚠️  使用当前位置: 推理时间={inference_time:.3f}s")
                
                # 记录调试数据
                input_data = {
                    "cam_image": current_images.get("cam4"),
                    "eih_image": current_images.get("eih"),
                    "robot_state": obs['robot0_joint_pos'],
                    "gripper_state": obs['robot0_gripper_width'][0] if 'robot0_gripper_width' in obs else 0.04
                }
                
                output_data = {
                    "joint_action": joint_action,
                    "gripper_action": gripper_action,
                    "gripper_width": obs['robot0_gripper_width'][0] if 'robot0_gripper_width' in obs else 0.04,
                    "full_action": cur_action
                }
                
                metadata = {
                    "inference_time": inference_time,
                    "step": step,
                    "timeout_count": timeout_count,
                    "n_action_steps": self.policy.policy.config.n_action_steps,
                    "chunk_size": self.policy.policy.config.chunk_size
                }
                
                # 保存记录
                record_id = self.logger.log_inference(input_data, output_data, metadata)
                
                # 异常检测
                anomalies = self.detector.detect_anomalies(input_data, output_data, metadata)
                if anomalies:
                    print(f"⚠️  检测到异常: {anomalies}")
                
                # 执行动作
                interface.execute_action(joint_action)
                interface.execute_gripper_action(gripper_action)
                
                # 每10步打印一次详细信息
                if step % 10 == 0:
                    current_time = time.monotonic() - t_start
                    avg_inference_time = np.mean(inference_times[-10:]) if len(inference_times) >= 10 else np.mean(inference_times)
                    print(f"Step {step}: 时间={current_time:.2f}s, 推理时间={inference_time:.3f}s (平均={avg_inference_time:.3f}s)")
                    print(f"  关节动作: {joint_action}")
                    print(f"  Gripper动作: {gripper_action}")
                    print(f"  记录ID: {record_id}")
                    if timeout_count > 0:
                        print(f"  超时次数: {timeout_count}")
                    if anomalies:
                        print(f"  异常: {anomalies}")
                
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
            # 保存会话总结
            print("\n保存调试数据...")
            summary_file = self.logger.save_session_summary()
            stats = self.logger.get_stats()
            print(f"调试数据已保存:")
            print(f"  总步数: {stats['total_steps']}")
            print(f"  已保存步数: {stats['saved_steps']}")
            print(f"  平均推理时间: {stats['avg_inference_time']:.3f}s")
            print(f"  异常检测次数: {self.detector.get_anomaly_stats()['anomaly_count']}")
            print(f"  会话总结: {summary_file}")
            
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
    # 设置默认参数，不需要命令行传参
    args = type('Args', (), {
        'model_path': "./outputs/train/act_franka_dataset/checkpoints/050000",  # 默认模型路径
        'device': "cuda",  # 默认使用GPU
        'config_path': "./config/robot_config.yaml",  # 默认配置文件路径
        'max_steps': 1000,  # 默认最大步数
        'test_mode': False,  # 默认使用实时模式（安全）
        'frequency': 10.0,  # 默认推理频率
        'debug_image': False,  # 默认不显示图像调试信息
        'debug_log_dir': "debug_logs",  # 默认调试日志目录
        'save_frequency': 1  # 默认每步都保存
    })()
    
    print("🔧 使用默认参数:")
    print(f"  模型路径: {args.model_path}")
    print(f"  设备: {args.device}")
    print(f"  配置文件: {args.config_path}")
    print(f"  最大步数: {args.max_steps}")
    print(f"  测试模式: {args.test_mode}")
    print(f"  推理频率: {args.frequency} Hz")
    print(f"  调试日志目录: {args.debug_log_dir}")
    print(f"  保存频率: 每{args.save_frequency}步")
    print("💡 如需修改参数，请直接编辑脚本中的默认值")
    
    # 检查配置文件
    if not os.path.exists(args.config_path):
        print(f"⚠️  配置文件不存在: {args.config_path}")
        print("💡 请确保配置文件路径正确，或修改脚本中的默认路径")
        return 1
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"⚠️  模型路径不存在: {args.model_path}")
        print("💡 请确保模型路径正确，或修改脚本中的默认路径")
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
            debug_log_dir=args.debug_log_dir,
            save_frequency=args.save_frequency
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
