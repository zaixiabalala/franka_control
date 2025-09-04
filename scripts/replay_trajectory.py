#!/usr/bin/env python3
"""
正确的轨迹复现脚本 - 直接使用轨迹数据模拟模型推理
不使用插值器，直接按时间索引获取动作数据
"""

import argparse
import time
import sys
import os
from pathlib import Path
import numpy as np
import scipy.spatial.transform as st
import torch
import yaml

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy_interface import create_policy_interface

class TrajectoryPolicy:
    """轨迹策略 - 直接使用预录制的关节轨迹数据模拟模型推理"""
    
    def __init__(self, 
                 trajectory_data: np.ndarray,
                 start_time: float,
                 data_frequency: float = 30.0):
        """
        初始化轨迹策略
        
        Args:
            trajectory_data: 轨迹数据 (N, 7) - 7维关节角度数据
            start_time: 开始时间
            data_frequency: 数据频率 (Hz)
        """
        self.trajectory_data = trajectory_data
        self.start_time = start_time
        self.data_frequency = data_frequency
        self.dt = 1.0 / data_frequency
        
    def __call__(self, obs):
        """
        策略函数 - 模拟模型推理
        
        Args:
            obs: 观测字典 (这里不使用，只是为了接口兼容)
            
        Returns:
            action: 7维关节角度 [j1, j2, j3, j4, j5, j6, j7]
        """
        # 计算当前时间
        current_time = time.time() - self.start_time
        
        # 根据时间计算数据索引
        data_index = int(current_time * self.data_frequency)
        
        # 确保索引不超出范围
        if data_index >= len(self.trajectory_data):
            data_index = len(self.trajectory_data) - 1
        
        # 直接返回对应的动作数据
        action = self.trajectory_data[data_index].copy()
        
        # 调试信息：每100次调用打印一次
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1
        
        if self._call_count % 100 == 0:
            print(f"策略调用 #{self._call_count}: 时间={current_time:.2f}s, 索引={data_index}, 动作={action}")
        
        return action


class CorrectTrajectoryReplayer:
    """正确的轨迹复现器"""
    
    def __init__(self, 
                 config_path: str,
                 angles_dir: str,
                 policy_frequency: float = 20.0,
                 data_frequency: float = 30.0):
        """
        初始化轨迹复现器
        
        Args:
            config_path: 配置文件路径
            angles_dir: 轨迹数据目录
            policy_frequency: 策略推理频率 (Hz)
            data_frequency: 数据采集频率 (Hz)
        """
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'robot_config.yaml')
        self.angles_dir = Path(angles_dir)
        self.policy_frequency = policy_frequency
        self.data_frequency = data_frequency
        self.joint_dim = 7
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 直接加载关节数据作为轨迹数据
        self.trajectory_data = self._load_joints_data()
        
    def _load_joints_data(self):
        """只加载关节数据，不进行转换"""
        # 查找轨迹文件
        pat1 = sorted(self.angles_dir.glob("angle_cam0_*.npy"))
        files = pat1 if len(pat1) > 0 else sorted(self.angles_dir.glob("*.npy"))
        
        if len(files) == 0:
            raise FileNotFoundError(f"未在 {self.angles_dir} 找到角度 .npy 文件")
        
        print(f"找到 {len(files)} 个轨迹文件")
        
        # 加载关节数据
        all_joints = []
        for i, f in enumerate(files):
            arr = np.load(f)
            if arr.shape[0] != 8:
                raise ValueError(f"{f} 维度异常，期望 8，得到 {arr.shape}")
            
            joints_deg = np.array(arr[:self.joint_dim], dtype=np.float32)
            joints_rad = np.radians(joints_deg).astype(np.float32)
            all_joints.append(joints_rad)
        
        all_joints = np.array(all_joints)
        print(f"关节轨迹数据: {len(all_joints)} 帧")
        print(f"数据频率: {self.data_frequency}Hz")
        print(f"总时长: {len(all_joints) / self.data_frequency:.2f}s")
        
        return all_joints
    
    
    def run(self):
        """执行轨迹复现"""
        print(f"开始轨迹复现")
        print(f"策略推理频率: {self.policy_frequency}Hz")
        print(f"数据采集频率: {self.data_frequency}Hz")
        
        # 先创建一个临时策略用于启动接口
        temp_policy = lambda obs: np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
        
        # 创建策略接口
        interface = create_policy_interface(self.config_path, temp_policy)
        
        try:
            # 启动策略接口
            print("启动策略接口...")
            interface.start()
            print("策略接口已启动!")
            
            # # 获取机器人接口进行正向运动学计算
            # from real_world.franka_interpolation_controller import FrankaInterface
            # robot_ip = self.config['robot']['ip']
            # robot_port = self.config['robot']['port']
            # robot_interface = FrankaInterface(ip=robot_ip, port=robot_port)
            
            # 直接使用关节数据作为轨迹数据
            print("使用关节轨迹数据...")
            trajectory_data = self.trajectory_data
            
            print(f"轨迹数据形状: {trajectory_data.shape}")
            print(f"轨迹数据范围: 关节角度 [{trajectory_data.min():.3f}, {trajectory_data.max():.3f}]")
            
            # 创建真正的轨迹策略
            start_time = time.time()
            policy = TrajectoryPolicy(
                trajectory_data=trajectory_data,
                start_time=start_time,
                data_frequency=self.data_frequency
            )
            
            # 更新策略接口的策略
            interface.policy_fn = policy
            
            # 获取初始观测
            obs = interface.get_observation()
            print(f"初始关节位置: {obs['robot0_joint_pos']}")
            print(f"初始关节速度: {obs['robot0_joint_vel']}")

            # 计算总执行时间
            total_data_time = len(trajectory_data) / self.data_frequency
            print(f"预计执行时间: {total_data_time:.2f}s")
            print(f"策略频率: {self.policy_frequency}Hz")
            print(f"数据频率: {self.data_frequency}Hz")
            
            # 运行策略
            print("开始轨迹复现...")
            print("按 Ctrl+C 停止")
            
            step = 0
            last_progress_time = time.time()
            
            while True:
                current_time = time.time() - start_time
                
                # 检查是否完成
                if current_time >= total_data_time:
                    print("轨迹复现完成!")
                    break
                
                # 获取观测
                obs = interface.get_observation()
                current_pos = obs['robot0_joint_pos']  # 使用关节位置
                
                # 执行策略 (模拟模型推理)
                action = policy(obs)
                
                # 计算数据索引用于调试
                data_index = int(current_time * self.data_frequency)
                if data_index >= len(trajectory_data):
                    data_index = len(trajectory_data) - 1
                
                # 每10步打印一次详细信息
                if step % 10 == 0:
                    print(f"Step {step}: 时间={current_time:.2f}s, 数据索引={data_index}")
                    print(f"  当前关节: {current_pos}")
                    print(f"  目标关节: {action}")
                    print(f"  关节误差: {np.linalg.norm(action - current_pos):.4f}rad")
                
                # 执行动作
                try:
                    interface.execute_action(action)
                except Exception as e:
                    print(f"执行动作失败: {e}")
                    print(f"  当前步数: {step}")
                    print(f"  当前时间: {current_time:.2f}s")
                    print(f"  动作: {action}")
                    raise
                
                # 进度显示
                if step % (self.policy_frequency * 2) == 0:  # 每2秒显示一次
                    progress = (current_time / total_data_time) * 100
                    print(f"进度: {progress:.1f}% ({current_time:.1f}s/{total_data_time:.1f}s)")
                
                step += 1
                
                # 按策略频率等待
                time.sleep(1.0 / self.policy_frequency)
                
        except KeyboardInterrupt:
            print("\n轨迹复现被用户中断")
        except Exception as e:
            print(f"轨迹复现出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 停止策略接口
            print("停止策略接口...")
            interface.stop()
            print("轨迹复现结束")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="正确的轨迹复现脚本")
    parser.add_argument("--config", type=str, 
                        default="config/robot_config.yaml",
                        help="机器人配置文件路径")
    parser.add_argument("--angles_dir", type=str, 
                        default="/home/franka/code/polymetis/examples/angles1/angles",
                        help="轨迹数据目录")
    parser.add_argument("--policy_frequency", type=float, default=20.0,
                        help="策略推理频率 (Hz)")
    parser.add_argument("--data_frequency", type=float, default=30.0,
                        help="数据采集频率 (Hz)")
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    # 检查数据目录
    if not os.path.exists(args.angles_dir):
        print(f"错误: 数据目录不存在: {args.angles_dir}")
        return
    
    # 创建轨迹复现器
    try:
        replayer = CorrectTrajectoryReplayer(
            config_path=args.config,
            angles_dir=args.angles_dir,
            policy_frequency=args.policy_frequency,
            data_frequency=args.data_frequency
        )
        
        # 执行复现
        replayer.run()
        
    except Exception as e:
        print(f"轨迹复现失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
