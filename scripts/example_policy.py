#!/usr/bin/env python3
"""
策略示例脚本
演示如何使用不同的策略控制Franka机器人
"""

import sys
import os
import time
import numpy as np
import yaml

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy_interface import create_policy_interface, SimplePolicy


class CirclePolicy:
    """圆形运动策略"""
    
    def __init__(self, center, radius=0.05, height=0.3, frequency=0.1):
        """
        初始化圆形运动策略
        
        Args:
            center: 圆心位置 [x, y]
            radius: 半径 (米)
            height: 高度 (米)
            frequency: 运动频率 (Hz)
        """
        self.center = np.array(center)
        self.radius = radius
        self.height = height
        self.frequency = frequency
        self.start_time = None
        
    def __call__(self, obs):
        """策略函数"""
        if self.start_time is None:
            self.start_time = time.monotonic()
            
        # 计算时间
        t = time.monotonic() - self.start_time
        
        # 计算圆形轨迹
        angle = 2 * np.pi * self.frequency * t
        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        z = self.height
        
        # 保持当前旋转
        current_rot = obs['robot0_eef_rot_axis_angle']
        
        return np.array([x, y, z, current_rot[0], current_rot[1], current_rot[2]])


class SinusoidalPolicy:
    """正弦波运动策略"""
    
    def __init__(self, amplitude=0.05, frequency=0.2, axis='x'):
        """
        初始化正弦波运动策略
        
        Args:
            amplitude: 振幅 (米)
            frequency: 频率 (Hz)
            axis: 运动轴 ('x', 'y', 'z')
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.axis = axis
        self.start_time = None
        self.base_pose = None
        
    def __call__(self, obs):
        """策略函数"""
        if self.start_time is None:
            self.start_time = time.monotonic()
            self.base_pose = np.concatenate([
                obs['robot0_eef_pos'],
                obs['robot0_eef_rot_axis_angle']
            ])
            
        # 计算时间
        t = time.monotonic() - self.start_time
        
        # 计算正弦波偏移
        offset = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        
        # 应用偏移
        action = self.base_pose.copy()
        if self.axis == 'x':
            action[0] += offset
        elif self.axis == 'y':
            action[1] += offset
        elif self.axis == 'z':
            action[2] += offset
            
        return action


def step_callback(step, obs, action):
    """步进回调函数"""
    if step % 50 == 0:  # 每50步打印一次
        print(f"Step {step}: Position {obs['robot0_eef_pos']}, Action {action[:3]}")


def main():
    """主函数"""
    print("Franka机器人策略示例")
    print("=" * 50)
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'robot_config.yaml')
    
    # 选择策略
    print("请选择策略:")
    print("1. 简单目标策略")
    print("2. 圆形运动策略")
    print("3. 正弦波运动策略")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == '1':
        # 简单目标策略 - 关节位置控制
        target_joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])  # 7个关节角度
        policy = SimplePolicy(target_joints)
        print(f"使用简单目标策略，目标关节: {target_joints}")
        
    elif choice == '2':
        # 圆形运动策略
        center = [0.5, 0.0]
        radius = 0.05
        height = 0.3
        policy = CirclePolicy(center, radius, height)
        print(f"使用圆形运动策略，圆心: {center}, 半径: {radius}m, 高度: {height}m")
        
    elif choice == '3':
        # 正弦波运动策略
        axis = input("选择运动轴 (x/y/z): ").strip().lower()
        if axis not in ['x', 'y', 'z']:
            axis = 'x'
        policy = SinusoidalPolicy(amplitude=0.03, frequency=0.1, axis=axis)
        print(f"使用正弦波运动策略，轴: {axis}")
        
    else:
        print("无效选择，使用默认简单策略")
        target_joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        policy = SimplePolicy(target_joints)
    
    # 创建策略接口
    try:
        interface = create_policy_interface(config_path, policy)
        
        print("\n启动策略接口...")
        with interface:
            print("策略接口已启动!")
            
            # 获取初始观测
            obs = interface.get_observation()
            print(f"初始位置: {obs['robot0_eef_pos']}")
            print(f"初始旋转: {obs['robot0_eef_rot_axis_angle']}")
            
            # 运行策略
            print("\n开始运行策略...")
            print("按 Ctrl+C 停止")
            
            interface.run_policy(
                max_steps=1000,  # 最多运行1000步
                step_callback=step_callback
            )
            
    except KeyboardInterrupt:
        print("\n用户中断，停止策略...")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("策略运行结束")


if __name__ == '__main__':
    main()
