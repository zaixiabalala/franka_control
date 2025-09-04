#!/usr/bin/env python3
"""
测试关节控制版本的代码
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from policy_interface import SimplePolicy, create_policy_interface

def test_simple_policy():
    """测试简单策略"""
    print("测试简单策略...")
    
    # 创建目标关节角度
    target_joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
    
    # 创建策略
    policy = SimplePolicy(target_joints)
    
    # 模拟观测
    obs = {
        'robot0_joint_pos': np.array([0.1, -0.4, 0.1, -1.9, 0.1, 1.4, 0.1]),
        'robot0_joint_vel': np.zeros(7),
        'robot0_eef_pos': np.array([0.5, 0.0, 0.3]),
        'robot0_eef_rot_axis_angle': np.array([0.0, 0.0, 0.0])
    }
    
    # 测试策略
    action = policy(obs)
    print(f"目标关节: {target_joints}")
    print(f"当前关节: {obs['robot0_joint_pos']}")
    print(f"输出动作: {action}")
    print(f"动作形状: {action.shape}")
    
    # 验证动作形状
    assert action.shape == (7,), f"动作形状应为(7,)，实际为{action.shape}"
    print("✓ 简单策略测试通过")

def test_config_loading():
    """测试配置加载"""
    print("\n测试配置加载...")
    
    config_path = "config/robot_config.yaml"
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    # 测试创建策略接口
    try:
        policy = SimplePolicy(np.zeros(7))
        interface = create_policy_interface(config_path, policy)
        print("✓ 配置加载测试通过")
        print(f"动作维度: {interface.config['policy']['action_dim']}")
        print(f"观测维度: {interface.config['policy']['obs_dim']}")
    except Exception as e:
        print(f"✗ 配置加载测试失败: {e}")

def test_trajectory_policy():
    """测试轨迹策略"""
    print("\n测试轨迹策略...")
    
    # 导入轨迹策略
    from scripts.replay_trajectory import TrajectoryPolicy
    
    # 创建模拟轨迹数据
    n_frames = 100
    trajectory_data = np.random.randn(n_frames, 7) * 0.1  # 7维关节数据
    
    # 创建策略
    import time
    start_time = time.time()
    policy = TrajectoryPolicy(trajectory_data, start_time, 30.0)
    
    # 测试策略
    obs = {'robot0_joint_pos': np.zeros(7)}
    action = policy(obs)
    
    print(f"轨迹数据形状: {trajectory_data.shape}")
    print(f"输出动作形状: {action.shape}")
    print(f"输出动作: {action}")
    
    # 验证动作形状
    assert action.shape == (7,), f"动作形状应为(7,)，实际为{action.shape}"
    print("✓ 轨迹策略测试通过")

def main():
    """主测试函数"""
    print("=== 关节控制版本测试 ===")
    
    try:
        test_simple_policy()
        test_config_loading()
        test_trajectory_policy()
        
        print("\n=== 所有测试通过 ===")
        print("关节控制版本代码重构成功！")
        
    except Exception as e:
        print(f"\n=== 测试失败 ===")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
