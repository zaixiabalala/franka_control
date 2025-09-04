#!/usr/bin/env python3
"""
轨迹复现测试脚本
用于测试轨迹复现功能是否正常工作
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.replay_trajectory import TrajectoryPolicy


def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")
    
    # 创建简单的测试轨迹 (5秒，30Hz)
    duration = 5.0
    frequency = 30.0
    n_frames = int(duration * frequency)
    
    # 创建时间序列
    times = np.arange(n_frames) / frequency
    
    # 创建简单的关节轨迹 (7个关节)
    joints = np.zeros((n_frames, 7))
    for i in range(7):
        # 每个关节做简单的正弦运动
        joints[:, i] = 0.1 * np.sin(2 * np.pi * 0.2 * times + i * 0.5)
    
    print(f"测试数据: {n_frames} 帧, {duration} 秒")
    return times, joints


def test_trajectory_data():
    """测试轨迹数据"""
    print("\n测试轨迹数据...")
    
    # 创建测试数据
    times, joints = create_test_data()
    
    # 创建模拟的6维动作数据
    n_frames = len(joints)
    trajectory_data = np.zeros((n_frames, 6))
    trajectory_data[:, :3] = [0.5, 0.0, 0.3]  # 固定位置
    trajectory_data[:, 3:] = [0.0, 0.0, 0.0]  # 固定旋转
    
    print(f"轨迹数据测试: {n_frames} 帧")
    print(f"轨迹数据形状: {trajectory_data.shape}")
    
    # 检查轨迹数据
    assert trajectory_data.shape == (n_frames, 6), f"轨迹数据形状错误: {trajectory_data.shape}"
    print("轨迹数据测试通过!")


def test_policy():
    """测试策略"""
    print("\n测试策略...")
    
    # 创建测试数据
    times, joints = create_test_data()
    
    # 创建模拟的6维动作数据
    n_frames = len(joints)
    trajectory_data = np.zeros((n_frames, 6))
    trajectory_data[:, :3] = [0.5, 0.0, 0.3]  # 固定位置
    trajectory_data[:, 3:] = [0.0, 0.0, 0.0]  # 固定旋转
    
    # 创建策略
    policy = TrajectoryPolicy(
        trajectory_data=trajectory_data,
        start_time=0.0,
        data_frequency=30.0
    )
    
    # 测试策略
    dummy_obs = {
        'robot0_eef_pos': np.array([0.5, 0.0, 0.3]),
        'robot0_eef_rot_axis_angle': np.array([0.0, 0.0, 0.0])
    }
    
    action = policy(dummy_obs)
    print(f"策略测试: 动作形状 {action.shape}")
    assert action.shape == (6,), f"动作形状错误: {action.shape}"
    print("策略测试通过!")


def test_data_loading():
    """测试数据加载功能"""
    print("\n测试数据加载...")
    
    # 创建临时测试数据
    test_dir = Path("/tmp/test_angles")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # 创建测试文件
        for i in range(10):
            # 8维数据: 7个关节 + 1个夹爪
            data = np.random.rand(8) * 180  # 度数
            np.save(test_dir / f"angle_cam0_{i:04d}.npy", data)
        
        print(f"创建测试文件: {test_dir}")
        
        # 测试文件查找
        pat1 = sorted(test_dir.glob("angle_cam0_*.npy"))
        files = pat1 if len(pat1) > 0 else sorted(test_dir.glob("*.npy"))
        
        print(f"找到 {len(files)} 个文件")
        assert len(files) == 10, f"文件数量错误: {len(files)}"
        
        # 测试数据加载
        all_joints = []
        for f in files:
            arr = np.load(f)
            assert arr.shape[0] == 8, f"数据维度错误: {arr.shape}"
            
            joints_deg = np.array(arr[:7], dtype=np.float32)
            joints_rad = np.radians(joints_deg).astype(np.float32)
            all_joints.append(joints_rad)
        
        all_joints = np.array(all_joints)
        print(f"加载数据形状: {all_joints.shape}")
        assert all_joints.shape == (10, 7), f"数据形状错误: {all_joints.shape}"
        
        print("数据加载测试通过!")
        
    finally:
        # 清理测试文件
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)


def main():
    """主测试函数"""
    print("轨迹复现功能测试")
    print("=" * 50)
    
    try:
        test_trajectory_data()
        test_data_loading()
        test_policy()
        
        print("\n" + "=" * 50)
        print("所有测试通过! 轨迹复现功能正常")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
