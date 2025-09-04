#!/usr/bin/env python3
"""
测试关节轨迹插值器
"""

import numpy as np
import time
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.joint_trajectory_interpolator import JointTrajectoryInterpolator

def test_basic_interpolation():
    """测试基本插值功能"""
    print("=== 测试基本插值功能 ===")
    
    # 创建测试数据
    times = np.array([0.0, 1.0, 2.0, 3.0])
    joints = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 初始位置
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 1秒后
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2秒后
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # 3秒后
    ])
    
    # 创建插值器
    interp = JointTrajectoryInterpolator(times, joints)
    
    # 测试插值
    test_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    interpolated_joints = interp(test_times)
    
    print(f"原始时间点: {times}")
    print(f"测试时间点: {test_times}")
    print(f"插值结果形状: {interpolated_joints.shape}")
    
    # 验证边界条件
    assert np.allclose(interpolated_joints[0], joints[0]), "起始点不匹配"
    assert np.allclose(interpolated_joints[-1], joints[-1]), "结束点不匹配"
    
    print("✓ 基本插值功能测试通过")

def test_single_step():
    """测试单步插值"""
    print("\n=== 测试单步插值 ===")
    
    # 创建单步数据
    times = np.array([0.0])
    joints = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    
    interp = JointTrajectoryInterpolator(times, joints)
    
    # 测试插值
    test_times = np.array([0.0, 0.5, 1.0, 2.0])
    interpolated_joints = interp(test_times)
    
    print(f"单步关节位置: {joints[0]}")
    print(f"插值结果: {interpolated_joints}")
    
    # 验证所有结果都相同
    for i in range(len(interpolated_joints)):
        assert np.allclose(interpolated_joints[i], joints[0]), f"第{i}个插值结果不匹配"
    
    print("✓ 单步插值测试通过")

def test_drive_to_waypoint():
    """测试驱动到路径点功能"""
    print("\n=== 测试驱动到路径点功能 ===")
    
    # 创建初始插值器
    times = np.array([0.0])
    joints = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    interp = JointTrajectoryInterpolator(times, joints)
    
    # 驱动到新位置
    target_joints = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    new_interp = interp.drive_to_waypoint(
        joints=target_joints,
        time=2.0,
        curr_time=0.0,
        max_joint_speed=1.0
    )
    
    print(f"目标关节位置: {target_joints}")
    print(f"新插值器时间点: {new_interp.times}")
    print(f"新插值器关节位置: {new_interp.joints}")
    
    # 验证结果
    assert len(new_interp.times) == 2, "应该有2个时间点"
    assert new_interp.times[0] == 0.0, "起始时间应该是0.0"
    assert new_interp.times[1] >= 2.0, "结束时间应该至少是2.0"
    
    # 验证插值结果
    result = new_interp(2.0)
    assert np.allclose(result, target_joints), "最终位置应该等于目标位置"
    
    print("✓ 驱动到路径点功能测试通过")

def test_schedule_waypoint():
    """测试调度路径点功能"""
    print("\n=== 测试调度路径点功能 ===")
    
    # 创建初始插值器
    times = np.array([0.0, 1.0])
    joints = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    interp = JointTrajectoryInterpolator(times, joints)
    
    # 调度新路径点
    target_joints = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    new_interp = interp.schedule_waypoint(
        joints=target_joints,
        time=3.0,
        max_joint_speed=1.0,
        curr_time=0.5,
        last_waypoint_time=1.0
    )
    
    print(f"目标关节位置: {target_joints}")
    print(f"新插值器时间点: {new_interp.times}")
    print(f"新插值器关节位置形状: {new_interp.joints.shape}")
    
    # 验证结果
    assert len(new_interp.times) >= 2, "应该有至少2个时间点"
    assert new_interp.times[-1] >= 3.0, "最后时间点应该至少是3.0"
    
    # 验证插值结果
    result = new_interp(3.0)
    assert np.allclose(result, target_joints), "最终位置应该等于目标位置"
    
    print("✓ 调度路径点功能测试通过")

def test_trim():
    """测试修剪功能"""
    print("\n=== 测试修剪功能 ===")
    
    # 创建测试数据
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    joints = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    
    interp = JointTrajectoryInterpolator(times, joints)
    
    # 修剪到指定范围
    trimmed_interp = interp.trim(1.5, 3.5)
    
    print(f"原始时间点: {times}")
    print(f"修剪后时间点: {trimmed_interp.times}")
    print(f"修剪后关节位置形状: {trimmed_interp.joints.shape}")
    
    # 验证结果
    assert trimmed_interp.times[0] == 1.5, "起始时间应该是1.5"
    assert trimmed_interp.times[-1] == 3.5, "结束时间应该是3.5"
    
    print("✓ 修剪功能测试通过")

def main():
    """主测试函数"""
    print("=== 关节轨迹插值器测试 ===")
    
    try:
        test_basic_interpolation()
        test_single_step()
        test_drive_to_waypoint()
        test_schedule_waypoint()
        test_trim()
        
        print("\n=== 所有测试通过 ===")
        print("关节轨迹插值器实现成功！")
        
    except Exception as e:
        print(f"\n=== 测试失败 ===")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
