"""
Franka Control Final - 基于UMI重构的Franka机器人控制系统

这个包提供了完整的Franka机器人控制功能，包括：
- 关节空间控制：直接控制7个关节角度
- Gripper控制：支持WSG gripper的编码器控制
- 共享内存系统：高效的进程间数据共享
- 轨迹复现：支持录制轨迹的精确复现
- 策略接口：统一的策略控制接口

主要组件：
- shared_memory: 共享内存系统
- common: 通用工具函数
- real_world: 真实世界控制接口
- policy_interface: 策略控制接口
"""

__version__ = "1.0.0"
__author__ = "Based on UMI (Universal Manipulation Interface)"

# 导入主要接口
from .policy_interface import PolicyInterface, SimplePolicy, create_policy_interface
from .real_world import FrankaInterface, FrankaInterpolationController, WSGController, Command
from .shared_memory import SharedNDArray, SharedMemoryRingBuffer, SharedMemoryQueue
from .common import JointTrajectoryInterpolator, convert_gripper_encoder_to_width

__all__ = [
    'PolicyInterface',
    'SimplePolicy', 
    'create_policy_interface',
    'FrankaInterface',
    'FrankaInterpolationController',
    'WSGController',
    'Command',
    'SharedNDArray',
    'SharedMemoryRingBuffer',
    'SharedMemoryQueue',
    'JointTrajectoryInterpolator',
    'convert_gripper_encoder_to_width'
]
