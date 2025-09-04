"""
Franka Control UMI - 基于UMI的Franka机器人控制系统

这个包提供了从UMI中提取的Franka机器人控制功能，包括：
- 共享内存系统：高效的进程间数据共享
- Franka控制接口：通过ZeroRPC与机器人通信
- 插值控制器：平滑的轨迹控制
- 策略接口：统一的策略控制接口
- 通用工具：姿态处理、插值、时间控制等

主要组件：
- shared_memory: 共享内存系统
- common: 通用工具函数
- real_world: 真实世界控制接口
- policy_interface: 策略控制接口
"""

__version__ = "1.0.0"
__author__ = "Extracted from UMI"

# 导入主要接口
from .policy_interface import PolicyInterface, SimplePolicy, create_policy_interface
from .real_world import FrankaInterface, FrankaInterpolationController, Command
from .shared_memory import SharedNDArray, SharedMemoryRingBuffer, SharedMemoryQueue

__all__ = [
    'PolicyInterface',
    'SimplePolicy', 
    'create_policy_interface',
    'FrankaInterface',
    'FrankaInterpolationController',
    'Command',
    'SharedNDArray',
    'SharedMemoryRingBuffer',
    'SharedMemoryQueue'
]
