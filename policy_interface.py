"""
策略接口模块
提供统一的策略控制接口，支持不同的策略实现

基于UMI (Universal Manipulation Interface) 项目重构：
- 从笛卡尔位姿控制改为关节空间控制
- 添加了gripper控制支持
- 优化了策略接口设计
"""

import time
import numpy as np
from typing import Dict, Any, Optional, Callable
from multiprocessing.managers import SharedMemoryManager

from real_world.franka_interpolation_controller import FrankaInterpolationController
from real_world.wsg_controller import WSGController
from shared_memory import SharedMemoryRingBuffer
from common.gripper_util import convert_gripper_encoder_to_width, limit_gripper_step


class PolicyInterface:
    """策略接口，提供统一的机器人控制接口"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 policy_fn: Optional[Callable] = None):
        """
        初始化策略接口
        
        Args:
            config: 配置字典
            policy_fn: 策略函数，接受观测返回动作
        """
        self.config = config
        self.policy_fn = policy_fn
        self.shm_manager = None
        self.controller = None
        self.gripper_controller = None
        self.is_running = False
        
    def start(self):
        """启动策略接口"""
        if self.is_running:
            return
            
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        
        # 创建Franka控制器
        robot_config = self.config['robot']
        shm_config = self.config['shared_memory']
        policy_config = self.config['policy']
        
        self.controller = FrankaInterpolationController(
            shm_manager=self.shm_manager,
            robot_ip=robot_config['ip'],
            robot_port=robot_config['port'],
            frequency=robot_config['frequency'],
            Kx_scale=robot_config['Kx_scale'],
            Kxd_scale=robot_config['Kxd_scale'],
            joints_init=robot_config['joints_init'],
            joints_init_duration=robot_config['joints_init_duration'],
            soft_real_time=robot_config['soft_real_time'],
            verbose=robot_config['verbose'],
            get_max_k=shm_config['get_max_k'],
            receive_latency=policy_config['obs_latency']
        )
        
        self.controller.start()
        self.controller.start_wait()
        
        # 创建Gripper控制器（如果配置了gripper）
        if 'gripper' in self.config:
            gripper_config = self.config['gripper']
            self.gripper_controller = WSGController(
                shm_manager=self.shm_manager,
                hostname=gripper_config['hostname'],
                port=gripper_config['port'],
                frequency=gripper_config['frequency'],
                move_max_speed=gripper_config['move_max_speed'],
                verbose=gripper_config['verbose']
            )
            self.gripper_controller.start()
            self.gripper_controller.start_wait()
        
        self.is_running = True
        
    def stop(self):
        """停止策略接口"""
        if not self.is_running:
            return
            
        if self.gripper_controller:
            self.gripper_controller.stop()
            self.gripper_controller = None
            
        if self.controller:
            self.controller.stop()
            self.controller = None
            
        if self.shm_manager:
            self.shm_manager.shutdown()
            self.shm_manager = None
            
        self.is_running = False
        
    def get_observation(self) -> Dict[str, np.ndarray]:
        """获取当前观测"""
        if not self.is_running:
            raise RuntimeError("策略接口未启动")
            
        state = self.controller.get_state()
        
        # 构建观测字典 - 主要使用关节信息
        obs = {
            'robot0_eef_pos': state['ActualTCPPose'][:3],  # 保留用于兼容性
            'robot0_eef_rot_axis_angle': state['ActualTCPPose'][3:],  # 保留用于兼容性
            'robot0_joint_pos': state['ActualQ'],  # 主要使用
            'robot0_joint_vel': state['ActualQd'],  # 主要使用
            'timestamp': state['robot_timestamp']
        }
        
        # 添加gripper状态
        obs['robot0_gripper_width'] = np.array([0.04])  # 默认值
        if self.gripper_controller is not None:
            try:
                gripper_state = self.gripper_controller.get_state()
                if gripper_state is not None:
                    # 使用正确的字段名
                    if 'gripper_position' in gripper_state:
                        obs['robot0_gripper_width'] = np.array([gripper_state['gripper_position']])
                    elif 'gripper_state' in gripper_state:
                        # 如果gripper_state是编码器值，需要转换
                        obs['robot0_gripper_encoder'] = np.array([gripper_state['gripper_state']])
            except Exception as e:
                print(f"获取gripper状态失败: {e}")
        
        return obs
        
    def execute_action(self, action: np.ndarray, target_time: Optional[float] = None):
        """
        执行动作
        
        Args:
            action: 动作数组，形状为(7,)，包含7个关节角度
            target_time: 目标时间，如果为None则使用当前时间+延迟
        """
        if not self.is_running:
            raise RuntimeError("策略接口未启动")
            
        if target_time is None:
            policy_config = self.config['policy']
            target_time = time.time() + policy_config['action_latency']
            
        # 确保动作形状正确
        action = np.array(action)
        if action.shape != (7,):
            raise ValueError(f"动作形状应为(7,)，实际为{action.shape}")
            
        self.controller.schedule_waypoint(action, target_time)
    
    def execute_gripper_action(self, gripper_encoder: float, target_time: Optional[float] = None):
        """
        执行gripper动作
        
        Args:
            gripper_encoder: gripper编码器值
            target_time: 目标时间，如果为None则使用当前时间+延迟
        """
        if not self.is_running:
            raise RuntimeError("策略接口未启动")
            
        if self.gripper_controller is None:
            return  # 如果没有gripper控制器，直接返回
            
        if target_time is None:
            policy_config = self.config['policy']
            target_time = time.time() + policy_config['action_latency']
            
        # 将编码器值转换为gripper宽度
        gripper_width = convert_gripper_encoder_to_width(gripper_encoder)
        
        # 发送gripper命令
        self.gripper_controller.schedule_waypoint(gripper_width, target_time)
        
    def run_policy(self, 
                   max_steps: Optional[int] = None,
                   step_callback: Optional[Callable] = None):
        """
        运行策略
        
        Args:
            max_steps: 最大步数，None表示无限运行
            step_callback: 每步回调函数
        """
        if not self.is_running:
            raise RuntimeError("策略接口未启动")
            
        if self.policy_fn is None:
            raise RuntimeError("未设置策略函数")
            
        step = 0
        try:
            while True:
                if max_steps is not None and step >= max_steps:
                    break
                    
                # 获取观测
                obs = self.get_observation()
                
                # 执行策略
                action = self.policy_fn(obs)
                
                # 执行动作
                self.execute_action(action)
                
                # 执行gripper动作（如果策略支持）
                if hasattr(self.policy_fn, 'get_gripper_action'):
                    gripper_action = self.policy_fn.get_gripper_action(obs)
                    self.execute_gripper_action(gripper_action)
                
                # 调用回调函数
                if step_callback:
                    step_callback(step, obs, action)
                    
                step += 1
                
                # 短暂等待
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("策略运行被用户中断")
        except Exception as e:
            print(f"策略运行出错: {e}")
            
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


class SimplePolicy:
    """简单策略示例 - 关节位置控制"""
    
    def __init__(self, target_joints: np.ndarray):
        """
        初始化简单策略
        
        Args:
            target_joints: 目标关节角度 [j1, j2, j3, j4, j5, j6, j7]
        """
        self.target_joints = np.array(target_joints)
        if self.target_joints.shape != (7,):
            raise ValueError(f"目标关节形状应为(7,)，实际为{self.target_joints.shape}")
        
    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """策略函数"""
        current_joints = obs['robot0_joint_pos']
        
        # 简单的PD控制
        error = self.target_joints - current_joints
        action = current_joints + 0.1 * error  # 简单的比例控制
        
        return action


def create_policy_interface(config_path: str, policy_fn: Optional[Callable] = None):
    """
    创建策略接口的便捷函数
    
    Args:
        config_path: 配置文件路径
        policy_fn: 策略函数
        
    Returns:
        PolicyInterface实例
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return PolicyInterface(config, policy_fn)
