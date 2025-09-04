#!/usr/bin/env python3
"""
Franka机器人控制脚本
演示如何使用FrankaInterpolationController控制机器人
"""

import sys
import os
import time
import numpy as np
import yaml
from multiprocessing.managers import SharedMemoryManager

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_world.franka_interpolation_controller import FrankaInterpolationController
from common.precise_sleep import precise_wait

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    """主函数"""
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'robot_config.yaml')
    config = load_config(config_path)
    
    robot_config = config['robot']
    shm_config = config['shared_memory']
    
    print("Franka机器人控制演示")
    print(f"机器人IP: {robot_config['ip']}")
    print(f"控制频率: {robot_config['frequency']} Hz")
    
    with SharedMemoryManager() as shm_manager:
        # 创建Franka控制器
        controller = FrankaInterpolationController(
            shm_manager=shm_manager,
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
            receive_latency=config['policy']['obs_latency']
        )
        
        try:
            # 启动控制器
            print("启动Franka控制器...")
            controller.start()
            
            # 等待控制器就绪
            print("等待控制器就绪...")
            controller.start_wait()
            print("控制器已就绪!")
            
            # 获取当前状态
            state = controller.get_state()
            current_pose = state['ActualTCPPose']
            print(f"当前末端执行器位置: {current_pose[:3]}")
            print(f"当前末端执行器旋转: {current_pose[3:]}")
            
            # 演示控制
            print("\n开始控制演示...")
            
            # 示例1: 使用servoL命令
            print("示例1: 使用servoL命令移动机器人")
            target_pose = current_pose.copy()
            target_pose[2] += 0.05  # 向上移动5cm
            controller.servoL(target_pose, duration=2.0)
            time.sleep(3)
            
            # 示例2: 使用schedule_waypoint命令
            print("示例2: 使用schedule_waypoint命令")
            target_pose[2] -= 0.05  # 回到原位置
            target_time = time.time() + 2.0
            controller.schedule_waypoint(target_pose, target_time)
            time.sleep(3)
            
            # 示例3: 连续控制
            print("示例3: 连续控制演示")
            for i in range(5):
                target_pose = current_pose.copy()
                target_pose[0] += 0.02 * np.sin(i * 0.5)  # 正弦运动
                target_pose[1] += 0.02 * np.cos(i * 0.5)
                target_time = time.time() + 0.5
                controller.schedule_waypoint(target_pose, target_time)
                time.sleep(0.6)
            
            print("控制演示完成!")
            
        except KeyboardInterrupt:
            print("\n用户中断，停止控制...")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            # 停止控制器
            print("停止控制器...")
            controller.stop()
            print("控制器已停止")

if __name__ == '__main__':
    main()
