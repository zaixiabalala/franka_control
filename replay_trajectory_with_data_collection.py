#!/usr/bin/env python3
"""
轨迹复现与数据采集脚本
在机械臂复现轨迹的同时，实时采集机械臂关节数据和相机图像
"""

import argparse
import time
import sys
import os
from pathlib import Path
import numpy as np
import cv2
import threading
import yaml
from queue import Queue
from multiprocessing.managers import SharedMemoryManager

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy_interface import create_policy_interface
from real_world.franka_interpolation_controller import FrankaInterpolationController
from r3kit.devgitices.camera.realsense import config as rs_cfg
from r3kit.devices.camera.realsense.d415 import D415
import pyrealsense2 as rs

# === 配置参数 ===
FPS = 30
INTERVAL = 1.0 / FPS
SAVE_EVERY_N_FRAMES = 1

# 设置相机配置
D415_CAMERAS = {
    "cam_0": "327322062498"
}

# RealSense配置
rs_cfg.D415_STREAMS = [
    (rs.stream.depth, 640, 480, rs.format.z16, FPS),
    (rs.stream.color, 640, 480, rs.format.bgr8, FPS),
]


class TrajectoryPolicy:
    """轨迹策略 - 直接使用预录制的关节轨迹数据模拟模型推理"""
    
    def __init__(self, 
                 trajectory_data: np.ndarray,
                 gripper_data: np.ndarray,
                 start_time: float,
                 data_frequency: float = 30.0,
                 use_step_based: bool = True):
        """
        初始化轨迹策略
        
        Args:
            trajectory_data: 轨迹数据 (N, 7) - 7维关节角度数据
            gripper_data: gripper数据 (N,) - gripper编码器数据
            start_time: 开始时间
            data_frequency: 数据频率 (Hz)
            use_step_based: 是否使用基于步数的索引（True=基于步数，False=基于时间）
        """
        self.trajectory_data = trajectory_data
        self.gripper_data = gripper_data
        self.start_time = start_time
        self.data_frequency = data_frequency
        self.dt = 1.0 / data_frequency
        self.use_step_based = use_step_based
        self.current_step = 0  # 当前步数
        
    def __call__(self, obs):
        """
        策略函数 - 模拟模型推理
        
        Args:
            obs: 观测字典 (这里不使用，只是为了接口兼容)
            
        Returns:
            action: 7维关节角度 [j1, j2, j3, j4, j5, j6, j7]
        """
        if self.use_step_based:
            # 【新模式】基于步数：每次调用递增步数
            data_index = self.current_step
            self.current_step += 1
        else:
            # 【旧模式】基于时间：根据实际经过的时间计算索引
            current_time = time.monotonic() - self.start_time
            data_index = int(current_time * self.data_frequency)
        
        # 确保索引不超出范围
        if data_index >= len(self.trajectory_data):
            data_index = len(self.trajectory_data) - 1
        
        # 直接返回对应的动作数据
        action = self.trajectory_data[data_index].copy()
        
        return action
    
    def get_gripper_action(self, obs):
        """获取gripper动作"""
        if self.use_step_based:
            # 基于步数：使用当前步数（已经在 __call__ 中递增了）
            data_index = self.current_step - 1  # 减1因为 __call__ 已经递增
        else:
            # 基于时间
            current_time = time.monotonic() - self.start_time
            data_index = int(current_time * self.data_frequency)
        
        if data_index >= len(self.gripper_data):
            data_index = len(self.gripper_data) - 1
        if data_index < 0:
            data_index = 0
        
        # 夹爪动作减去2cm，让夹爪关得更紧
        gripper_action = self.gripper_data[data_index] - 0.017  # 减去2cm
        return max(0.0, gripper_action)  # 确保不小于0
    
    def is_finished(self):
        """检查轨迹是否已执行完成"""
        return self.current_step >= len(self.trajectory_data)


class DataCollector:
    """数据采集器 - 从机械臂和相机采集数据"""
    
    def __init__(self, 
                 controller: FrankaInterpolationController,
                 gripper_controller,
                 cameras: dict,
                 save_dir: str,
                 frequency: float = 30.0):
        """
        初始化数据采集器
        
        Args:
            controller: Franka控制器
            gripper_controller: Gripper控制器
            cameras: 相机字典
            save_dir: 保存目录
            frequency: 采集频率 (Hz)
        """
        self.controller = controller
        self.gripper_controller = gripper_controller
        self.cameras = cameras
        self.save_dir = Path(save_dir)
        self.frequency = frequency
        self.interval = 1.0 / frequency
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for name in cameras.keys():
            cam_dir = self.save_dir / name
            cam_dir.mkdir(exist_ok=True)
            (cam_dir / "color").mkdir(exist_ok=True)
            (cam_dir / "depth").mkdir(exist_ok=True)
        
        # 数据存储
        self.frame_idx = 0
        self.timestamps = []
        self.running = False
        self.thread = None
        
    def start(self):
        """启动数据采集"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        print(f"数据采集已启动，频率: {self.frequency}Hz")
        
    def stop(self):
        """停止数据采集"""
        if not self.running:
            return
            
        self.running = False
        if self.thread:
            self.thread.join()
        print("数据采集已停止")
        
    def _collect_loop(self):
        """数据采集循环"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            self.timestamps.append(current_time)
            
            try:
                # 获取机械臂状态
                state = self.controller.get_state()
                
                # 提取关节数据 (7维关节角度，转换为度数)
                joint_pos = state['ActualQ']  # 弧度
                
                # 获取gripper数据
                gripper_width = 0.0  # 默认值
                if self.gripper_controller is not None:
                    try:
                        gripper_state = self.gripper_controller.get_state()
                        if gripper_state is not None and 'gripper_position' in gripper_state:
                            gripper_width = gripper_state['gripper_position']  # 单位：米
                    except Exception as e:
                        if self.frame_idx % 100 == 0:  # 只偶尔打印错误
                            print(f"获取gripper状态失败: {e}")
                
                angle_data = np.concatenate([joint_pos, [gripper_width]])
                
                # 保存关节数据
                angle_path = self.save_dir / f"angle_cam0_{self.frame_idx:05d}.npy"
                np.save(angle_path, angle_data)
                
                # 采集所有相机的图像
                for name, cam in self.cameras.items():
                    try:
                        color_frame, depth_frame = cam.get()
                        
                        if color_frame is not None and depth_frame is not None:
                            # 保存彩色图
                            color_path = self.save_dir / name / "color" / f"{self.frame_idx:016d}.jpg"
                            cv2.imwrite(str(color_path), color_frame)
                            
                            # 保存深度图
                            depth_path = self.save_dir / name / "depth" / f"{self.frame_idx:016d}.png"
                            cv2.imwrite(str(depth_path), depth_frame)
                            
                    except Exception as e:
                        print(f"相机 {name} 采集失败: {e}")
                
                # 控制采集频率
                elapsed = time.time() - last_time
                sleep_time = self.interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_time = time.time()
                
                # 每100帧打印一次进度
                if self.frame_idx % 100 == 0:
                    print(f"已采集帧数: {self.frame_idx}")
                
                self.frame_idx += 1
                
            except Exception as e:
                print(f"数据采集出错: {e}")
                time.sleep(0.01)  # 短暂等待后继续


class TrajectoryReplayerWithDataCollection:
    """带数据采集的轨迹复现器"""
    
    def __init__(self, 
                 config_path: str,
                 angles_dir: str,
                 save_dir: str,
                 policy_frequency: float = 20.0,
                 data_frequency: float = 30.0,
                 angles_unit: str = 'deg',
                 use_step_based: bool = True):
        """
        初始化轨迹复现器
        
        Args:
            config_path: 配置文件路径
            angles_dir: 轨迹数据目录
            save_dir: 数据保存目录
            policy_frequency: 策略推理频率 (Hz)
            data_frequency: 数据采集频率 (Hz)
            angles_unit: 输入轨迹关节角单位 ('deg' 或 'rad')
            use_step_based: 是否使用基于步数的索引（True=步数，False=时间）
        """
        self.config_path = config_path
        self.angles_dir = Path(angles_dir)
        self.save_dir = save_dir
        self.policy_frequency = policy_frequency
        self.data_frequency = data_frequency
        self.joint_dim = 7
        self.angles_unit = angles_unit  # 'deg' or 'rad'
        self.use_step_based = use_step_based
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载轨迹数据
        self.trajectory_data, self.gripper_data = self._load_joints_data()
        
        # 初始化相机
        self.cameras = {
            name: D415(id=serial, depth=True, name=name)
            for name, serial in D415_CAMERAS.items()
        }
        
    def _load_joints_data(self):
        """加载关节数据"""
        # 查找轨迹文件
        pat1 = sorted(self.angles_dir.glob("angle_cam0_*.npy"))
        files = pat1 if len(pat1) > 0 else sorted(self.angles_dir.glob("*.npy"))
        
        if len(files) == 0:
            raise FileNotFoundError(f"未在 {self.angles_dir} 找到角度 .npy 文件")
        
        print(f"找到 {len(files)} 个轨迹文件")
        
        # 加载关节数据和gripper数据
        all_joints = []
        all_grippers = []
        for i, f in enumerate(files):
            arr = np.load(f)
            if arr.shape[0] != 8:
                raise ValueError(f"{f} 维度异常，期望 8，得到 {arr.shape}")
            
            # 前7维是关节角度
            joints = np.array(arr[:self.joint_dim], dtype=np.float32)
            # 若输入为角度，则转换为弧度；若为弧度则直接使用
            if self.angles_unit == 'deg':
                joints = np.radians(joints).astype(np.float32)
            all_joints.append(joints)
            
            # 第8维是gripper编码器值
            gripper_encoder = float(arr[self.joint_dim])
            all_grippers.append(gripper_encoder)
        
        all_joints = np.array(all_joints)
        all_grippers = np.array(all_grippers)
        print(f"关节轨迹数据: {len(all_joints)} 帧")
        print(f"Gripper轨迹数据: {len(all_grippers)} 帧")
        print(f"数据频率: {self.data_frequency}Hz")
        print(f"输入角度单位: {self.angles_unit}")
        print(f"总时长: {len(all_joints) / self.data_frequency:.2f}s")
        
        return all_joints, all_grippers
    
    def run(self):
        """执行轨迹复现与数据采集"""
        print(f"开始轨迹复现与数据采集")
        print(f"策略推理频率: {self.policy_frequency}Hz")
        print(f"数据采集频率: {self.data_frequency}Hz")
        print(f"数据保存目录: {self.save_dir}")
        
        # 创建保存目录
        session_dir = Path(self.save_dir) / time.strftime("replay_record_%Y%m%d_%H%M%S", time.localtime())
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建策略接口
        interface = create_policy_interface(self.config_path)
        
        try:
            # 启动策略接口
            print("启动策略接口...")
            interface.start()
            print("策略接口已启动!")
            
            # 创建数据采集器
            data_collector = DataCollector(
                controller=interface.controller,
                gripper_controller=interface.gripper_controller,  # 传递 gripper 控制器
                cameras=self.cameras,
                save_dir=str(session_dir),
                frequency=self.data_frequency
            )
            
            # 创建轨迹策略
            start_time = time.monotonic()
            policy = TrajectoryPolicy(
                trajectory_data=self.trajectory_data,
                gripper_data=self.gripper_data,
                start_time=start_time,
                data_frequency=self.data_frequency,
                use_step_based=self.use_step_based
            )
            
            mode_str = "基于步数" if self.use_step_based else "基于时间"
            print(f"轨迹索引模式: {mode_str}")
            
            # 更新策略接口的策略
            interface.policy_fn = policy
            
            # 获取初始观测
            obs = interface.get_observation()
            print(f"初始关节位置: {obs['robot0_joint_pos']}")
            print(f"初始Gripper宽度: {obs['robot0_gripper_width']}")
            
            # 计算总步数和预计时间
            total_steps = len(self.trajectory_data)
            estimated_time = total_steps / self.policy_frequency
            print(f"总轨迹点数: {total_steps}")
            print(f"策略频率: {self.policy_frequency}Hz")
            print(f"预计执行时间: {estimated_time:.2f}s (实际时间会根据机械臂响应略有不同)")
            
            # 启动数据采集
            print("启动数据采集...")
            data_collector.start()
            
            # 运行策略
            print("开始轨迹复现...")
            print("按 Ctrl+C 停止")
            
            step = 0
            last_progress_time = time.monotonic()
            
            while True:
                current_time = time.monotonic() - start_time
                
                # 检查是否完成（基于轨迹点数）
                if policy.is_finished():
                    print("轨迹复现完成!")
                    break
                
                # 获取观测
                obs = interface.get_observation()
                current_pos = obs['robot0_joint_pos']
                
                # 执行策略 (模拟模型推理)
                action = policy(obs)
                
                # 当前数据索引（用于调试显示）
                data_index = policy.current_step - 1
                
                # 每10步打印一次详细信息
                if step % 10 == 0:
                    print(f"Step {step}: 时间={current_time:.2f}s, 轨迹索引={data_index}/{total_steps-1}")
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
                
                # 执行gripper动作
                try:
                    gripper_action = policy.get_gripper_action(obs)
                    interface.execute_gripper_action(gripper_action)
                except Exception as e:
                    print(f"执行gripper动作失败: {e}")
                    # 不抛出异常，继续执行
                
                # 进度显示
                if step % (self.policy_frequency * 2) == 0:  # 每2秒显示一次
                    progress = (policy.current_step / total_steps) * 100
                    print(f"进度: {progress:.1f}% ({policy.current_step}/{total_steps} 帧, 已用时 {current_time:.1f}s)")
                
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
            # 停止数据采集
            print("停止数据采集...")
            data_collector.stop()
            
            # 停止策略接口
            print("停止策略接口...")
            interface.stop()
            
            # 打印统计信息
            actual_duration = time.monotonic() - start_time
            if data_collector.timestamps:
                intervals = [(data_collector.timestamps[i + 1] - data_collector.timestamps[i]) * 1000 
                           for i in range(len(data_collector.timestamps) - 1)]
                errors = [abs(i - 1000 / self.data_frequency) for i in intervals]
                if errors:
                    print(f"\n数据采集统计:")
                    print(f"共采集帧数: {len(data_collector.timestamps)}")
                    print(f"平均帧间隔误差（目标 {1000 / self.data_frequency:.2f} ms）: {sum(errors)/len(errors):.2f} ms")
            
            print(f"\n轨迹复现统计:")
            print(f"执行轨迹点数: {policy.current_step}/{total_steps}")
            print(f"实际执行时间: {actual_duration:.2f}s")
            print(f"平均策略频率: {policy.current_step / actual_duration:.1f}Hz (目标 {self.policy_frequency}Hz)")
            
            print(f"\n数据保存至：{session_dir}")
            print("轨迹复现与数据采集结束")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="轨迹复现与数据采集脚本")
    parser.add_argument("--config", type=str, 
                        default="/home/robotflow/my_code/other_codes/franka_control/config/robot_config.yaml",
                        help="机器人配置文件路径")
    parser.add_argument("--angles_dir", type=str, 
                        default="/media/robotflow/Elements SE/compress/angles_010_record_20250917_144922/angles",
                        help="轨迹数据目录")
    parser.add_argument("--save_dir", type=str,
                        default="/home/robotflow/my_code/other_codes/franka_control/record_replay",
                        help="数据保存目录")
    parser.add_argument("--policy_frequency", type=float, default=17.0,
                        help="策略推理频率 (Hz)")
    parser.add_argument("--data_frequency", type=float, default=30.0,
                        help="数据采集频率 (Hz)")
    parser.add_argument("--angles_unit", type=str, choices=["deg", "rad"], default="rad",
                        help="输入轨迹关节角单位：deg 或 rad")
    parser.add_argument("--use_time_based", action="store_true",
                        help="使用基于时间的轨迹索引（默认使用基于步数）")
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    # 检查数据目录
    if not os.path.exists(args.angles_dir):
        print(f"错误: 数据目录不存在: {args.angles_dir}")
        return
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建轨迹复现器
    try:
        replayer = TrajectoryReplayerWithDataCollection(
            config_path=args.config,
            angles_dir=args.angles_dir,
            save_dir=args.save_dir,
            policy_frequency=args.policy_frequency,
            data_frequency=args.data_frequency,
            angles_unit=args.angles_unit,
            use_step_based=not args.use_time_based  # 默认基于步数，除非指定 --use_time_based
        )
        
        # 执行复现与采集
        replayer.run()
        
    except Exception as e:
        print(f"轨迹复现与数据采集失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
