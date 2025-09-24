#!/usr/bin/env python3
"""
关节动作突变分析脚本
专注于分析ACT模型推理过程中的关节位置突变
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse
import time
import json
from collections import deque
import threading

# 添加项目路径
project_dir = Path(__file__).parent.parent
model_lerobot_path = project_dir / "model" / "lerobot" / "src"
sys.path.insert(0, str(model_lerobot_path))
sys.path.insert(0, str(project_dir))

# 导入必要的模块
from common.gripper_util import convert_gripper_width_to_encoder
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.constants import OBS_IMAGES, ACTION, OBS_STATE
from policy_interface import create_policy_interface
from common.precise_sleep import precise_wait

# 相机相关导入
import pyrealsense2 as rs
from r3kit.devices.camera.realsense import config as rs_cfg
from r3kit.devices.camera.realsense.d415 import D415

class ActionDataCollector:
    """动作数据收集器"""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        
        # 动作数据存储
        self.timestamps = deque(maxlen=max_history)
        self.joint_actions = deque(maxlen=max_history)  # 7维关节动作
        self.gripper_actions = deque(maxlen=max_history)  # 1维夹爪动作
        self.full_actions = deque(maxlen=max_history)  # 8维完整动作
        
        # 推理性能数据
        self.inference_times = deque(maxlen=max_history)
        self.action_change_rates = deque(maxlen=max_history)
        
        # 突变检测数据
        self.mutation_threshold = 0.1  # 突变阈值 (rad)
        self.mutations = []  # 存储突变信息
        
        # 线程安全
        self.lock = threading.Lock()
        
    def add_action(self, action, inference_time, action_type="normal"):
        """添加动作数据"""
        with self.lock:
            current_time = time.monotonic()
            
            # 存储数据
            self.timestamps.append(current_time)
            self.joint_actions.append(action[:7].copy())
            self.gripper_actions.append(action[7])
            self.full_actions.append(action.copy())
            self.inference_times.append(inference_time)
            
            # 计算动作变化率和检测突变
            if len(self.joint_actions) > 1:
                prev_joint = self.joint_actions[-2]
                curr_joint = self.joint_actions[-1]
                change_rate = np.linalg.norm(curr_joint - prev_joint)
                self.action_change_rates.append(change_rate)
                
                # 检测突变
                if change_rate > self.mutation_threshold:
                    mutation_info = {
                        'step': len(self.joint_actions) - 1,
                        'timestamp': current_time,
                        'change_rate': float(change_rate),
                        'prev_joint_pos': prev_joint.tolist(),
                        'curr_joint_pos': curr_joint.tolist(),
                        'joint_changes': (curr_joint - prev_joint).tolist(),
                        'inference_time': float(inference_time),
                        'action_type': action_type
                    }
                    self.mutations.append(mutation_info)
                    
                    # 打印突变信息
                    print(f"🚨 检测到关节突变!")
                    print(f"  步骤: {mutation_info['step']}")
                    print(f"  时间: {current_time:.3f}s")
                    print(f"  变化率: {change_rate:.6f} rad")
                    print(f"  前一步关节位置: {prev_joint}")
                    print(f"  当前关节位置: {curr_joint}")
                    print(f"  关节变化: {curr_joint - prev_joint}")
                    print(f"  推理时间: {inference_time:.3f}s")
                    print(f"  动作类型: {action_type}")
                    print("-" * 50)
            else:
                self.action_change_rates.append(0.0)
    
    def get_data(self):
        """获取当前数据（线程安全）"""
        with self.lock:
            return {
                'timestamps': list(self.timestamps),
                'joint_actions': np.array(list(self.joint_actions)),
                'gripper_actions': list(self.gripper_actions),
                'full_actions': np.array(list(self.full_actions)),
                'inference_times': list(self.inference_times),
                'action_change_rates': list(self.action_change_rates),
                'mutations': self.mutations.copy()
            }
    
    def analyze_mutations(self):
        """分析突变模式"""
        if not self.mutations:
            print("未检测到任何突变")
            return
        
        print(f"\n=== 突变分析报告 ===")
        print(f"总突变次数: {len(self.mutations)}")
        
        # 按推理时间分析突变
        inference_time_mutations = [m for m in self.mutations if m['action_type'] == 'normal']
        fallback_mutations = [m for m in self.mutations if m['action_type'] == 'fallback']
        
        print(f"正常推理中的突变: {len(inference_time_mutations)}")
        print(f"降级策略中的突变: {len(fallback_mutations)}")
        
        if inference_time_mutations:
            print(f"\n--- 正常推理突变分析 ---")
            for i, mutation in enumerate(inference_time_mutations):
                print(f"突变 {i+1}:")
                print(f"  步骤: {mutation['step']}")
                print(f"  变化率: {mutation['change_rate']:.6f} rad")
                print(f"  推理时间: {mutation['inference_time']:.3f}s")
                print(f"  主要变化关节: {self._find_max_change_joint(mutation['joint_changes'])}")
        
        # 分析突变的时间模式
        self._analyze_mutation_timing()
    
    def _find_max_change_joint(self, joint_changes):
        """找到变化最大的关节"""
        joint_changes = np.array(joint_changes)
        max_joint_idx = np.argmax(np.abs(joint_changes))
        return f"关节{max_joint_idx+1} (变化: {joint_changes[max_joint_idx]:.4f} rad)"
    
    def _analyze_mutation_timing(self):
        """分析突变的时间模式"""
        if len(self.mutations) < 2:
            return
        
        print(f"\n--- 突变时间模式分析 ---")
        
        # 分析突变间隔
        intervals = []
        for i in range(1, len(self.mutations)):
            interval = self.mutations[i]['step'] - self.mutations[i-1]['step']
            intervals.append(interval)
        
        if intervals:
            print(f"突变间隔统计:")
            print(f"  平均间隔: {np.mean(intervals):.1f} 步")
            print(f"  最小间隔: {np.min(intervals)} 步")
            print(f"  最大间隔: {np.max(intervals)} 步")
            
            # 检查是否有规律的间隔（如25步间隔）
            interval_counts = {}
            for interval in intervals:
                interval_counts[interval] = interval_counts.get(interval, 0) + 1
            
            print(f"  间隔分布: {dict(sorted(interval_counts.items()))}")
            
            # 特别检查25步间隔
            if 25 in interval_counts:
                print(f"  ⚠️ 发现25步间隔的突变: {interval_counts[25]} 次")
                print(f"     这可能与ACT模型的n_action_steps=25有关！")
    
    def save_data(self, filepath):
        """保存数据到文件"""
        data = self.get_data()
        # 转换numpy数组为列表以便JSON序列化
        for key in ['joint_actions', 'full_actions']:
            if key in data:
                data[key] = data[key].tolist()
        
        # 转换所有numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            else:
                return obj
        
        data = convert_numpy_types(data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"动作数据已保存到: {filepath}")


class ActionAnalyzer:
    """动作分析器"""
    
    def __init__(self, data_collector, save_dir=None):
        self.data_collector = data_collector
        self.save_dir = save_dir or Path.cwd()
        
        # 执行分析
        self.run_analysis()
        
    def run_analysis(self):
        """运行分析"""
        print("=== 开始动作分析 ===")
        
        # 分析突变
        self.data_collector.analyze_mutations()
        
        # 生成统计报告
        self.generate_statistics_report()
        
        print("=== 分析完成 ===")
    
    def generate_statistics_report(self):
        """生成统计报告"""
        data = self.data_collector.get_data()
        
        if len(data['timestamps']) == 0:
            print("没有数据可分析")
            return
        
        # 生成时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 生成文本报告
        self.save_statistics_report(data, timestamp)
    
    def save_statistics_report(self, data, timestamp):
        """保存统计报告"""
        timestamps = np.array(data['timestamps'])
        joint_actions = data['joint_actions']
        change_rates = data['action_change_rates']
        inference_times = data['inference_times']
        mutations = data['mutations']
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("关节动作突变分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 基本信息
        report_lines.append("【基本信息】")
        report_lines.append(f"总步数: {len(data['timestamps'])}")
        if len(timestamps) > 0:
            total_time = timestamps[-1] - timestamps[0]
            report_lines.append(f"总运行时间: {total_time:.2f} 秒")
            report_lines.append(f"平均频率: {len(timestamps)/total_time:.2f} Hz")
        report_lines.append("")
        
        # 突变分析
        report_lines.append("【突变分析】")
        report_lines.append(f"检测到的突变总数: {len(mutations)}")
        report_lines.append(f"突变阈值: {self.data_collector.mutation_threshold} rad")
        report_lines.append("")
        
        if mutations:
            report_lines.append("详细突变信息:")
            for i, mutation in enumerate(mutations):
                report_lines.append(f"\n突变 {i+1}:")
                report_lines.append(f"  步骤: {mutation['step']}")
                report_lines.append(f"  时间戳: {mutation['timestamp']:.3f}")
                report_lines.append(f"  变化率: {mutation['change_rate']:.6f} rad")
                report_lines.append(f"  推理时间: {mutation['inference_time']:.3f}s")
                report_lines.append(f"  动作类型: {mutation['action_type']}")
                report_lines.append(f"  前一步关节位置: {mutation['prev_joint_pos']}")
                report_lines.append(f"  当前关节位置: {mutation['curr_joint_pos']}")
                report_lines.append(f"  关节变化: {mutation['joint_changes']}")
                
                # 找出变化最大的关节
                joint_changes = np.array(mutation['joint_changes'])
                max_joint_idx = np.argmax(np.abs(joint_changes))
                report_lines.append(f"  主要变化关节: 关节{max_joint_idx+1} (变化: {joint_changes[max_joint_idx]:.4f} rad)")
        else:
            report_lines.append("未检测到任何突变")
        
        report_lines.append("")
        
        # 推理性能分析
        report_lines.append("【推理性能分析】")
        if len(inference_times) > 0:
            report_lines.append(f"平均推理时间: {np.mean(inference_times):.4f} 秒")
            report_lines.append(f"最大推理时间: {np.max(inference_times):.4f} 秒")
            report_lines.append(f"最小推理时间: {np.min(inference_times):.4f} 秒")
            report_lines.append(f"推理时间标准差: {np.std(inference_times):.4f} 秒")
            
            timeout_count = sum(1 for t in inference_times if t > 0.18)
            report_lines.append(f"超时次数 (>0.18s): {timeout_count}")
            report_lines.append(f"超时比例: {timeout_count/len(inference_times)*100:.1f}%")
        report_lines.append("")
        
        # 动作质量分析
        report_lines.append("【动作质量分析】")
        if len(change_rates) > 0:
            report_lines.append(f"平均动作变化率: {np.mean(change_rates):.6f} rad")
            report_lines.append(f"最大动作变化率: {np.max(change_rates):.6f} rad")
            report_lines.append(f"动作变化率标准差: {np.std(change_rates):.6f} rad")
            
            high_change_count = sum(1 for rate in change_rates if rate > self.data_collector.mutation_threshold)
            report_lines.append(f"高变化率次数 (>{self.data_collector.mutation_threshold} rad): {high_change_count}")
            report_lines.append(f"高变化率比例: {high_change_count/len(change_rates)*100:.1f}%")
        report_lines.append("")
        
        # 25步间隔分析
        if len(mutations) > 1:
            report_lines.append("【25步间隔分析】")
            intervals = []
            for i in range(1, len(mutations)):
                interval = mutations[i]['step'] - mutations[i-1]['step']
                intervals.append(interval)
            
            interval_counts = {}
            for interval in intervals:
                interval_counts[interval] = interval_counts.get(interval, 0) + 1
            
            report_lines.append(f"突变间隔分布: {dict(sorted(interval_counts.items()))}")
            
            if 25 in interval_counts:
                report_lines.append(f"⚠️ 发现{interval_counts[25]}次25步间隔的突变！")
                report_lines.append("     这强烈暗示突变与ACT模型的n_action_steps=25有关")
                report_lines.append("     建议检查模型在每次推理25步后的行为")
            else:
                report_lines.append("未发现明显的25步间隔模式")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # 保存文本报告
        report_text = "\n".join(report_lines)
        report_path = self.save_dir / f"mutation_analysis_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"突变分析报告已保存到: {report_path}")
        
        # 同时打印到控制台
        print("\n" + report_text)


class DebugActionInferenceRunner:
    """带动作可视化的推理运行器"""
    
    def __init__(self, model_path, config_path, device="cuda", max_steps=1000, 
                 test_mode=False, frequency=10.0, use_eih=True):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.max_steps = max_steps
        self.test_mode = test_mode
        self.frequency = frequency
        self.use_eih = use_eih
        
        # 创建数据收集器
        self.data_collector = ActionDataCollector()
        
        # 初始化相机系统（复用原有代码）
        self.camera_system = self._init_camera_system()
        
        # 初始化ACT策略（复用原有代码）
        self.policy = self._init_policy()
        
        print(f"调试推理运行器初始化完成")
        print(f"数据收集器已启动，最大历史记录: {self.data_collector.max_history}")
    
    def _init_camera_system(self):
        """初始化相机系统（复用原有代码）"""
        # 这里复用原有的CameraSystem类
        from scripts.inference_with_policy_interface_updated import CameraSystem
        return CameraSystem()
    
    def _init_policy(self):
        """初始化策略（复用原有代码）"""
        # 这里复用原有的ACTPolicyWrapper类
        from scripts.inference_with_policy_interface_updated import ACTPolicyWrapper
        return ACTPolicyWrapper(
            model_path=self.model_path,
            device=self.device,
            camera_system=self.camera_system,
            debug_image=False,
            use_eih=self.use_eih
        )
    
    def run_with_visualization(self):
        """运行推理并实时可视化"""
        print("启动动作突变分析...")
        print("注意：分析将在数据收集完成后进行")
        
        try:
            if self.test_mode:
                print("使用测试模式进行动作分析")
                self._run_test_mode_with_collection()
            else:
                print("使用实时推理模式进行动作分析")
                self._run_real_time_mode_with_collection()
        except KeyboardInterrupt:
            print("\n用户中断，停止动作分析...")
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 保存数据到脚本所在目录
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            script_dir = Path(__file__).parent
            save_path = script_dir / f"action_debug_data_{timestamp}.json"
            self.data_collector.save_data(str(save_path))
            
            # 启动分析
            print("启动突变分析...")
            analyzer = ActionAnalyzer(self.data_collector, save_dir=script_dir)
            
            # 清理资源
            self.cleanup()
    
    def _run_test_mode_with_collection(self):
        """测试模式：收集动作数据"""
        print("开始测试推理并收集动作数据...")
        for i in range(100):  # 运行100步进行测试
            print(f"\n=== 测试推理 {i + 1} ===")
            
            # 模拟观测数据
            obs = {
                'robot0_joint_pos': np.random.uniform(-1, 1, 7),
                'robot0_joint_vel': np.random.uniform(-0.1, 0.1, 7),
                'robot0_eef_pos': np.random.uniform(0.3, 0.7, 3),
                'robot0_eef_rot_axis_angle': np.random.uniform(-1, 1, 3),
                'robot0_gripper_width': np.random.uniform(0.0, 0.08, 1),
                'timestamp': time.monotonic()
            }
            
            # 执行策略并测量推理时间
            t_start = time.monotonic()
            cur_action = self.policy(obs)
            t_end = time.monotonic()
            inference_time = t_end - t_start
            
            # 收集动作数据
            self.data_collector.add_action(cur_action, inference_time)
            
            print(f"推理时间: {inference_time:.3f}s")
            print(f"动作: {cur_action}")
            
            time.sleep(0.1)  # 100ms间隔
    
    def _run_real_time_mode_with_collection(self):
        """实时推理模式：收集动作数据"""
        try:
            # 创建策略接口
            interface = create_policy_interface(self.config_path, self.policy)
            
            print("启动策略接口...")
            interface.start()
            print("策略接口已启动!")
            
            # 获取初始观测
            obs = interface.get_observation()
            print(f"初始关节位置: {obs['robot0_joint_pos']}")
            
            # 运行策略并收集数据
            print(f"\n开始运行策略并收集动作数据...")
            print(f"推理频率: {self.frequency} Hz")
            print("按 Ctrl+C 停止")
            
            dt = 1.0 / self.frequency
            t_start = time.monotonic()
            step = 0
            
            # 超时降级策略相关变量
            last_joint_action = None
            last_gripper_action = None
            inference_times = []
            max_inference_time = 0.18  # 最大允许推理时间 (180ms)
            timeout_count = 0
            
            while True:
                if self.max_steps is not None and step >= self.max_steps:
                    print(f"达到最大步数 {self.max_steps}，停止运行")
                    break
                
                # 计算当前周期结束时间
                t_cycle_end = t_start + (step + 1) * dt
                t_cycle_start = time.monotonic()
                
                # 获取观测
                obs = interface.get_observation()
                
                # 执行策略 - 添加超时检查
                t_inference_start = time.monotonic()
                try:
                    cur_action = self.policy(obs)
                    joint_action = cur_action[:self.policy.joint_dim]
                    gripper_action = cur_action[self.policy.joint_dim]
                    t_inference_end = time.monotonic()
                    inference_time = t_inference_end - t_inference_start
                    inference_times.append(inference_time)
                    
                    # 更新最后有效的动作
                    last_joint_action = joint_action.copy()
                    last_gripper_action = gripper_action.copy()
                    timeout_count = 0
                    
                except Exception as e:
                    print(f"推理失败: {e}")
                    t_inference_end = time.monotonic()
                    inference_time = t_inference_end - t_inference_start
                    inference_times.append(inference_time)
                    timeout_count += 1
                
                # 检查是否超时
                current_time = time.monotonic()
                elapsed_time = current_time - t_cycle_start
                remaining_time = t_cycle_end - current_time
                
                # 如果推理时间过长或剩余时间不足，使用降级策略
                if (inference_time > max_inference_time or 
                    remaining_time < 0.01):  # 剩余时间少于10ms
                    
                    if last_joint_action is not None and last_gripper_action is not None:
                        # 使用上次的有效动作
                        joint_action = last_joint_action
                        gripper_action = last_gripper_action
                        print(f"⚠️  使用降级策略: 推理时间={inference_time:.3f}s, 剩余时间={remaining_time:.3f}s")
                        
                        # 记录降级策略的使用情况
                        fallback_action = np.concatenate([joint_action, [gripper_action]])
                        self.data_collector.add_action(fallback_action, inference_time, "fallback")
                        
                    else:
                        # 不要使用当前位置，而是跳过这次执行
                        joint_action = obs['robot0_joint_pos'] + np.random.normal(0, 0.001, 7)
                        gripper_action = 0.04  # 默认夹爪宽度
                        print(f"⚠️  使用随机扰动动作，等待有效推理: 推理时间={inference_time:.3f}s")
                        
                        # 记录随机扰动动作
                        random_action = np.concatenate([joint_action, [gripper_action]])
                        self.data_collector.add_action(random_action, inference_time, "random")
                        continue  # 跳过这次循环
                else:
                    # 正常情况，收集原始动作数据
                    cur_action = np.concatenate([joint_action, [gripper_action]])
                    self.data_collector.add_action(cur_action, inference_time, "normal")
                
                # 执行动作（在实际机器人上）
                interface.execute_action(joint_action)
                interface.execute_gripper_action(gripper_action)
                
                # 每10步打印一次详细信息
                if step % 10 == 0:
                    current_time = time.monotonic() - t_start
                    avg_inference_time = np.mean(inference_times[-10:]) if len(inference_times) >= 10 else np.mean(inference_times)
                    print(f"Step {step}: 时间={current_time:.2f}s, 推理时间={inference_time:.3f}s (平均={avg_inference_time:.3f}s)")
                    print(f"  关节动作: {joint_action}")
                    print(f"  Gripper动作: {gripper_action}")
                    if timeout_count > 0:
                        print(f"  超时次数: {timeout_count}")
                
                step += 1
                
                # 等待到下一个周期
                precise_wait(t_cycle_end)
                
        except KeyboardInterrupt:
            print("\n用户中断，停止策略...")
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 停止策略接口
            if 'interface' in locals():
                print("停止策略接口...")
                interface.stop()
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'camera_system'):
            self.camera_system.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="关节动作突变分析脚本")
    parser.add_argument("--model_path", type=str, 
                       default="/home/robotflow/Downloads/060000/pretrained_model",
                       help="训练好的模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备 (cpu/cuda)")
    parser.add_argument("--config_path", type=str,
                       default="/home/robotflow/my_code/other_codes/franka_control-main/config/robot_config.yaml",
                       help="机器人配置文件路径")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="最大运行步数")
    parser.add_argument("--test_mode", action="store_true", default=False,
                       help="测试模式（不连接真实机器人）")
    parser.add_argument("--frequency", type=float, default=10.0,
                       help="推理频率 (Hz)")
    parser.add_argument("--use_eih", action="store_true", default=True,
                       help="使用eye-in-hand视角作为输入")
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config_path):
        print(f"错误: 配置文件不存在: {args.config_path}")
        return 1
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return 1
    
    # 创建并运行调试推理运行器
    try:
        runner = DebugActionInferenceRunner(
            model_path=args.model_path,
            config_path=args.config_path,
            device=args.device,
            max_steps=args.max_steps,
            test_mode=args.test_mode,
            frequency=args.frequency,
            use_eih=args.use_eih
        )
        
        # 运行推理并分析
        runner.run_with_visualization()
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("关节动作突变分析完成")
    return 0


if __name__ == "__main__":
    exit(main())