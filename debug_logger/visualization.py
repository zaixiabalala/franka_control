#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化调试工具

提供实时和离线可视化功能，用于分析policy推理过程
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime


class DebugVisualizer:
    """调试可视化工具"""
    
    def __init__(self, log_dir: str = "debug_logs"):
        """
        初始化可视化工具
        
        Args:
            log_dir: 日志目录路径
        """
        self.log_dir = Path(log_dir)
        self.images_dir = self.log_dir / "images"
        self.data_dir = self.log_dir / "data"
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"🎨 可视化工具已初始化: {self.log_dir}")
    
    def create_debug_image(self, 
                          cam_image: np.ndarray,
                          eih_image: np.ndarray,
                          joint_action: np.ndarray,
                          gripper_action: int,
                          gripper_width: float,
                          step: int,
                          inference_time: float = 0.0,
                          anomalies: List[str] = None) -> np.ndarray:
        """
        创建调试图像，显示输入和预测结果
        
        Args:
            cam_image: CAM图像
            eih_image: EIH图像
            joint_action: 关节动作 (8维)
            gripper_action: gripper编码器值
            gripper_width: gripper物理宽度
            step: 步数
            inference_time: 推理时间
            anomalies: 异常列表
            
        Returns:
            np.ndarray: 合成的调试图像
        """
        # 调整图像大小
        cam_resized = cv2.resize(cam_image, (320, 240))
        eih_resized = cv2.resize(eih_image, (320, 240))
        
        # 创建主画布
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # 放置图像
        canvas[20:260, 20:340] = cam_resized
        canvas[20:260, 380:700] = eih_resized
        
        # 添加标题
        cv2.putText(canvas, f"Step: {step}", (20, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, f"Time: {inference_time:.3f}s", (20, 330), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制关节动作柱状图
        joint_x = 20
        joint_y = 350
        joint_width = 30
        joint_height = 100
        
        for i, action in enumerate(joint_action):
            x = joint_x + i * (joint_width + 10)
            height = int(abs(action) * joint_height * 2)  # 缩放显示
            color = (0, 255, 0) if action >= 0 else (0, 0, 255)
            
            cv2.rectangle(canvas, 
                         (x, joint_y + joint_height - height),
                         (x + joint_width, joint_y + joint_height),
                         color, -1)
            
            # 添加关节标签
            cv2.putText(canvas, f"J{i}", (x, joint_y + joint_height + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制gripper信息
        gripper_x = 500
        gripper_y = 350
        
        cv2.putText(canvas, f"Gripper:", (gripper_x, gripper_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Encoder: {gripper_action}", (gripper_x, gripper_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"Width: {gripper_width:.3f}m", (gripper_x, gripper_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 绘制gripper状态条
        gripper_bar_width = 200
        gripper_bar_height = 20
        gripper_progress = gripper_action / 255.0
        
        cv2.rectangle(canvas, (gripper_x, gripper_y + 70), 
                     (gripper_x + gripper_bar_width, gripper_y + 70 + gripper_bar_height),
                     (100, 100, 100), -1)
        cv2.rectangle(canvas, (gripper_x, gripper_y + 70), 
                     (gripper_x + int(gripper_progress * gripper_bar_width), gripper_y + 70 + gripper_bar_height),
                     (0, 255, 0), -1)
        
        # 显示异常信息
        if anomalies:
            anomaly_y = 500
            cv2.putText(canvas, "Anomalies:", (20, anomaly_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            for i, anomaly in enumerate(anomalies):
                cv2.putText(canvas, f"- {anomaly}", (20, anomaly_y + 25 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return canvas
    
    def create_action_timeline(self, records: List[Dict[str, Any]], 
                              output_path: Optional[str] = None) -> str:
        """
        创建动作时间线图
        
        Args:
            records: 推理记录列表
            output_path: 输出路径
            
        Returns:
            str: 保存路径
        """
        if not records:
            print("⚠️  没有记录数据")
            return ""
        
        # 提取数据
        steps = [r["step"] for r in records]
        joint_actions = [r["output"]["joint_action"] for r in records if "joint_action" in r["output"]]
        gripper_actions = [r["output"]["gripper_action"] for r in records if "gripper_action" in r["output"]]
        inference_times = [r["metadata"].get("inference_time", 0) for r in records]
        
        if not joint_actions:
            print("⚠️  没有关节动作数据")
            return ""
        
        # 创建图形
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # 关节动作时间线
        joint_actions = np.array(joint_actions)
        for i in range(joint_actions.shape[1]):
            axes[0].plot(steps, joint_actions[:, i], label=f'Joint {i}', alpha=0.7)
        axes[0].set_title('Joint Actions Over Time')
        axes[0].set_ylabel('Action Value')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Gripper动作时间线
        if gripper_actions:
            axes[1].plot(steps, gripper_actions, 'b-', linewidth=2, label='Gripper Action')
            axes[1].set_title('Gripper Action Over Time')
            axes[1].set_ylabel('Encoder Value')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 推理时间
        if inference_times:
            axes[2].plot(steps, inference_times, 'r-', linewidth=2, label='Inference Time')
            axes[2].set_title('Inference Time Over Time')
            axes[2].set_xlabel('Step')
            axes[2].set_ylabel('Time (s)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        if not output_path:
            output_path = self.log_dir / f"action_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 动作时间线已保存: {output_path}")
        return str(output_path)
    
    def create_action_distribution(self, records: List[Dict[str, Any]], 
                                  output_path: Optional[str] = None) -> str:
        """
        创建动作分布图
        
        Args:
            records: 推理记录列表
            output_path: 输出路径
            
        Returns:
            str: 保存路径
        """
        if not records:
            print("⚠️  没有记录数据")
            return ""
        
        # 提取关节动作数据
        joint_actions = []
        for r in records:
            if "joint_action" in r["output"]:
                joint_actions.append(r["output"]["joint_action"])
        
        if not joint_actions:
            print("⚠️  没有关节动作数据")
            return ""
        
        joint_actions = np.array(joint_actions)
        
        # 创建图形
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(min(8, joint_actions.shape[1])):
            axes[i].hist(joint_actions[:, i], bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Joint {i} Action Distribution')
            axes[i].set_xlabel('Action Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        if not output_path:
            output_path = self.log_dir / f"action_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 动作分布图已保存: {output_path}")
        return str(output_path)
    
    def create_debug_video(self, 
                          records: List[Dict[str, Any]], 
                          output_path: Optional[str] = None,
                          fps: int = 10) -> str:
        """
        创建调试视频
        
        Args:
            records: 推理记录列表
            output_path: 输出路径
            fps: 视频帧率
            
        Returns:
            str: 保存路径
        """
        if not records:
            print("⚠️  没有记录数据")
            return ""
        
        if not output_path:
            output_path = self.log_dir / f"debug_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (800, 600))
        
        print(f"🎬 开始创建调试视频...")
        
        for i, record in enumerate(records):
            # 加载图像
            cam_path = record["input"].get("cam_image_path")
            eih_path = record["input"].get("eih_image_path")
            
            if not cam_path or not eih_path:
                continue
            
            try:
                cam_image = cv2.imread(cam_path)
                eih_image = cv2.imread(eih_path)
                
                if cam_image is None or eih_image is None:
                    continue
                
                # 创建调试图像
                debug_image = self.create_debug_image(
                    cam_image=cam_image,
                    eih_image=eih_image,
                    joint_action=np.array(record["output"]["joint_action"]),
                    gripper_action=record["output"]["gripper_action"],
                    gripper_width=record["output"].get("gripper_width", 0.0),
                    step=record["step"],
                    inference_time=record["metadata"].get("inference_time", 0.0)
                )
                
                out.write(debug_image)
                
                if i % 100 == 0:
                    print(f"  处理进度: {i}/{len(records)}")
                    
            except Exception as e:
                print(f"⚠️  处理记录 {i} 时出错: {e}")
                continue
        
        out.release()
        print(f"🎬 调试视频已保存: {output_path}")
        return str(output_path)
    
    def load_records_from_files(self, max_records: int = 1000) -> List[Dict[str, Any]]:
        """
        从文件加载记录
        
        Args:
            max_records: 最大加载记录数
            
        Returns:
            List[Dict[str, Any]]: 记录列表
        """
        records = []
        data_files = sorted(self.data_dir.glob("*.json"))
        
        for i, file_path in enumerate(data_files):
            if i >= max_records:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    record = json.load(f)
                    records.append(record)
            except Exception as e:
                print(f"⚠️  加载文件 {file_path} 时出错: {e}")
                continue
        
        print(f"📁 已加载 {len(records)} 条记录")
        return records
