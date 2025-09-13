#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理数据记录器

用于实时保存policy部署过程中的所有数据，包括：
- 输入图像（cam, eih）
- 机器人状态
- 预测动作
- 推理时间
- 元数据
"""

import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading
from collections import deque


class InferenceLogger:
    """推理数据记录器"""
    
    def __init__(self, 
                 log_dir: str = "debug_logs",
                 max_logs: int = 10000,
                 save_images: bool = True,
                 save_frequency: int = 1):
        """
        初始化推理记录器
        
        Args:
            log_dir: 日志保存目录
            max_logs: 最大保存日志数量
            save_images: 是否保存图像
            save_frequency: 保存频率（每N步保存一次）
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.images_dir = self.log_dir / "images"
        self.data_dir = self.log_dir / "data"
        self.images_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.max_logs = max_logs
        self.save_images = save_images
        self.save_frequency = save_frequency
        
        # 数据缓存
        self.log_queue = deque(maxlen=max_logs)
        self.step_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            "total_steps": 0,
            "saved_steps": 0,
            "anomaly_count": 0,
            "avg_inference_time": 0.0,
            "start_time": time.time()
        }
        
        print(f"🔍 推理记录器已初始化:")
        print(f"  日志目录: {self.log_dir}")
        print(f"  会话ID: {self.session_id}")
        print(f"  保存频率: 每{save_frequency}步")
        print(f"  图像保存: {'启用' if save_images else '禁用'}")
    
    def log_inference(self, 
                     input_data: Dict[str, Any],
                     output_data: Dict[str, Any], 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录一次推理过程
        
        Args:
            input_data: 输入数据，包含图像和状态
            output_data: 输出数据，包含预测动作
            metadata: 元数据，包含推理时间等
            
        Returns:
            str: 记录ID
        """
        with self.lock:
            self.step_count += 1
            self.stats["total_steps"] += 1
            
            # 生成记录ID
            record_id = f"{self.session_id}_step_{self.step_count:06d}"
            
            # 准备记录数据
            record = {
                "record_id": record_id,
                "step": self.step_count,
                "timestamp": datetime.now().isoformat(),
                "input": self._process_input_data(input_data, record_id),
                "output": self._process_output_data(output_data),
                "metadata": metadata or {}
            }
            
            # 添加到队列
            self.log_queue.append(record)
            
            # 按频率保存
            if self.step_count % self.save_frequency == 0:
                self._save_record(record)
                self.stats["saved_steps"] += 1
            
            # 更新统计信息
            if metadata and "inference_time" in metadata:
                self._update_inference_stats(metadata["inference_time"])
            
            return record_id
    
    def _process_input_data(self, input_data: Dict[str, Any], record_id: str) -> Dict[str, Any]:
        """处理输入数据"""
        processed = {}
        
        # 处理图像
        if self.save_images:
            if "cam_image" in input_data:
                cam_path = self.images_dir / f"{record_id}_cam.png"
                cv2.imwrite(str(cam_path), input_data["cam_image"])
                processed["cam_image_path"] = str(cam_path)
            
            if "eih_image" in input_data:
                eih_path = self.images_dir / f"{record_id}_eih.png"
                cv2.imwrite(str(eih_path), input_data["eih_image"])
                processed["eih_image_path"] = str(eih_path)
        
        # 处理状态数据
        if "robot_state" in input_data:
            processed["robot_state"] = input_data["robot_state"].tolist() if hasattr(input_data["robot_state"], 'tolist') else input_data["robot_state"]
        
        if "gripper_state" in input_data:
            processed["gripper_state"] = float(input_data["gripper_state"])
        
        return processed
    
    def _process_output_data(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输出数据"""
        processed = {}
        
        if "joint_action" in output_data:
            processed["joint_action"] = output_data["joint_action"].tolist() if hasattr(output_data["joint_action"], 'tolist') else output_data["joint_action"]
        
        if "gripper_action" in output_data:
            processed["gripper_action"] = int(output_data["gripper_action"])
        
        if "gripper_width" in output_data:
            processed["gripper_width"] = float(output_data["gripper_width"])
        
        if "full_action" in output_data:
            processed["full_action"] = output_data["full_action"].tolist() if hasattr(output_data["full_action"], 'tolist') else output_data["full_action"]
        
        return processed
    
    def _save_record(self, record: Dict[str, Any]) -> None:
        """保存单条记录到文件"""
        record_file = self.data_dir / f"{record['record_id']}.json"
        
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    
    def _update_inference_stats(self, inference_time: float) -> None:
        """更新推理时间统计"""
        # 简单的移动平均
        alpha = 0.1
        self.stats["avg_inference_time"] = (
            alpha * inference_time + 
            (1 - alpha) * self.stats["avg_inference_time"]
        )
    
    def get_latest_records(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最新的N条记录"""
        with self.lock:
            return list(self.log_queue)[-n:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            current_time = time.time()
            runtime = current_time - self.stats["start_time"]
            
            return {
                **self.stats,
                "runtime_seconds": runtime,
                "steps_per_second": self.stats["total_steps"] / runtime if runtime > 0 else 0,
                "queue_size": len(self.log_queue)
            }
    
    def save_session_summary(self) -> str:
        """保存会话总结"""
        summary_file = self.log_dir / f"{self.session_id}_summary.json"
        
        summary = {
            "session_id": self.session_id,
            "start_time": datetime.fromtimestamp(self.stats["start_time"]).isoformat(),
            "end_time": datetime.now().isoformat(),
            "stats": self.get_stats(),
            "config": {
                "max_logs": self.max_logs,
                "save_images": self.save_images,
                "save_frequency": self.save_frequency
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 会话总结已保存: {summary_file}")
        return str(summary_file)
    
    def clear_logs(self) -> None:
        """清空所有日志"""
        with self.lock:
            self.log_queue.clear()
            self.step_count = 0
            self.stats["total_steps"] = 0
            self.stats["saved_steps"] = 0
            self.stats["anomaly_count"] = 0
            print("🗑️  日志已清空")
    
    def export_to_csv(self, output_file: Optional[str] = None) -> str:
        """导出数据到CSV文件"""
        import pandas as pd
        
        if not output_file:
            output_file = self.log_dir / f"{self.session_id}_data.csv"
        
        # 准备数据
        data = []
        for record in self.log_queue:
            row = {
                "step": record["step"],
                "timestamp": record["timestamp"],
                "inference_time": record["metadata"].get("inference_time", 0),
            }
            
            # 添加关节动作
            if "joint_action" in record["output"]:
                for i, action in enumerate(record["output"]["joint_action"]):
                    row[f"joint_{i}"] = action
            
            # 添加gripper动作
            if "gripper_action" in record["output"]:
                row["gripper_action"] = record["output"]["gripper_action"]
            
            data.append(row)
        
        # 保存CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        print(f"📈 数据已导出到CSV: {output_file}")
        return str(output_file)
