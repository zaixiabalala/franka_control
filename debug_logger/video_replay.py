#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频回放生成器

用于从保存的推理数据生成调试视频，支持：
- 多视角回放
- 动作可视化
- 异常标记
- 时间同步
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import threading
import time


class VideoReplayGenerator:
    """视频回放生成器"""
    
    def __init__(self, log_dir: str = "debug_logs"):
        """
        初始化视频回放生成器
        
        Args:
            log_dir: 日志目录路径
        """
        self.log_dir = Path(log_dir)
        self.images_dir = self.log_dir / "images"
        self.data_dir = self.log_dir / "data"
        
        print(f"🎬 视频回放生成器已初始化: {self.log_dir}")
    
    def generate_debug_video(self, 
                           session_id: Optional[str] = None,
                           output_path: Optional[str] = None,
                           fps: int = 10,
                           show_anomalies: bool = True,
                           show_actions: bool = True,
                           max_frames: int = 1000) -> str:
        """
        生成调试视频
        
        Args:
            session_id: 会话ID，如果为None则使用最新的
            output_path: 输出路径
            fps: 视频帧率
            show_anomalies: 是否显示异常标记
            show_actions: 是否显示动作信息
            max_frames: 最大帧数
            
        Returns:
            str: 视频保存路径
        """
        # 获取记录数据
        records = self._load_records(session_id, max_frames)
        if not records:
            print("⚠️  没有找到记录数据")
            return ""
        
        # 设置输出路径
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.log_dir / f"debug_replay_{timestamp}.mp4"
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (1200, 800))
        
        print(f"🎬 开始生成调试视频...")
        print(f"  记录数: {len(records)}")
        print(f"  帧率: {fps}")
        print(f"  输出路径: {output_path}")
        
        try:
            for i, record in enumerate(records):
                # 创建帧
                frame = self._create_debug_frame(record, show_anomalies, show_actions)
                
                if frame is not None:
                    out.write(frame)
                
                # 显示进度
                if i % 50 == 0:
                    progress = (i + 1) / len(records) * 100
                    print(f"  进度: {progress:.1f}% ({i+1}/{len(records)})")
        
        except Exception as e:
            print(f"❌ 生成视频时出错: {e}")
            return ""
        
        finally:
            out.release()
        
        print(f"✅ 调试视频已生成: {output_path}")
        return str(output_path)
    
    def generate_comparison_video(self, 
                                session_ids: List[str],
                                output_path: Optional[str] = None,
                                fps: int = 10) -> str:
        """
        生成对比视频（多个会话）
        
        Args:
            session_ids: 会话ID列表
            output_path: 输出路径
            fps: 视频帧率
            
        Returns:
            str: 视频保存路径
        """
        if len(session_ids) < 2:
            print("⚠️  需要至少2个会话进行对比")
            return ""
        
        # 加载所有会话的记录
        all_records = []
        for session_id in session_ids:
            records = self._load_records(session_id)
            all_records.append(records)
        
        # 找到最短的记录长度
        min_length = min(len(records) for records in all_records)
        
        # 设置输出路径
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.log_dir / f"comparison_replay_{timestamp}.mp4"
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (1600, 600))
        
        print(f"🎬 开始生成对比视频...")
        print(f"  会话数: {len(session_ids)}")
        print(f"  帧数: {min_length}")
        
        try:
            for i in range(min_length):
                # 创建对比帧
                frame = self._create_comparison_frame([records[i] for records in all_records], session_ids)
                
                if frame is not None:
                    out.write(frame)
                
                # 显示进度
                if i % 50 == 0:
                    progress = (i + 1) / min_length * 100
                    print(f"  进度: {progress:.1f}% ({i+1}/{min_length})")
        
        except Exception as e:
            print(f"❌ 生成对比视频时出错: {e}")
            return ""
        
        finally:
            out.release()
        
        print(f"✅ 对比视频已生成: {output_path}")
        return str(output_path)
    
    def _load_records(self, session_id: Optional[str] = None, max_frames: int = 1000) -> List[Dict[str, Any]]:
        """加载记录数据"""
        records = []
        
        if session_id:
            # 加载指定会话的记录
            pattern = f"*{session_id}*"
            data_files = sorted(self.data_dir.glob(pattern))
        else:
            # 加载所有记录
            data_files = sorted(self.data_dir.glob("*.json"))
        
        for i, file_path in enumerate(data_files):
            if i >= max_frames:
                break
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    record = json.load(f)
                    records.append(record)
            except Exception as e:
                print(f"⚠️  加载文件 {file_path} 时出错: {e}")
                continue
        
        return records
    
    def _create_debug_frame(self, 
                           record: Dict[str, Any], 
                           show_anomalies: bool = True,
                           show_actions: bool = True) -> Optional[np.ndarray]:
        """创建调试帧"""
        try:
            # 加载图像
            cam_path = record["input"].get("cam_image_path")
            eih_path = record["input"].get("eih_image_path")
            
            if not cam_path or not eih_path:
                return None
            
            cam_image = cv2.imread(cam_path)
            eih_image = cv2.imread(eih_path)
            
            if cam_image is None or eih_image is None:
                return None
            
            # 调整图像大小
            cam_resized = cv2.resize(cam_image, (400, 300))
            eih_resized = cv2.resize(eih_image, (400, 300))
            
            # 创建画布
            canvas = np.zeros((800, 1200, 3), dtype=np.uint8)
            
            # 放置图像
            canvas[50:350, 50:450] = cam_resized
            canvas[50:350, 500:900] = eih_resized
            
            # 添加标题
            cv2.putText(canvas, f"Step: {record['step']}", (50, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(canvas, f"Time: {record['timestamp']}", (50, 430), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 显示动作信息
            if show_actions and "joint_action" in record["output"]:
                self._draw_action_info(canvas, record["output"], (50, 460))
            
            # 显示异常信息
            if show_anomalies and "anomalies" in record:
                self._draw_anomaly_info(canvas, record["anomalies"], (50, 600))
            
            return canvas
            
        except Exception as e:
            print(f"⚠️  创建调试帧时出错: {e}")
            return None
    
    def _create_comparison_frame(self, 
                               records: List[Dict[str, Any]], 
                               session_ids: List[str]) -> Optional[np.ndarray]:
        """创建对比帧"""
        try:
            # 创建画布
            canvas = np.zeros((600, 1600, 3), dtype=np.uint8)
            
            # 为每个会话创建子画面
            for i, (record, session_id) in enumerate(zip(records, session_ids)):
                x_offset = i * 400
                
                # 加载图像
                cam_path = record["input"].get("cam_image_path")
                if cam_path:
                    cam_image = cv2.imread(cam_path)
                    if cam_image is not None:
                        cam_resized = cv2.resize(cam_image, (300, 200))
                        canvas[50:250, x_offset+50:x_offset+350] = cam_resized
                
                # 添加会话标签
                cv2.putText(canvas, f"Session: {session_id}", (x_offset+50, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(canvas, f"Step: {record['step']}", (x_offset+50, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 显示动作信息
                if "joint_action" in record["output"]:
                    self._draw_action_info(canvas, record["output"], (x_offset+50, 320))
            
            return canvas
            
        except Exception as e:
            print(f"⚠️  创建对比帧时出错: {e}")
            return None
    
    def _draw_action_info(self, canvas: np.ndarray, output_data: Dict[str, Any], pos: Tuple[int, int]) -> None:
        """绘制动作信息"""
        x, y = pos
        
        # 绘制关节动作
        if "joint_action" in output_data:
            joint_action = output_data["joint_action"]
            for i, action in enumerate(joint_action):
                bar_x = x + i * 30
                bar_height = int(abs(action) * 50)
                color = (0, 255, 0) if action >= 0 else (0, 0, 255)
                
                cv2.rectangle(canvas, (bar_x, y - bar_height), (bar_x + 25, y), color, -1)
                cv2.putText(canvas, f"J{i}", (bar_x, y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 绘制gripper信息
        if "gripper_action" in output_data:
            gripper_action = output_data["gripper_action"]
            cv2.putText(canvas, f"Gripper: {gripper_action}", (x, y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_anomaly_info(self, canvas: np.ndarray, anomalies: List[str], pos: Tuple[int, int]) -> None:
        """绘制异常信息"""
        x, y = pos
        
        cv2.putText(canvas, "Anomalies:", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        for i, anomaly in enumerate(anomalies[:3]):  # 最多显示3个异常
            cv2.putText(canvas, f"- {anomaly}", (x, y + 20 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def list_sessions(self) -> List[str]:
        """列出所有会话"""
        sessions = set()
        
        for file_path in self.data_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    record = json.load(f)
                    if "record_id" in record:
                        session_id = record["record_id"].split("_step_")[0]
                        sessions.add(session_id)
            except:
                continue
        
        return sorted(list(sessions))
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        records = self._load_records(session_id)
        
        if not records:
            return {}
        
        # 统计信息
        steps = [r["step"] for r in records]
        inference_times = [r["metadata"].get("inference_time", 0) for r in records]
        
        return {
            "session_id": session_id,
            "total_steps": len(records),
            "start_step": min(steps),
            "end_step": max(steps),
            "avg_inference_time": np.mean(inference_times) if inference_times else 0,
            "min_inference_time": min(inference_times) if inference_times else 0,
            "max_inference_time": max(inference_times) if inference_times else 0
        }
