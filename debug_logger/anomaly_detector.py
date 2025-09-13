#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测器

用于检测policy推理过程中的异常行为，包括：
- 动作突变
- 推理时间异常
- 无效动作值
- 图像质量异常
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import cv2


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, 
                 action_threshold: float = 0.5,
                 inference_time_threshold: float = 0.1,
                 gripper_threshold: int = 10,
                 history_size: int = 10):
        """
        初始化异常检测器
        
        Args:
            action_threshold: 动作突变阈值
            inference_time_threshold: 推理时间阈值（秒）
            gripper_threshold: gripper动作突变阈值
            history_size: 历史数据大小
        """
        self.action_threshold = action_threshold
        self.inference_time_threshold = inference_time_threshold
        self.gripper_threshold = gripper_threshold
        self.history_size = history_size
        
        # 历史数据缓存
        self.action_history = deque(maxlen=history_size)
        self.gripper_history = deque(maxlen=history_size)
        self.inference_time_history = deque(maxlen=history_size)
        
        # 统计信息
        self.anomaly_count = 0
        self.total_checks = 0
        
        print(f"🔍 异常检测器已初始化:")
        print(f"  动作阈值: {action_threshold}")
        print(f"  推理时间阈值: {inference_time_threshold}s")
        print(f"  Gripper阈值: {gripper_threshold}")
        print(f"  历史大小: {history_size}")
    
    def detect_anomalies(self, 
                        input_data: Dict[str, Any],
                        output_data: Dict[str, Any],
                        metadata: Dict[str, Any]) -> List[str]:
        """
        检测异常
        
        Args:
            input_data: 输入数据
            output_data: 输出数据
            metadata: 元数据
            
        Returns:
            List[str]: 异常列表
        """
        self.total_checks += 1
        anomalies = []
        
        # 检测动作突变
        action_anomalies = self._detect_action_anomalies(output_data)
        anomalies.extend(action_anomalies)
        
        # 检测推理时间异常
        time_anomalies = self._detect_time_anomalies(metadata)
        anomalies.extend(time_anomalies)
        
        # 检测gripper异常
        gripper_anomalies = self._detect_gripper_anomalies(output_data)
        anomalies.extend(gripper_anomalies)
        
        # 检测图像质量异常
        image_anomalies = self._detect_image_anomalies(input_data)
        anomalies.extend(image_anomalies)
        
        # 检测状态异常
        state_anomalies = self._detect_state_anomalies(input_data)
        anomalies.extend(state_anomalies)
        
        # 更新历史数据
        self._update_history(output_data, metadata)
        
        # 更新异常计数
        if anomalies:
            self.anomaly_count += 1
        
        return anomalies
    
    def _detect_action_anomalies(self, output_data: Dict[str, Any]) -> List[str]:
        """检测动作异常"""
        anomalies = []
        
        if "joint_action" not in output_data:
            return anomalies
        
        current_action = np.array(output_data["joint_action"])
        
        # 检测动作值范围异常
        if np.any(np.abs(current_action) > 2.0):  # 假设动作范围是[-2, 2]
            anomalies.append("Action values out of range")
        
        # 检测动作突变
        if len(self.action_history) > 0:
            last_action = np.array(self.action_history[-1])
            action_diff = np.abs(current_action - last_action)
            
            if np.max(action_diff) > self.action_threshold:
                anomalies.append(f"Large action jump: max_diff={np.max(action_diff):.3f}")
            
            # 检测动作序列异常（如全零）
            if np.allclose(current_action, 0, atol=1e-6):
                anomalies.append("All actions are zero")
            
            # 检测动作序列异常（如全相同）
            if len(set(current_action)) == 1:
                anomalies.append("All actions are identical")
        
        return anomalies
    
    def _detect_time_anomalies(self, metadata: Dict[str, Any]) -> List[str]:
        """检测时间异常"""
        anomalies = []
        
        if "inference_time" not in metadata:
            return anomalies
        
        inference_time = metadata["inference_time"]
        
        # 检测推理时间过长
        if inference_time > self.inference_time_threshold:
            anomalies.append(f"Slow inference: {inference_time:.3f}s")
        
        # 检测推理时间异常变化
        if len(self.inference_time_history) > 0:
            last_time = self.inference_time_history[-1]
            time_diff = abs(inference_time - last_time)
            
            if time_diff > self.inference_time_threshold * 0.5:
                anomalies.append(f"Sudden inference time change: {time_diff:.3f}s")
        
        return anomalies
    
    def _detect_gripper_anomalies(self, output_data: Dict[str, Any]) -> List[str]:
        """检测gripper异常"""
        anomalies = []
        
        if "gripper_action" not in output_data:
            return anomalies
        
        current_gripper = output_data["gripper_action"]
        
        # 检测gripper值范围异常
        if not (0 <= current_gripper <= 255):
            anomalies.append(f"Invalid gripper value: {current_gripper}")
        
        # 检测gripper突变
        if len(self.gripper_history) > 0:
            last_gripper = self.gripper_history[-1]
            gripper_diff = abs(current_gripper - last_gripper)
            
            if gripper_diff > self.gripper_threshold:
                anomalies.append(f"Large gripper jump: {gripper_diff}")
        
        return anomalies
    
    def _detect_image_anomalies(self, input_data: Dict[str, Any]) -> List[str]:
        """检测图像质量异常"""
        anomalies = []
        
        # 检测CAM图像
        if "cam_image" in input_data:
            cam_anomalies = self._check_image_quality(input_data["cam_image"], "CAM")
            anomalies.extend(cam_anomalies)
        
        # 检测EIH图像
        if "eih_image" in input_data:
            eih_anomalies = self._check_image_quality(input_data["eih_image"], "EIH")
            anomalies.extend(eih_anomalies)
        
        return anomalies
    
    def _check_image_quality(self, image: np.ndarray, image_type: str) -> List[str]:
        """检查图像质量"""
        anomalies = []
        
        if image is None:
            anomalies.append(f"{image_type} image is None")
            return anomalies
        
        # 检查图像尺寸
        if len(image.shape) != 3 or image.shape[2] != 3:
            anomalies.append(f"{image_type} image has wrong shape: {image.shape}")
        
        # 检查图像是否全黑
        if np.mean(image) < 10:
            anomalies.append(f"{image_type} image is too dark")
        
        # 检查图像是否全白
        if np.mean(image) > 245:
            anomalies.append(f"{image_type} image is too bright")
        
        # 检查图像对比度
        contrast = np.std(image)
        if contrast < 20:
            anomalies.append(f"{image_type} image has low contrast: {contrast:.1f}")
        
        # 检查图像是否有噪声（通过拉普拉斯算子）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            anomalies.append(f"{image_type} image appears blurry: {laplacian_var:.1f}")
        
        return anomalies
    
    def _detect_state_anomalies(self, input_data: Dict[str, Any]) -> List[str]:
        """检测状态异常"""
        anomalies = []
        
        if "robot_state" not in input_data:
            return anomalies
        
        robot_state = np.array(input_data["robot_state"])
        
        # 检测状态值范围异常
        if np.any(np.abs(robot_state) > 10.0):  # 假设状态范围是[-10, 10]
            anomalies.append("Robot state values out of range")
        
        # 检测状态突变
        if len(self.action_history) > 0 and "robot_state" in self.action_history[-1]:
            last_state = np.array(self.action_history[-1]["robot_state"])
            state_diff = np.abs(robot_state - last_state)
            
            if np.max(state_diff) > 1.0:  # 状态突变阈值
                anomalies.append(f"Large state jump: max_diff={np.max(state_diff):.3f}")
        
        return anomalies
    
    def _update_history(self, output_data: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """更新历史数据"""
        # 更新动作历史
        if "joint_action" in output_data:
            self.action_history.append({
                "joint_action": output_data["joint_action"],
                "gripper_action": output_data.get("gripper_action", 0),
                "robot_state": output_data.get("robot_state", [])
            })
        
        # 更新gripper历史
        if "gripper_action" in output_data:
            self.gripper_history.append(output_data["gripper_action"])
        
        # 更新推理时间历史
        if "inference_time" in metadata:
            self.inference_time_history.append(metadata["inference_time"])
    
    def get_anomaly_stats(self) -> Dict[str, Any]:
        """获取异常统计信息"""
        anomaly_rate = self.anomaly_count / self.total_checks if self.total_checks > 0 else 0
        
        return {
            "total_checks": self.total_checks,
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": anomaly_rate,
            "history_size": len(self.action_history)
        }
    
    def reset_history(self) -> None:
        """重置历史数据"""
        self.action_history.clear()
        self.gripper_history.clear()
        self.inference_time_history.clear()
        self.anomaly_count = 0
        self.total_checks = 0
        print("🔄 异常检测器历史已重置")
    
    def set_thresholds(self, 
                      action_threshold: Optional[float] = None,
                      inference_time_threshold: Optional[float] = None,
                      gripper_threshold: Optional[int] = None) -> None:
        """动态调整阈值"""
        if action_threshold is not None:
            self.action_threshold = action_threshold
        if inference_time_threshold is not None:
            self.inference_time_threshold = inference_time_threshold
        if gripper_threshold is not None:
            self.gripper_threshold = gripper_threshold
        
        print(f"⚙️  阈值已更新:")
        print(f"  动作阈值: {self.action_threshold}")
        print(f"  推理时间阈值: {self.inference_time_threshold}s")
        print(f"  Gripper阈值: {self.gripper_threshold}")
