# examples/test.py
from r3kit.devices.encoder.pdcd.angler import Angler
from r3kit.devices.camera.realsense.d415 import D415
import time
import numpy as np
import cv2
import os

def process_angles(angles, zero_point, middle_point):
    """处理角度数据，处理跳变"""
    processed_angles = np.copy(angles)
    for i in range(len(processed_angles)):
        angle = np.copy(processed_angles[i])
        angle[angle < zero_point] += 360
        angle -= middle_point
        processed_angles[i] = angle
    return processed_angles

def align_data(angler_data, camera_data):
    """对齐编码器和相机的数据"""
    # 获取时间戳
    angler_timestamps = np.array(angler_data["timestamp_ms"])
    camera_timestamps = np.array(camera_data["timestamp_ms"])
    
    # 找到时间戳重叠的范围
    start_time = max(angler_timestamps[0], camera_timestamps[0])
    end_time = min(angler_timestamps[-1], camera_timestamps[-1])
    
    # 找到对应的索引
    angler_start_idx = np.searchsorted(angler_timestamps, start_time)
    angler_end_idx = np.searchsorted(angler_timestamps, end_time, side='right')
    camera_start_idx = np.searchsorted(camera_timestamps, start_time)
    camera_end_idx = np.searchsorted(camera_timestamps, end_time, side='right')
    
    # 提取对齐后的数据
    aligned_angler_data = {
        "angle": angler_data["angle"][angler_start_idx:angler_end_idx],
        "timestamp_ms": angler_timestamps[angler_start_idx:angler_end_idx]
    }
    
    aligned_camera_data = {
        "color": camera_data["color"][camera_start_idx:camera_end_idx],
        "depth": camera_data["depth"][camera_start_idx:camera_end_idx],
        "timestamp_ms": camera_timestamps[camera_start_idx:camera_end_idx]
    }
    
    return aligned_angler_data, aligned_camera_data

def save_sync_data(angler_data, camera_data, save_dir, zero_point, middle_point, camera):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 对齐数据
    aligned_angler_data, aligned_camera_data = align_data(angler_data, camera_data)
    
    # 处理角度数据
    processed_angles = process_angles(np.array(aligned_angler_data["angle"]), zero_point, middle_point)
    
    # 保存处理后的角度数据
    np.save(os.path.join(save_dir, "angles.npy"), processed_angles)
    np.save(os.path.join(save_dir, "angle_timestamps.npy"), np.array(aligned_angler_data["timestamp_ms"]))
    
    # 保存相机数据
    camera.save_streaming(save_dir, aligned_camera_data)
    
    # 保存时间戳对齐信息
    np.save(os.path.join(save_dir, "camera_timestamps.npy"), np.array(aligned_camera_data["timestamp_ms"]))
    
    # 计算并保存采样频率信息
    angler_freq = len(aligned_angler_data["timestamp_ms"]) / (aligned_angler_data["timestamp_ms"][-1] - aligned_angler_data["timestamp_ms"][0]) * 1000
    camera_freq = len(aligned_camera_data["timestamp_ms"]) / (aligned_camera_data["timestamp_ms"][-1] - aligned_camera_data["timestamp_ms"][0]) * 1000
    with open(os.path.join(save_dir, "sampling_freq.txt"), "w") as f:
        f.write(f"Angler sampling frequency: {angler_freq:.2f} Hz\n")
        f.write(f"Camera sampling frequency: {camera_freq:.2f} Hz\n")

def main():
    # 初始化设备
    encoder = Angler(id='/dev/ttyUSB0', index=[1,2,3,4,5,6,7], baudrate=1000000, fps=30, gap=0.002)
    camera = D415(id='104122063633', depth=True)  # 使用您的相机ID
    
    # 定义零点和中心点
    zero_point = np.array([10, 90, 0, 0, 180, 0, 120])
    middle_point = np.array([203.03, 339.08, 190.9, 251.46, 7.21, 123.4, 293.64])
    
    # 开始数据采集
    print("开始数据采集...")
    encoder.start_streaming()
    camera.start_streaming()
    
    # 等待用户输入来停止采集
    input("按回车键停止数据采集...")
    
    # 停止采集并获取数据
    angler_data = encoder.stop_streaming()
    camera_data = camera.stop_streaming()
    
    # 保存数据
    save_dir = "sync_data"
    save_sync_data(angler_data, camera_data, save_dir, zero_point, middle_point, camera)
    
    print(f"数据采集完成！数据已保存到 {save_dir} 目录")

if __name__ == "__main__":
    main()