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

def save_sync_data(angler_data, camera_data, save_dir, zero_point, middle_point):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 处理角度数据
    processed_angles = process_angles(np.array(angler_data["angle"]), zero_point, middle_point)
    
    # 保存处理后的角度数据
    np.save(os.path.join(save_dir, "angles.npy"), processed_angles)
    np.save(os.path.join(save_dir, "angle_timestamps.npy"), np.array(angler_data["timestamp_ms"]))
    
    # 保存相机数据
    camera.save_streaming(save_dir, camera_data)

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
    save_sync_data(angler_data, camera_data, save_dir, zero_point, middle_point)
    
    print(f"数据采集完成！数据已保存到 {save_dir} 目录")

if __name__ == "__main__":
    main()