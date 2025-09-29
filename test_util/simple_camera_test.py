#!/usr/bin/env python3
"""
简单相机测试脚本 - 快速检查cam4相机状态

使用方法:
  python simple_camera_test.py
"""

import sys
from pathlib import Path

# 添加项目路径
project_dir = Path(__file__).parent.parent
r3kit_path = project_dir / "model" / "r3kit"
sys.path.insert(0, str(r3kit_path))

def check_camera_devices():
    """检查相机设备"""
    print("🔍 检查相机设备...")
    
    import os
    video_devices = []
    for i in range(10):  # 检查 /dev/video0 到 /dev/video9
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            video_devices.append(device_path)
    
    if video_devices:
        print(f"✅ 找到相机设备: {video_devices}")
    else:
        print("❌ 未找到相机设备")
    
    return video_devices

def check_camera_processes():
    """检查占用相机的进程"""
    print("\n🔍 检查占用相机的进程...")
    
    import subprocess
    try:
        result = subprocess.run(['lsof', '/dev/video*'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            print("⚠️  发现占用相机的进程:")
            print(result.stdout)
        else:
            print("✅ 没有进程占用相机设备")
    except Exception as e:
        print(f"⚠️  检查进程时出错: {e}")

def test_r3kit_import():
    """测试r3kit导入"""
    print("\n🔍 测试r3kit模块导入...")
    
    try:
        import pyrealsense2 as rs
        print("✅ pyrealsense2 导入成功")
    except ImportError as e:
        print(f"❌ pyrealsense2 导入失败: {e}")
        return False
    
    try:
        from r3kit.devices.camera.realsense import config as rs_cfg
        from r3kit.devices.camera.realsense.d415 import D415
        print("✅ r3kit相机模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ r3kit相机模块导入失败: {e}")
        return False

def test_camera_connection():
    """测试相机连接"""
    print("\n🔍 测试相机连接...")
    
    try:
        from r3kit.devices.camera.realsense import config as rs_cfg
        from r3kit.devices.camera.realsense.d415 import D415
        import pyrealsense2 as rs
        
        # 配置流
        rs_cfg.D415_STREAMS = [
            (rs.stream.depth, 640, 480, rs.format.z16, 30),
            (rs.stream.color, 640, 480, rs.format.bgr8, 30),
        ]
        
        # 尝试连接相机
        cam = D415(id="327322062498", depth=True, name="cam4")
        print("✅ 相机连接成功!")
        
        # 尝试获取图像
        import time
        time.sleep(1)
        color, depth = cam.get()
        
        if color is not None:
            print(f"✅ 成功获取彩色图像: {color.shape}")
        else:
            print("❌ 获取彩色图像失败")
            
        if depth is not None:
            print(f"✅ 成功获取深度图像: {depth.shape}")
        else:
            print("❌ 获取深度图像失败")
        
        # 清理
        if hasattr(cam, 'stop'):
            cam.stop()
        elif hasattr(cam, 'close'):
            cam.close()
            
        return True
        
    except Exception as e:
        print(f"❌ 相机连接失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始简单相机测试")
    print("=" * 50)
    
    # 检查设备
    devices = check_camera_devices()
    
    # 检查进程
    check_camera_processes()
    
    # 测试导入
    if not test_r3kit_import():
        print("\n❌ 模块导入失败，无法继续测试")
        return 1
    
    # 测试连接
    if test_camera_connection():
        print("\n✅ 相机测试成功!")
        return 0
    else:
        print("\n❌ 相机测试失败!")
        return 1

if __name__ == "__main__":
    exit(main())
