#!/usr/bin/env python3
"""
相机测试脚本 - 使用r3kit测试cam4相机连接和拍照功能

使用方法:
  python test_camera_cam4.py

功能:
  - 测试cam4相机连接
  - 拍摄RGB图像
  - 保存图像到文件
  - 显示相机状态信息
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# 添加项目路径
project_dir = Path(__file__).parent.parent
r3kit_path = project_dir / "model" / "r3kit"
sys.path.insert(0, str(r3kit_path))

# 导入r3kit相机相关模块
try:
    import pyrealsense2 as rs
    from r3kit.devices.camera.realsense import config as rs_cfg
    from r3kit.devices.camera.realsense.d415 import D415
    print("✅ r3kit相机模块导入成功")
except ImportError as e:
    print(f"❌ r3kit相机模块导入失败: {e}")
    sys.exit(1)

# 相机配置
FPS = 30
CAM4_SERIAL = "327322062498"  # cam4的序列号

def test_camera_connection():
    """测试相机连接"""
    print("🔍 开始测试cam4相机连接...")
    
    # 配置流
    rs_cfg.D415_STREAMS = [
        (rs.stream.depth, 640, 480, rs.format.z16, FPS),
        (rs.stream.color, 640, 480, rs.format.bgr8, FPS),
    ]
    
    try:
        # 创建相机实例
        print(f"📷 正在连接相机 cam4 (序列号: {CAM4_SERIAL})...")
        cam = D415(id=CAM4_SERIAL, depth=True, name="cam4")
        print("✅ 相机连接成功!")
        
        # 等待相机稳定
        print("⏳ 等待相机稳定...")
        time.sleep(2)
        
        return cam
        
    except Exception as e:
        print(f"❌ 相机连接失败: {e}")
        return None

def capture_image(cam):
    """拍摄图像"""
    print("📸 正在拍摄图像...")
    
    try:
        # 获取图像
        color, depth = cam.get()
        
        if color is None:
            print("❌ 获取彩色图像失败")
            return None, None
            
        if depth is None:
            print("❌ 获取深度图像失败")
            return color, None
            
        print("✅ 图像获取成功!")
        print(f"   彩色图像尺寸: {color.shape}")
        print(f"   深度图像尺寸: {depth.shape}")
        
        return color, depth
        
    except Exception as e:
        print(f"❌ 图像获取失败: {e}")
        return None, None

def save_images(color_img, depth_img, output_dir="camera_test_output"):
    """保存图像"""
    print("💾 正在保存图像...")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    try:
        if color_img is not None:
            # 保存RGB图像
            rgb_path = output_path / f"cam4_rgb_{timestamp}.jpg"
            cv2.imwrite(str(rgb_path), color_img)
            print(f"✅ RGB图像已保存: {rgb_path}")
            
            # 保存BGR图像（OpenCV格式）
            bgr_path = output_path / f"cam4_bgr_{timestamp}.jpg"
            cv2.imwrite(str(bgr_path), color_img)
            print(f"✅ BGR图像已保存: {bgr_path}")
        
        if depth_img is not None:
            # 保存深度图像
            depth_path = output_path / f"cam4_depth_{timestamp}.png"
            cv2.imwrite(str(depth_path), depth_img)
            print(f"✅ 深度图像已保存: {depth_path}")
            
            # 保存深度图像的可视化版本
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
            depth_vis_path = output_path / f"cam4_depth_vis_{timestamp}.jpg"
            cv2.imwrite(str(depth_vis_path), depth_vis)
            print(f"✅ 深度可视化图像已保存: {depth_vis_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 图像保存失败: {e}")
        return False

def display_image_info(color_img, depth_img):
    """显示图像信息"""
    print("\n📊 图像信息:")
    
    if color_img is not None:
        print(f"   彩色图像:")
        print(f"     - 尺寸: {color_img.shape}")
        print(f"     - 数据类型: {color_img.dtype}")
        print(f"     - 像素值范围: {color_img.min()} - {color_img.max()}")
    
    if depth_img is not None:
        print(f"   深度图像:")
        print(f"     - 尺寸: {depth_img.shape}")
        print(f"     - 数据类型: {depth_img.dtype}")
        print(f"     - 深度值范围: {depth_img.min()} - {depth_img.max()} mm")

def main():
    """主函数"""
    print("🚀 开始cam4相机测试")
    print("=" * 50)
    
    # 测试相机连接
    cam = test_camera_connection()
    if cam is None:
        print("❌ 相机连接失败，测试终止")
        return 1
    
    try:
        # 拍摄图像
        color_img, depth_img = capture_image(cam)
        if color_img is None:
            print("❌ 图像拍摄失败，测试终止")
            return 1
        
        # 显示图像信息
        display_image_info(color_img, depth_img)
        
        # 保存图像
        success = save_images(color_img, depth_img)
        if not success:
            print("❌ 图像保存失败")
            return 1
        
        print("\n✅ 相机测试完成!")
        print("📁 图像已保存到 camera_test_output/ 目录")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断测试")
        return 1
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # 清理资源
        try:
            if hasattr(cam, 'stop'):
                cam.stop()
            elif hasattr(cam, 'close'):
                cam.close()
            print("🧹 相机资源已清理")
        except Exception as e:
            print(f"⚠️  清理相机资源时出错: {e}")

if __name__ == "__main__":
    exit(main())
