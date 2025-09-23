import pyrealsense2 as rs
from r3kit.devices.camera.realsense import config as rs_cfg
from r3kit.devices.camera.realsense.d415 import D415
import cv2
import numpy as np


class CameraSystem:
    """相机系统接口 - 复用"""
    
    def __init__(self,
        fps=30,
        cameras={   
            "cam4": "327322062498",  # 固定机位视角
            "eih": "038522062288",   # eye-in-hand视角（
        }
    ):
        self.cameras = {}
        self.camera_names = cameras.keys()  # 支持双视角
        self.use_realsense = True
        self.fps = fps
        self.d415_cameras = cameras
        
        # 与采集脚本保持一致的流配置
        rs_cfg.D415_STREAMS = [
            (rs.stream.depth, 640,480, rs.format.z16, self.fps),
            (rs.stream.color, 640,480, rs.format.bgr8, self.fps),
        ]
        for name in self.camera_names:
            serial = self.d415_cameras.get(name)
            if serial is None:
                print(f"{name} 缺少序列号，跳过")
                continue
            try:
                cam = D415(id=serial, depth=True, name=name)
                self.cameras[name] = cam
                print(f"成功初始化相机 {name} (序列号: {serial})")
            except Exception as e:
                print(f"初始化相机 {name} 失败: {e}")
                continue
                
        if len(self.cameras) > 0:
            self.use_realsense = True
            print(f"使用 RealSense D415，相机数量: {len(self.cameras)}")
            print(f"可用相机: {list(self.cameras.keys())}")
        else:
            print("警告: 没有成功初始化任何相机")
    
    def get_image(self, cam_name):
        """获取指定相机的图像"""
        if cam_name not in self.cameras:
            return None
        
        try:
            if self.use_realsense:
                # r3kit D415 接口
                color, depth = self.cameras[cam_name].get()
                if color is None:
                    return None
                # 转 RGB（下游预处理默认以 RGB 处理）
                frame_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                return frame_rgb
            else:
                # OpenCV 摄像头
                ret, frame = self.cameras[cam_name].read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return frame_rgb
                return None
        except Exception as e:
            print(f"获取 {cam_name} 图像失败: {e}")
            return None
    
    def get_image_and_depth(self, cam_name):
        """获取指定相机的RGB和深度图像"""
        if cam_name not in self.cameras:
            return None, None
        
        try:
            if self.use_realsense:
                # r3kit D415 接口
                color, depth = self.cameras[cam_name].get()
                if color is None or depth is None:
                    return None, None
                # 转 RGB
                frame_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                return frame_rgb, depth
            else:
                # OpenCV 摄像头 - 生成模拟深度图
                ret, frame = self.cameras[cam_name].read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fake_depth = np.ones((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.float32) * 0.5
                    return frame_rgb, fake_depth
                return None, None
        except Exception as e:
            print(f"获取 {cam_name} 图像和深度失败: {e}")
            return None, None
    
    def get_all_images(self):
        """获取所有相机的图像"""
        images = {}
        for cam_name in self.camera_names:
            image = self.get_image(cam_name)
            if image is not None:
                images[cam_name] = image
            else:
                # 生成模拟图像作为 fallback
                images[cam_name] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                print(f"警告: {cam_name} 相机图像获取失败，使用模拟图像")
        
        return images
    
    def close(self):
        """关闭所有相机"""
        for cam_name, cap in self.cameras.items():
            try:
                if self.use_realsense:
                    # D415 类可能没有 stop 方法，使用 __del__ 或者不做任何操作
                    if hasattr(cap, 'stop'):
                        cap.stop()
                    elif hasattr(cap, 'close'):
                        cap.close()
                    # 对于 r3kit D415，通常由析构函数自动处理
                else:
                    cap.release()
                print(f"{cam_name} 已关闭")
            except Exception as e:
                print(f"关闭 {cam_name} 失败: {e}")