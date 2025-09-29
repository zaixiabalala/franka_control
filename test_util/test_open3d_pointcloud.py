#!/usr/bin/env python3
"""
独立的Open3D点云生成测试脚本

用于测试Open3D库兼容性和点云生成功能，独立于机械臂和模型
"""

import os
import sys
import numpy as np
import cv2
import time
from pathlib import Path

# 添加项目路径 - 按照1.py中的顺序
project_dir = Path(__file__).parent.parent
model_lerobot_path = project_dir / "model" / "lerobot" / "src"
r3kit_path = project_dir / "model" / "r3kit"
sys.path.insert(0, str(model_lerobot_path))
sys.path.insert(0, str(r3kit_path))  # 添加r3kit路径
sys.path.insert(0, str(project_dir))  # 添加项目根目录到路径

# 导入RISE相关常量
rise_path = project_dir / "RISE"
sys.path.insert(0, str(rise_path))

# 导入相机相关模块
import pyrealsense2 as rs
from r3kit.devices.camera.realsense import config as rs_cfg
from r3kit.devices.camera.realsense.d415 import D415

try:
    import open3d as o3d
    from utils.constants import IMG_MEAN, IMG_STD, WORKSPACE_MIN, WORKSPACE_MAX
    print("✅ 成功导入Open3D和RISE常量")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 相机配置
FPS = 30
D415_CAMERAS = {   
    "cam4": "327322062498",  # 固定机位视角
}

# 相机内参（与1.py中保持一致）
CAM_INTRINSICS = np.array([[606.268127441406, 0, 319.728454589844, 0],
                          [0, 605.743286132812, 234.524749755859, 0],
                          [0, 0, 1, 0]])

class Open3DTester:
    """Open3D点云生成测试器"""
    
    def __init__(self):
        self.camera = None
        self.test_results = {}
        
    def test_open3d_basic(self):
        """测试Open3D基本功能"""
        print("\n=== 测试Open3D基本功能 ===")
        
        try:
            # 测试1: 创建基本几何体
            print("1. 测试创建基本几何体...")
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            print(f"   ✅ 球体创建成功，顶点数: {len(sphere.vertices)}")
            
            # 测试2: 创建点云
            print("2. 测试创建点云...")
            pcd = o3d.geometry.PointCloud()
            points = np.random.rand(1000, 3)
            pcd.points = o3d.utility.Vector3dVector(points)
            print(f"   ✅ 点云创建成功，点数: {len(pcd.points)}")
            
            # 测试3: 体素下采样
            print("3. 测试体素下采样...")
            voxel_size = 0.005
            downsampled = pcd.voxel_down_sample(voxel_size)
            print(f"   ✅ 体素下采样成功，原始点数: {len(pcd.points)}, 下采样后: {len(downsampled.points)}")
            
            self.test_results['open3d_basic'] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Open3D基本功能测试失败: {e}")
            self.test_results['open3d_basic'] = False
            return False
    
    def test_open3d_image_creation(self):
        """测试Open3D图像创建功能"""
        print("\n=== 测试Open3D图像创建功能 ===")
        
        try:
            # 测试1: 创建彩色图像
            print("1. 测试创建彩色图像...")
            color_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            color_image = o3d.geometry.Image(color_array)
            print(f"   ✅ 彩色图像创建成功，尺寸: {color_array.shape[1]}x{color_array.shape[0]}")
            
            # 测试2: 创建深度图像
            print("2. 测试创建深度图像...")
            depth_array = np.random.uniform(0.3, 1.0, (480, 640)).astype(np.float32)
            depth_image = o3d.geometry.Image(depth_array)
            print(f"   ✅ 深度图像创建成功，尺寸: {depth_array.shape[1]}x{depth_array.shape[0]}")
            
            # 测试3: 创建相机内参
            print("3. 测试创建相机内参...")
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=640, height=480,
                fx=CAM_INTRINSICS[0, 0], fy=CAM_INTRINSICS[1, 1],
                cx=CAM_INTRINSICS[0, 2], cy=CAM_INTRINSICS[1, 2]
            )
            print(f"   ✅ 相机内参创建成功，fx={intrinsic.intrinsic_matrix[0,0]}, fy={intrinsic.intrinsic_matrix[1,1]}")
            
            self.test_results['open3d_image'] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Open3D图像创建测试失败: {e}")
            self.test_results['open3d_image'] = False
            return False
    
    def test_rgbd_to_pointcloud(self):
        """测试RGB-D图像转点云功能"""
        print("\n=== 测试RGB-D图像转点云功能 ===")
        
        try:
            # 创建测试数据
            print("1. 创建测试RGB-D数据...")
            color_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth_array = np.random.uniform(0.3, 1.0, (480, 640)).astype(np.float32)
            
            # 创建Open3D图像对象
            color_image = o3d.geometry.Image(color_array)
            depth_image = o3d.geometry.Image(depth_array)
            
            # 创建相机内参
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=640, height=480,
                fx=CAM_INTRINSICS[0, 0], fy=CAM_INTRINSICS[1, 1],
                cx=CAM_INTRINSICS[0, 2], cy=CAM_INTRINSICS[1, 2]
            )
            
            print(f"   ✅ 测试数据创建成功，颜色图像: {color_array.shape[1]}x{color_array.shape[0]}, 深度图像: {depth_array.shape[1]}x{depth_array.shape[0]}")
            
            # 测试2: 创建RGBD图像
            print("2. 测试创建RGBD图像...")
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image, depth_scale=1.0, convert_rgb_to_intensity=False
            )
            print(f"   ✅ RGBD图像创建成功，颜色: {color_array.shape[1]}x{color_array.shape[0]}, 深度: {depth_array.shape[1]}x{depth_array.shape[0]}")
            
            # 测试3: 从RGBD创建点云
            print("3. 测试从RGBD创建点云...")
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            print(f"   ✅ 点云创建成功，点数: {len(point_cloud.points)}")
            
            # 测试4: 体素下采样
            print("4. 测试体素下采样...")
            voxel_size = 0.005
            downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)
            print(f"   ✅ 体素下采样成功，原始点数: {len(point_cloud.points)}, 下采样后: {len(downsampled_cloud.points)}")
            
            self.test_results['rgbd_to_pointcloud'] = True
            return True
            
        except Exception as e:
            print(f"   ❌ RGB-D转点云测试失败: {e}")
            self.test_results['rgbd_to_pointcloud'] = False
            return False
    
    def test_camera_acquisition(self):
        """测试相机数据获取"""
        print("\n=== 测试相机数据获取 ===")
        
        try:
            # 配置流
            rs_cfg.D415_STREAMS = [
                (rs.stream.depth, 640, 480, rs.format.z16, FPS),
                (rs.stream.color, 640, 480, rs.format.bgr8, FPS),
            ]
            
            # 初始化相机
            print("1. 初始化RealSense D415相机...")
            serial = D415_CAMERAS["cam4"]
            camera = D415(id=serial, depth=True, name="cam4")
            print(f"   ✅ 相机初始化成功，序列号: {serial}")
            
            # 获取图像
            print("2. 获取RGB-D图像...")
            color, depth = camera.get()
            
            if color is None or depth is None:
                print("   ❌ 相机图像获取失败")
                self.test_results['camera_acquisition'] = False
                return False
            
            print(f"   ✅ 图像获取成功，颜色: {color.shape}, 深度: {depth.shape}")
            
            # 转换颜色格式
            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            print(f"   ✅ 颜色格式转换成功，RGB形状: {color_rgb.shape}")
            
            # 测试深度图数据类型和范围
            print(f"3. 深度图分析:")
            print(f"   数据类型: {depth.dtype}")
            print(f"   最小值: {depth.min()}")
            print(f"   最大值: {depth.max()}")
            print(f"   平均值: {depth.mean():.2f}")
            print(f"   非零像素数: {np.count_nonzero(depth)}")
            
            self.camera = camera
            self.test_results['camera_acquisition'] = True
            return True
            
        except Exception as e:
            print(f"   ❌ 相机数据获取测试失败: {e}")
            self.test_results['camera_acquisition'] = False
            return False
    
    def test_real_camera_pointcloud(self):
        """使用真实相机数据测试点云生成"""
        print("\n=== 使用真实相机数据测试点云生成 ===")
        
        if not self.camera:
            print("   ❌ 相机未初始化，跳过测试")
            return False
        
        try:
            # 获取图像
            print("1. 获取真实RGB-D图像...")
            color, depth = self.camera.get()
            
            if color is None or depth is None:
                print("   ❌ 真实图像获取失败")
                return False
            
            # 转换颜色格式
            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            print(f"   ✅ 真实图像获取成功，颜色: {color_rgb.shape}, 深度: {depth.shape}")
            
            # 深度图预处理
            print("2. 深度图预处理...")
            # 将深度图从uint16转换为float32，并转换为米单位
            depth_float = depth.astype(np.float32) / 1000.0  # 假设原始单位是毫米
            print(f"   转换后深度范围: {depth_float.min():.3f} - {depth_float.max():.3f} 米")
            
            # 创建Open3D图像
            print("3. 创建Open3D图像对象...")
            color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
            depth_o3d = o3d.geometry.Image(depth_float)
            
            # 创建相机内参
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=640, height=480,
                fx=CAM_INTRINSICS[0, 0], fy=CAM_INTRINSICS[1, 1],
                cx=CAM_INTRINSICS[0, 2], cy=CAM_INTRINSICS[1, 2]
            )
            
            # 创建RGBD图像
            print("4. 创建RGBD图像...")
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
            )
            
            # 创建点云
            print("5. 创建点云...")
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            print(f"   ✅ 原始点云创建成功，点数: {len(point_cloud.points)}")
            
            if len(point_cloud.points) == 0:
                print("   ❌ 点云为空！")
                self.test_results['real_pointcloud'] = False
                return False
            
            # 体素下采样
            print("6. 体素下采样...")
            voxel_size = 0.005
            downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)
            print(f"   ✅ 体素下采样成功，下采样后点数: {len(downsampled_cloud.points)}")
            
            # 工作空间裁剪测试
            print("7. 工作空间裁剪测试...")
            points = np.array(point_cloud.points)
            colors = np.array(point_cloud.colors)
            
            print(f"   原始点数: {len(points)}")
            print(f"   工作空间范围: MIN={WORKSPACE_MIN}, MAX={WORKSPACE_MAX}")
            
            # 显示点云范围
            if len(points) > 0:
                print(f"   点云X范围: {points[:, 0].min():.3f} - {points[:, 0].max():.3f}")
                print(f"   点云Y范围: {points[:, 1].min():.3f} - {points[:, 1].max():.3f}")
                print(f"   点云Z范围: {points[:, 2].min():.3f} - {points[:, 2].max():.3f}")
                
                # 工作空间裁剪
                x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
                y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
                z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
                mask = (x_mask & y_mask & z_mask)
                
                cropped_points = points[mask]
                cropped_colors = colors[mask]
                print(f"   裁剪后点数: {len(cropped_points)}")
                
                if len(cropped_points) == 0:
                    print("   ⚠️  工作空间裁剪后点云为空！")
                    print("   建议检查工作空间范围设置")
                else:
                    print("   ✅ 工作空间裁剪成功")
            
            self.test_results['real_pointcloud'] = True
            return True
            
        except Exception as e:
            print(f"   ❌ 真实相机点云生成测试失败: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['real_pointcloud'] = False
            return False
    
    def test_simulated_data_pointcloud(self):
        """使用模拟数据测试点云生成（模拟1.py中的场景）"""
        print("\n=== 使用模拟数据测试点云生成 ===")
        
        try:
            # 模拟1.py中的数据生成
            print("1. 生成模拟RGB-D数据...")
            color_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth_img = np.ones((480, 640), dtype=np.float32) * 0.5  # 模拟深度图
            
            print(f"   ✅ 模拟数据生成成功，颜色: {color_img.shape}, 深度: {depth_img.shape}")
            
            # 模拟1.py中的预处理
            print("2. 模拟图像预处理...")
            start_w, end_w = 200, 560
            start_h, end_h = 0, 360
            
            cropped_rgb = color_img[start_h:end_h, start_w:end_w]
            cropped_depth = depth_img[start_h:end_h, start_w:end_w]
            
            print(f"   ✅ 图像裁剪成功，裁剪后: {cropped_rgb.shape}, {cropped_depth.shape}")
            
            # 创建Open3D图像对象
            print("3. 创建Open3D图像对象...")
            color_o3d = o3d.geometry.Image(cropped_rgb.astype(np.uint8))
            depth_o3d = o3d.geometry.Image(cropped_depth.astype(np.float32))
            
            # 创建相机内参（注意：这里使用的是裁剪后的尺寸）
            print("4. 创建相机内参...")
            # 这里需要调整内参以适应裁剪后的图像
            fx, fy = CAM_INTRINSICS[0, 0], CAM_INTRINSICS[1, 1]
            cx, cy = CAM_INTRINSICS[0, 2], CAM_INTRINSICS[1, 2]
            
            # 调整内参以适应裁剪
            cx_adjusted = cx - start_w
            cy_adjusted = cy - start_h
            
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=end_w-start_w, height=end_h-start_h,
                fx=fx, fy=fy, cx=cx_adjusted, cy=cy_adjusted
            )
            
            print(f"   调整后内参: fx={fx}, fy={fy}, cx={cx_adjusted}, cy={cy_adjusted}")
            
            # 创建RGBD图像
            print("5. 创建RGBD图像...")
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
            )
            
            # 创建点云
            print("6. 创建点云...")
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            print(f"   ✅ 模拟点云创建成功，点数: {len(point_cloud.points)}")
            
            if len(point_cloud.points) == 0:
                print("   ❌ 模拟点云为空！")
                self.test_results['simulated_pointcloud'] = False
                return False
            
            # 体素下采样
            print("7. 体素下采样...")
            voxel_size = 0.005
            downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)
            print(f"   ✅ 体素下采样成功，下采样后点数: {len(downsampled_cloud.points)}")
            
            # 工作空间裁剪
            print("8. 工作空间裁剪...")
            points = np.array(point_cloud.points)
            colors = np.array(point_cloud.colors)
            
            if len(points) > 0:
                x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
                y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
                z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
                mask = (x_mask & y_mask & z_mask)
                
                cropped_points = points[mask]
                print(f"   裁剪后点数: {len(cropped_points)}")
                
                if len(cropped_points) == 0:
                    print("   ⚠️  工作空间裁剪后点云为空！")
                else:
                    print("   ✅ 工作空间裁剪成功")
            
            self.test_results['simulated_pointcloud'] = True
            return True
            
        except Exception as e:
            print(f"   ❌ 模拟数据点云生成测试失败: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['simulated_pointcloud'] = False
            return False
    
    def print_version_info(self):
        """打印版本信息"""
        print("\n=== 版本信息 ===")
        try:
            import open3d as o3d
            print(f"Open3D版本: {o3d.__version__}")
        except:
            print("Open3D版本: 无法获取")
        
        try:
            import numpy as np
            print(f"NumPy版本: {np.__version__}")
        except:
            print("NumPy版本: 无法获取")
        
        try:
            import cv2
            print(f"OpenCV版本: {cv2.__version__}")
        except:
            print("OpenCV版本: 无法获取")
        
        try:
            import torch
            print(f"PyTorch版本: {torch.__version__}")
        except:
            print("PyTorch版本: 无法获取")
        
        print(f"Python版本: {sys.version}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🔍 开始Open3D点云生成兼容性测试...")
        print("=" * 60)
        
        # 打印版本信息
        self.print_version_info()
        
        # 运行测试
        tests = [
            ("Open3D基本功能", self.test_open3d_basic),
            ("Open3D图像创建", self.test_open3d_image_creation),
            ("RGB-D转点云", self.test_rgbd_to_pointcloud),
            ("相机数据获取", self.test_camera_acquisition),
            ("真实相机点云", self.test_real_camera_pointcloud),
            ("模拟数据点云", self.test_simulated_data_pointcloud),
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"❌ {test_name}测试异常: {e}")
                self.test_results[test_name.lower().replace(" ", "_")] = False
        
        # 输出测试结果
        self.print_test_results()
        
        # 清理资源
        if self.camera:
            try:
                if hasattr(self.camera, 'stop'):
                    self.camera.stop()
                elif hasattr(self.camera, 'close'):
                    self.camera.close()
            except:
                pass
    
    def print_test_results(self):
        """打印测试结果"""
        print("\n" + "=" * 60)
        print("📊 测试结果汇总")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name:30} {status}")
        
        # 分析问题
        print("\n🔍 问题分析:")
        failed_tests = [name for name, result in self.test_results.items() if not result]
        
        if not failed_tests:
            print("✅ 所有测试都通过了！Open3D库工作正常。")
            print("💡 建议检查1.py中的具体实现细节。")
        else:
            print(f"❌ 有 {len(failed_tests)} 个测试失败:")
            for test in failed_tests:
                print(f"   - {test}")
            
            if 'open3d_basic' in failed_tests or 'open3d_image' in failed_tests:
                print("\n🚨 主要问题：Open3D库兼容性问题")
                print("💡 建议解决方案:")
                print("   1. 重新安装Open3D: pip uninstall open3d && pip install open3d")
                print("   2. 检查NumPy版本兼容性")
                print("   3. 尝试安装特定版本: pip install open3d==0.17.0")
            
            if 'camera_acquisition' in failed_tests:
                print("\n🚨 相机获取问题")
                print("💡 建议解决方案:")
                print("   1. 检查相机连接")
                print("   2. 检查r3kit库安装")
                print("   3. 检查RealSense SDK")
            
            if 'real_pointcloud' in failed_tests and 'rgbd_to_pointcloud' not in failed_tests:
                print("\n🚨 真实相机点云生成问题")
                print("💡 可能原因:")
                print("   1. 深度图数据格式问题")
                print("   2. 相机内参不匹配")
                print("   3. 工作空间范围设置问题")


def main():
    """主函数"""
    print("🧪 Open3D点云生成兼容性测试工具")
    print("=" * 60)
    
    tester = Open3DTester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
