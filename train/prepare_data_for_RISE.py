import os
import glob
import shutil
import numpy as np
import open3d as o3d
from PIL import Image
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
INTRINSICS = {
    "1234567890": np.array([[606.268127441406, 0, 319.728454589844, 0],
                              [0, 605.743286132812, 234.524749755859, 0],
                              [0, 0, 1, 0]])
}
WORKSPACE_MIN = np.array([-0.5, -0.5, 0.5])
WORKSPACE_MAX = np.array([0.5, 0.5, 1.2])
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])
start_x = 180
end_x  = 540
start_y = 0
end_y = 360


def copy_angles_rgbs_depths(records_dir, scene_records_dir, origin_copy_dir, output_dir):
    """
    将records_dir和scene_records_dir中的angles、rgbs、depths复制到origin_copy_dir中
    """
    record_dirs = [f for f in os.listdir(records_dir) if os.path.isdir(os.path.join(records_dir, f)) and f.startswith('record_')]
    print(f"[Info] 找到{len(record_dirs)}个record_dir")
    err_num = 0
    for i, record_dir in enumerate(record_dirs):
        # TODO:当前只处理单相机cam_0，后续需要处理多相机
        angles_org_dir = os.path.join(records_dir, record_dir, "angles")
        rgbs_org_dir = os.path.join(scene_records_dir, "cam_0", f"{record_dir}_30000", "renders")
        depths_org_dir = os.path.join(scene_records_dir, "cam_0", f"{record_dir}_30000", "depth")

        # 确保三个文件夹数量一致
        if len(os.listdir(angles_org_dir)) != len(os.listdir(rgbs_org_dir)) or len(os.listdir(angles_org_dir)) != len(os.listdir(depths_org_dir)):
            err_num += 1
            print(f"[Error] {record_dir} 中的angles、rgbs、depths数量不一致")
            continue
        # 复制到origin_copy_dir，这里会自动创建文件夹
        shutil.copytree(angles_org_dir, os.path.join(output_dir, record_dir, "angles"), dirs_exist_ok=True)
        rgb_files = glob.glob(os.path.join(rgbs_org_dir, "*.png"))
        depth_files = glob.glob(os.path.join(depths_org_dir, "*.png"))
        os.makedirs(os.path.join(origin_copy_dir, record_dir, "rgbs"), exist_ok=True)
        os.makedirs(os.path.join(origin_copy_dir, record_dir, "depths"), exist_ok=True)
        for rgb_file, depth_file in zip(rgb_files, depth_files):
            rgb = cv2.imread(rgb_file)
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            # 对rgb和depth进行裁切
            rgb_cropped = rgb[start_y:end_y, start_x:end_x]
            depth_cropped = depth[start_y:end_y, start_x:end_x]
            # 保存到origin_copy_dir
            cv2.imwrite(os.path.join(origin_copy_dir, record_dir, "rgbs", os.path.basename(rgb_file)), rgb_cropped)
            cv2.imwrite(os.path.join(origin_copy_dir, record_dir, "depths", os.path.basename(depth_file)), depth_cropped)

        print(f"[Info] [{i+1}/{len(record_dirs)}] 复制完成")
    if err_num == 0:
        print(f"[Info] 每条记录数据长度一致")
    else:
        raise ValueError(f"[Error] 有 {err_num} 个数据长度不一致")


def create_pointclouds_and_save(origin_copy_dir, output_dir):
    """
    将origin_copy_dir中的angles、rgbs、depths创建点云并保存到output_dir中
    """
    record_dirs = [f for f in os.listdir(origin_copy_dir) if os.path.isdir(os.path.join(origin_copy_dir, f)) and f.startswith('record_')]
    for i, record_dir in enumerate(record_dirs):
        rgbs_org_dir = os.path.join(origin_copy_dir, record_dir, "rgbs")
        depths_org_dir = os.path.join(origin_copy_dir, record_dir, "depths")
        rgb_file_paths = glob.glob(os.path.join(rgbs_org_dir, "*.png"))
        depth_file_paths = glob.glob(os.path.join(depths_org_dir, "*.png"))

        # 创建点云保存目录
        pointcloud_dir = os.path.join(output_dir, record_dir, "pointclouds")
        os.makedirs(pointcloud_dir, exist_ok=True)

        # 对每帧rgb和depth创建对应帧的点云
        for rgb_file_path, depth_file_path in zip(rgb_file_paths, depth_file_paths):
            rgb = np.array(Image.open(rgb_file_path)).astype(np.uint8)
            depth = np.array(Image.open(depth_file_path)).astype(np.float32)
            h, w = depth.shape
            fx, fy = INTRINSICS["1234567890"][0, 0], INTRINSICS["1234567890"][1, 1]
            cx, cy = INTRINSICS["1234567890"][0, 2], INTRINSICS["1234567890"][1, 2]
            scale = 1000.0
            colors = o3d.geometry.Image(rgb.astype(np.uint8))
            depths = o3d.geometry.Image(depth.astype(np.float32))
            camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
            )
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                colors, depths, scale, convert_rgb_to_intensity = False
            )
            cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
            cloud = cloud.voxel_down_sample(0.005)
            # 进行体素过滤
            # cloud = filter_sparse_voxels(cloud,
            #                         voxel_size=0.03,           # 体素大小：5cm
            #                         min_points_per_voxel=5,     # 每个体素最少8个点
            #                         workspace_min=WORKSPACE_MIN,
            #                         workspace_max=WORKSPACE_MAX)
            # cloud = filter_sparse_voxels(cloud,
            #                         voxel_size=0.06,           # 体素大小：5cm
            #                         min_points_per_voxel=15,     # 每个体素最少8个点
            #                         workspace_min=WORKSPACE_MIN,
            #                         workspace_max=WORKSPACE_MAX)
            # cloud = filter_sparse_voxels(cloud,
            #                         voxel_size=0.1,           # 体素大小：5cm
            #                         min_points_per_voxel=20,     # 每个体素最少8个点
            #                         workspace_min=WORKSPACE_MIN,
            #                         workspace_max=WORKSPACE_MAX)
            # 处理点云数据
            points = np.array(cloud.points)
            colors = np.array(cloud.colors)
            # 应用ImageNet归一化
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud.colors = o3d.utility.Vector3dVector(colors)
            cloud.points = o3d.utility.Vector3dVector(points)

            o3d.io.write_point_cloud(os.path.join(pointcloud_dir, os.path.basename(rgb_file_path).replace(".png", ".pcd")), cloud)
        print(f"[Info] [{i+1}/{len(record_dirs)}] 创建点云完成")
    print(f"[Info] 所有点云创建完成")


def filter_sparse_voxels(cloud, voxel_size=0.01, min_points_per_voxel=5, workspace_min=None, workspace_max=None):
    # 获取点云数据
    points = np.array(cloud.points)
    colors = np.array(cloud.colors) if cloud.has_colors() else None

    if len(points) == 0:
        return cloud

    # 应用工作空间裁剪（如果指定）
    if workspace_min is not None and workspace_max is not None:
        x_mask = ((points[:, 0] >= workspace_min[0]) & (points[:, 0] <= workspace_max[0]))
        y_mask = ((points[:, 1] >= workspace_min[1]) & (points[:, 1] <= workspace_max[1]))
        z_mask = ((points[:, 2] >= workspace_min[2]) & (points[:, 2] <= workspace_max[2]))
        mask = (x_mask & y_mask & z_mask)
        points = points[mask]
        if colors is not None:
            colors = colors[mask]

    # 计算体素网格坐标
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)

    # 统计每个体素内的点数
    unique_voxels, counts = np.unique(voxel_coords, axis=0, return_counts=True)

    # 创建体素索引映射
    voxel_to_index = {}
    for i, voxel in enumerate(unique_voxels):
        voxel_key = tuple(voxel)
        voxel_to_index[voxel_key] = i

    # 过滤稀疏体素
    valid_indices = []
    for i, voxel_coord in enumerate(voxel_coords):
        voxel_key = tuple(voxel_coord)
        if voxel_key in voxel_to_index:
            voxel_count = counts[voxel_to_index[voxel_key]]
            if voxel_count >= min_points_per_voxel:
                valid_indices.append(i)

    # 创建过滤后的点云
    filtered_points = points[valid_indices]
    filtered_colors = colors[valid_indices] if colors is not None else None

    # 创建新的点云对象
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors is not None:
        filtered_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_cloud


def main(scene_records_dir, records_dir):
    assert os.path.exists(records_dir), f"records_dir not found: {records_dir}"
    assert os.path.exists(scene_records_dir), f"scene_records_dir not found: {scene_records_dir}"
    # 确保records_name一致
    assert records_dir.split("/")[-1] == scene_records_dir.split("/")[-1], f"records_name not found: {records_name}"
    records_name = records_dir.split("/")[-1]
    Train_Data_records_dir = os.path.join("/media/robotflow/SSDWH/3DGS/Data/Train_Data", f"RISE_{records_name}")
    origin_copy_dir = os.path.join(Train_Data_records_dir, "org")
    output_dir = os.path.join(Train_Data_records_dir, "RISE")

    # 创建Train_Data_records_dir
    if os.path.exists(Train_Data_records_dir):
        raise ValueError(f"Train_Data_records_dir already exists: {Train_Data_records_dir}")
    os.makedirs(origin_copy_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 复制rgbs、depths到origin_copy_dir，复制angles到output_dir
    copy_angles_rgbs_depths(records_dir, scene_records_dir, origin_copy_dir, output_dir)

    # 创建pointclouds并保存到output_dir
    print(f"[Info] 开始创建点云")
    create_pointclouds_and_save(origin_copy_dir, output_dir)

    print(f"[Info] 创建点云完成")
    print("======================== Finished ========================")

if __name__ == "__main__":
    records_dir = "/media/robotflow/SSDWH/3DGS/Data/Origin_Data/records_0929_test"
    scene_records_dir = "/media/robotflow/SSDWH/3DGS/Data/GS_Data/gaussian_splatting_data/scene_0929_test/records_0929_test"
    main(scene_records_dir, records_dir)