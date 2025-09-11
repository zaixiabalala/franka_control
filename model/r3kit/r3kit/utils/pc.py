from typing import Tuple, Dict
import numpy as np
import open3d as o3d
import yourdfpy


def voxelize(pc_xyz:np.ndarray, pc_rgb:np.ndarray, voxel_size:float) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_pc_xyz = np.asarray(downsampled_pcd.points)
    downsampled_pc_rgb = np.asarray(downsampled_pcd.colors)
    return (downsampled_pc_xyz, downsampled_pc_rgb)

def farthest_point_sample(point:np.ndarray, npoint:int) -> Tuple[np.ndarray, np.ndarray]:
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    centroids = centroids.astype(np.int32)
    point = point[centroids]
    return (point, centroids)


def mesh2pc(obj_path:str, num_points:int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(obj_path)
    pc = mesh.sample_points_uniformly(number_of_points=num_points)
    pc = np.asarray(pc.points)
    return pc

def urdf2pc(urdf_path:str, joints:Dict[str, float], num_points:int) -> np.ndarray:
    urdf = yourdfpy.URDF.load(urdf_path)
    urdf.update_cfg(joints)
    mesh = urdf.scene.to_mesh()
    geometry = o3d.geometry.TriangleMesh()
    geometry.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    geometry.triangles = o3d.utility.Vector3iVector(mesh.faces)
    pc = geometry.sample_points_uniformly(number_of_points=num_points)
    pc = np.asarray(pc.points)
    return pc
