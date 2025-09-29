#!/usr/bin/env python3
"""
点云可视化脚本

功能：
- 模式A（默认）：浏览最新时间戳目录下的所有 PLY 文件，按 N/P 切换
- 模式B：指定一个 *_data.npz，一次性显示 raw / proc / final_sparse / cloud_to_policy 四类点云

用法：
  A) 浏览 PLY：
     python scripts/view_pointclouds.py [debug_pointclouds]

  B) 查看 NPZ：
     python scripts/view_pointclouds.py --npz debug_pointclouds/<timestamp>/0000_data.npz

依赖：open3d
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except Exception as e:
    print(f"导入 open3d 失败，请先安装: pip install open3d\n错误: {e}")
    sys.exit(1)


def latest_timestamp_dir(root: Path) -> Path:
    subdirs = sorted([p for p in root.glob("*") if p.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"未在 {root} 下找到任何时间戳目录")
    return subdirs[-1]


def read_ply_list(target_dir: Path):
    plys = sorted(target_dir.glob("*.ply"))
    if not plys:
        raise FileNotFoundError(f"目录 {target_dir} 下没有 ply 文件")
    return plys


def make_pcd(points: np.ndarray, colors: np.ndarray = None):
    if points is None or len(points) == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        cols = np.clip(colors, 0.0, 1.0).astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def view_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    raw_pts = data.get("raw_points")
    raw_cols = data.get("raw_colors")
    proc_pts = data.get("proc_points")
    proc_cols = data.get("proc_colors")
    final_xyz = data.get("final_coords_xyz")
    cloud = data.get("cloud_to_policy")  # (N,6)

    geoms = []
    pcd_raw = make_pcd(raw_pts, raw_cols)
    if pcd_raw is not None:
        geoms.append(pcd_raw)

    if proc_pts is not None and proc_cols is not None:
        geoms.append(make_pcd(proc_pts, np.clip(proc_cols, 0.0, 1.0)))

    if final_xyz is not None:
        # final 使用 proc 的颜色（若长度匹配），否则用白色
        cols = proc_cols if (proc_cols is not None and len(proc_cols) == len(final_xyz)) else np.ones_like(final_xyz)
        geoms.append(make_pcd(final_xyz, np.clip(cols, 0.0, 1.0)))

    if cloud is not None:
        pts = cloud[:, :3]
        cols = cloud[:, 3:]
        # 归一化颜色可能非[0,1]，线性拉伸到[0,1]
        cmin, cmax = cols.min(), cols.max()
        cols = (cols - cmin) / (cmax - cmin + 1e-8)
        geoms.append(make_pcd(pts, np.clip(cols, 0.0, 1.0)))

    geoms = [g for g in geoms if g is not None]
    if not geoms:
        print("NPZ 中没有可视化的点云")
        return

    print("显示顺序: raw, proc, final_sparse, cloud_to_policy (若存在)")
    o3d.visualization.draw_geometries(geoms, window_name=npz_path.name)


def browse_ply(root_dir: Path):
    # 获取最新时间戳目录
    target = latest_timestamp_dir(root_dir)
    plys = read_ply_list(target)
    print(f"显示目录: {target}, PLY 文件数: {len(plys)}")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("PLY Browser")
    geom = None
    idx = 0

    def load_and_show(i):
        nonlocal geom
        pcd = o3d.io.read_point_cloud(str(plys[i]))
        if geom is None:
            geom = pcd
            vis.add_geometry(geom)
        else:
            geom.points = pcd.points
            geom.colors = pcd.colors
            vis.update_geometry(geom)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        print(f"显示: {plys[i].name}")

    def next_cb(_):
        nonlocal idx
        idx = (idx + 1) % len(plys)
        load_and_show(idx)
        return False

    def prev_cb(_):
        nonlocal idx
        idx = (idx - 1) % len(plys)
        load_and_show(idx)
        return False

    vis.register_key_callback(ord('N'), next_cb)
    vis.register_key_callback(ord('P'), prev_cb)

    print("按 N 下一帧，P 上一帧，ESC 退出")
    load_and_show(idx)
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="点云可视化 (PLY 浏览 / NPZ 四视图)")
    parser.add_argument("root", nargs="?", default="debug_pointclouds", help="PLY 根目录 (默认: debug_pointclouds)")
    parser.add_argument("--npz", dest="npz", default=None, help="指定一个 *_data.npz 进行四类点云可视化")
    args = parser.parse_args()

    if args.npz is not None:
        npz_path = Path(args.npz)
        if not npz_path.exists():
            print(f"NPZ 文件不存在: {npz_path}")
            return 1
        view_npz(npz_path)
        return 0

    root_dir = Path(args.root)
    if not root_dir.exists():
        print(f"根目录不存在: {root_dir}")
        return 1
    browse_ply(root_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())


