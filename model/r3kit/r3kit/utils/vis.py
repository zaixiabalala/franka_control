import os
from typing import List, Dict, Optional
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import yourdfpy
import concurrent
import concurrent.futures
from pynput import keyboard


def vis_pc(xyz:np.ndarray, rgb:Optional[np.ndarray]=None, show_frame:bool=True) -> None:
    '''
    xyz: (N, 3) in meter in camera frame
    rgb: (N, 3) in [0, 1]
    '''
    geometries = []
    if show_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries.append(frame)
    else:
        pass
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        pass
    geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries)


def rotation_vec2mat(vec:np.ndarray) -> np.ndarray:
    mat = np.zeros((3, 3))
    temp2 = np.cross(vec, np.array([1., 0., 0.]))
    if np.linalg.norm(temp2) < 1e-3:
        temp1 = np.cross(np.array([0., 1., 0.]), vec)
        temp1 /= np.linalg.norm(temp1)
        temp2 = np.cross(vec, temp1)
        temp2 /= np.linalg.norm(temp2)
    else:
        temp2 /= np.linalg.norm(temp2)
        temp1 = np.cross(temp2, vec)
        temp1 /= np.linalg.norm(temp1)
    mat[:, 0] = temp1
    mat[:, 1] = temp2
    mat[:, 2] = vec
    return mat


def draw_time(timestamps:List[float], path:str) -> None:
    num = len(timestamps) - 1
    x = list(range(num))
    y = [timestamps[idx+1] - timestamps[idx] for idx in range(num)]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('data')
    plt.ylabel('time')
    plt.savefig(path)

def draw_items(items:np.ndarray, path:str) -> None:
    assert len(items.shape) == 1 or len(items.shape) == 2
    if len(items.shape) == 1:
        T = items.shape[0]
        x = list(range(T))
        y = items
        plt.figure()
        plt.plot(x, y)
        plt.xlabel('time')
        plt.ylabel('value')
        plt.savefig(path)
    else:
        plt.figure()
        T, N = items.shape
        x = list(range(T))
        if N <= 3:
            for i in range(N):
                plt.subplot(1, N, i+1)
                plt.plot(x, items[:, i])
                plt.xlabel('time')
                plt.ylabel(f'value_{i}')
        else:
            for i in range(N):
                plt.subplot(int(np.ceil(N/3)), 3, i+1)
                plt.plot(x, items[:, i])
                plt.xlabel('time')
                plt.ylabel(f'value_{i}')
        plt.savefig(path)


def save_img(idx:int, path:str, frame:np.ndarray, suffix:str='png', normalize:bool=False, idx_bias:int=0) -> None:
    if normalize:
        cv2.imwrite(os.path.join(path, f"{str(idx+idx_bias).zfill(16)}.{suffix}"), ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8))
    else:
        cv2.imwrite(os.path.join(path, f"{str(idx+idx_bias).zfill(16)}.{suffix}"), frame)

def save_imgs(path:str, frame_list:List[np.ndarray], suffix:str='png', normalize:bool=False, idx_bias:int=0) -> None:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                save_img,
                idx,
                path,
                frame_list[idx],
                suffix,
                normalize,
                idx_bias
            )
            for idx in range(len(frame_list))
        ]

        for future in concurrent.futures.as_completed(futures):
            pass

def save_video(path:str, frame_list:List[np.ndarray], fps:int=30) -> None:
    height, width, _ = frame_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frame_list:
        out.write(frame)
    out.release()


class Sequence1DVisualizer:
    def __init__(self, width:int=640, height:int=480, left:int=0, top:int=0) -> None:
        self.items = {}
        self.width = width
        self.height = height
        self.left = left
        self.top = top
        self.left_init = left
        self.top_init = top
        self.colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:brown', 'tab:gray']
        plt.ion()
    
    def __del__(self) -> None:
        self.stop()
    
    def stop(self) -> None:
        plt.ioff()
        plt.close()
    
    def clear(self) -> None:
        self.items.clear()
        self.left = self.left_init
        self.top = self.top_init

    def update_item(self, name:str, item:np.ndarray, index:Optional[int]=None, zero:bool=False) -> None:
        assert len(item.shape) == 1
        if name in self.items:
            x = self.items[name]['x']
            ys = self.items[name]['ys']
            lines, fig, ax = self.items[name]['object']
            if index is None:
                index = 0 if len(x) == 0 else x[-1] + 1
            if len(x) == 0 or index > x[-1]:
                x.append(index)
                for i in range(item.shape[0]):
                    ys[i].append(item[i])
                    lines[i].set_xdata(x)
                    if zero:
                        lines[i].set_ydata(np.array(ys[i]) - ys[i][0])
                    else:
                        lines[i].set_ydata(ys[i])
            else:
                end_idx = np.where(np.array(x) <= index)[0][-1]
                x = x[:end_idx+1]
                for i in range(item.shape[0]):
                    ys[i] = ys[i][:end_idx+1]
                    ys[i][end_idx] = item[i]
                    lines[i].set_xdata(x)
                    if zero:
                        lines[i].set_ydata(np.array(ys[i]) - ys[i][0])
                    else:
                        lines[i].set_ydata(ys[i])
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            self.items[name]['x'] = x
            self.items[name]['ys'] = ys
        else:
            fig, ax = plt.subplots(figsize=(self.width/100, self.height/100), dpi=100)
            fig.canvas.manager.set_window_title(name)
            fig.canvas.manager.window.wm_geometry(f"+{self.left}+{self.top}")
            self.left += self.width
            x = []
            ys = [[] for _ in range(item.shape[0])]
            lines = []
            for i in range(item.shape[0]):
                line, = ax.plot([], [], color=self.colors[i % len(self.colors)], linestyle='-', label=f'{i}')
                lines.append(line)
            ax.legend()
            self.items[name] = {'object': (lines, fig, ax), 'x': x, 'ys': ys}
            self.update_item(name, item)


class Sequence2DVisualizer:
    def __init__(self, left:int=0, top:int=0) -> None:
        self.names = []
        self.left = left
        self.top = top
        self.left_init = left
        self.top_init = top
    
    def __del__(self) -> None:
        self.stop()
    
    def stop(self) -> None:
        cv2.destroyAllWindows()
    
    def clear(self) -> None:
        cv2.destroyAllWindows()
        self.names.clear()
        self.left = self.left_init
        self.top = self.top_init
    
    def update_image(self, name:str, image:np.ndarray, type:str, scale:float=1.0, **kwargs) -> None:
        if type == 'rgb':
            assert len(image.shape) == 3 and image.shape[2] == 3
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif type == 'bgr':
            assert len(image.shape) == 3 and image.shape[2] == 3
        elif type == 'gray':
            assert len(image.shape) == 2
        elif type == 'depth':
            assert len(image.shape) == 2
            image = (np.clip(image / kwargs['depth_max'], 0, 1) * 255).astype(np.uint8)
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        else:
            raise NotImplementedError
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_NEAREST if type == 'depth' else cv2.INTER_AREA)
        
        if name not in self.names:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, image)
        cv2.waitKey(1)
        if name not in self.names:
            cv2.moveWindow(name, self.left, self.top)
            x, y, width, height = cv2.getWindowImageRect(name)
            self.left += width
            self.names.append(name)


class Sequence3DVisualizer:
    def __init__(self, name:str='3D', width:int=1280, height:int=720, left:int=0, top:int=0) -> None:
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(width=width, height=height, left=left, top=top, visible=True, window_name=name)

        self.frames = {}
        self.arrows = {}
        self.points = {}
        self.meshs = {}
        self.urdfs = {}
        self.c2w = None
    
    def __del__(self) -> None:
        self.stop()
    
    def stop(self) -> None:
        self.visualizer.destroy_window()
    
    def clear(self) -> None:
        self.visualizer.clear_geometries()
        self.frames.clear()
        self.arrows.clear()
        self.points.clear()
        self.meshs.clear()
        self.urdfs.clear()
        self.c2w = None
    
    def update_frame(self, name:str, pose:Optional[np.ndarray]=None, size:Optional[float]=None) -> None:
        if name in self.frames:
            last_pose = self.frames[name]['pose']
            last_size = self.frames[name]['size']
            frame = self.frames[name]['object']

            if not (size is None or np.allclose(size, last_size)):
                frame.scale(size / last_size, center=last_pose[:3, 3])
                self.frames[name]['size'] = size
            if not (pose is None or np.allclose(pose, last_pose)):
                frame.transform(pose @ np.linalg.inv(last_pose))
                self.frames[name]['pose'] = pose
            
            self.visualizer.update_geometry(frame)
        else:
            assert pose is not None and size is not None
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
            frame.transform(pose)
            self.frames[name] = {'object': frame, 'pose': pose, 'size': size}
            self.visualizer.add_geometry(frame)
    
    def update_arrow(self, name:str, offset:Optional[np.ndarray]=None, direction:Optional[np.ndarray]=None, size:Optional[float]=None, color:Optional[np.ndarray]=None) -> None:
        if name in self.arrows:
            last_offset = self.arrows[name]['offset']
            last_direction = self.arrows[name]['direction']
            last_size = self.arrows[name]['size']
            last_color = self.arrows[name]['color']
            arrow = self.arrows[name]['object']

            arrow.translate(-last_offset)
            arrow.rotate(np.linalg.inv(rotation_vec2mat(last_direction)), center=[0, 0, 0])
            arrow.scale(1 / last_size, center=[0, 0, 0])

            if size is None:
                size = last_size
            arrow.scale(size, center=[0, 0, 0])
            if direction is None:
                direction = last_direction
            arrow.rotate(rotation_vec2mat(direction), center=[0, 0, 0])
            if offset is None:
                offset = last_offset
            arrow.translate(offset)
            if color is None:
                color = last_color
            arrow.paint_uniform_color(color)

            self.arrows[name]['offset'] = offset
            self.arrows[name]['direction'] = direction
            self.arrows[name]['size'] = size
            self.arrows[name]['color'] = color
            self.visualizer.update_geometry(arrow)
        else:
            assert offset is not None and direction is not None and size is not None and color is not None
            arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.025, cone_radius=0.05, cylinder_height=0.875, cone_height=0.125, 
                                                           resolution=20, cylinder_split=4, cone_split=1)
            arrow.paint_uniform_color(color)
            arrow.scale(size, center=[0, 0, 0])
            arrow.rotate(rotation_vec2mat(direction), center=[0, 0, 0])
            arrow.translate(offset)
            self.arrows[name] = {'object': arrow, 'offset': offset, 'direction': direction, 'size': size, 'color': color}
            self.visualizer.add_geometry(arrow)
    
    def update_points(self, name:str, xyzs:Optional[np.ndarray]=None, rgbs:Optional[np.ndarray]=None) -> None:
        if name in self.points:
            if xyzs is not None:
                points = self.points[name]['object']
                points.points = o3d.utility.Vector3dVector(xyzs)
            if rgbs is not None:
                points = self.points[name]['object']
                points.colors = o3d.utility.Vector3dVector(rgbs)
            self.visualizer.update_geometry(points)
        else:
            assert xyzs is not None and rgbs is not None
            points = o3d.geometry.PointCloud()
            points.points = o3d.utility.Vector3dVector(xyzs)
            points.colors = o3d.utility.Vector3dVector(rgbs)
            self.points[name] = {'object': points}
            self.visualizer.add_geometry(points)
    
    def update_mesh(self, name:str, path:Optional[str]=None, pose:Optional[np.ndarray]=None, scale:Optional[float]=None, color:Optional[np.ndarray]=None) -> None:
        if name in self.meshs:
            last_path = self.meshs[name]['path']
            last_pose = self.meshs[name]['pose']
            last_scale = self.meshs[name]['scale']
            last_color = self.meshs[name]['color']
            mesh = self.meshs[name]['object']

            if not (path is None or path == last_path):
                self.meshs[name]['path'] = path
                self.visualizer.remove_geometry(mesh)
                mesh = o3d.io.read_triangle_mesh(path)
                mesh.scale(last_scale, center=[0, 0, 0])
                mesh.transform(last_pose)
                self.meshs[name]['object'] = mesh
                self.visualizer.add_geometry(mesh)
            if not (scale is None or np.allclose(scale, last_scale)):
                mesh.scale(scale / last_scale, center=last_pose[:3, 3])
                self.meshs[name]['scale'] = scale
            if not (pose is None or np.allclose(pose, last_pose)):
                mesh.transform(pose @ np.linalg.inv(last_pose))
                self.meshs[name]['pose'] = pose
            if not (color is None or np.allclose(color, last_color)):
                mesh.paint_uniform_color(color)
                self.meshs[name]['color'] = color
            
            self.visualizer.update_geometry(mesh)
        else:
            assert path is not None and pose is not None and scale is not None
            mesh = o3d.io.read_triangle_mesh(path)
            mesh.scale(scale, center=[0, 0, 0])
            mesh.transform(pose)
            if color is not None:
                mesh.paint_uniform_color(color)
            self.meshs[name] = {'object': mesh, 'path':path, 'pose': pose, 'scale': scale, 'color': color}
            self.visualizer.add_geometry(mesh)
    
    def update_urdf(self, name:str, path:Optional[str]=None, joints:Optional[Dict[str, float]]=None, pose:Optional[np.ndarray]=None, scale:Optional[float]=None, color:Optional[np.ndarray]=None) -> None:
        if name in self.urdfs:
            last_path = self.urdfs[name]['path']
            last_pose = self.urdfs[name]['pose']
            last_scale = self.urdfs[name]['scale']
            last_color = self.urdfs[name]['color']
            urdf = self.urdfs[name]['object'][0]
            geometry = self.urdfs[name]['object'][1]

            if not (path is None or path == last_path):
                self.urdfs[name]['path'] = path
                self.visualizer.remove_geometry(geometry)
                urdf = yourdfpy.URDF.load(path)
                if joints is not None:
                    urdf.update_cfg(joints)
                mesh = urdf.scene.to_mesh()
                geometry = o3d.geometry.TriangleMesh()
                geometry.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                geometry.triangles = o3d.utility.Vector3iVector(mesh.faces)
                geometry.scale(last_scale, center=[0, 0, 0])
                geometry.transform(last_pose)
                self.urdfs[name]['object'] = (urdf, geometry)
                self.visualizer.add_geometry(geometry)
            if joints is not None:
                urdf.update_cfg(joints)
                mesh = urdf.scene.to_mesh()
                geometry.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                geometry.triangles = o3d.utility.Vector3iVector(mesh.faces)
                geometry.scale(last_scale, center=[0, 0, 0])
                geometry.transform(last_pose)
            if not (scale is None or np.allclose(scale, last_scale)):
                geometry.scale(scale / last_scale, center=last_pose[:3, 3])
                self.urdfs[name]['scale'] = scale
            if not (pose is None or np.allclose(pose, last_pose)):
                geometry.transform(pose @ np.linalg.inv(last_pose))
                self.urdfs[name]['pose'] = pose
            if not (color is None or np.allclose(color, last_color)):
                geometry.paint_uniform_color(color)
                self.urdfs[name]['color'] = color
            
            self.visualizer.update_geometry(geometry)
        else:
            assert path is not None and pose is not None and scale is not None
            urdf = yourdfpy.URDF.load(path)
            if joints is not None:
                urdf.update_cfg(joints)
            mesh = urdf.scene.to_mesh()
            geometry = o3d.geometry.TriangleMesh()
            geometry.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            geometry.triangles = o3d.utility.Vector3iVector(mesh.faces)
            geometry.scale(scale, center=[0, 0, 0])
            geometry.transform(pose)
            if color is not None:
                geometry.paint_uniform_color(color)
            self.urdfs[name] = {'object': (urdf, geometry), 'path': path, 'pose': pose, 'scale': scale, 'color': color}
            self.visualizer.add_geometry(geometry)
    
    def update_view(self, c2w:Optional[np.ndarray]=None, enforce:bool=False) -> None:
        if c2w is None or ((self.c2w is not None and np.allclose(c2w, self.c2w)) and (not enforce)):
            pass
        else:
            view_control = self.visualizer.get_view_control()
            params = view_control.convert_to_pinhole_camera_parameters()
            params.extrinsic = np.linalg.inv(c2w)
            view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
            self.c2w = c2w

        self.visualizer.poll_events()
        self.visualizer.update_renderer()


class SequenceKeyboardListener:
    def __init__(self, verbose:bool=True) -> None:
        self._verbose = verbose
        self.quit = False
        self.pause = False
        self.reset = False
        self.zero = False
        self.forward = False
        self.backward = False
        self.speed = 1
        self.save = False
        self.more = []

        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
    
    def __del__(self) -> None:
        self.stop()
    
    def stop(self) -> None:
        self.listener.stop()
    
    def _on_press(self, key:keyboard.Key) -> None:
        if key == keyboard.Key.esc:
            self.quit = True
            if self._verbose:
                print('Quit')
        elif key == keyboard.Key.space:
            self.pause = not self.pause
            if self._verbose:
                print('Pause' if self.pause else 'Resume')
        elif key == keyboard.Key.ctrl_r:
            self.reset = True
            if self._verbose:
                print('Reset')
        elif key == keyboard.Key.backspace:
            self.zero = True
            if self._verbose:
                print('Zero')
        elif key == keyboard.Key.shift_r:
            self.save = True
            if self._verbose:
                print('Save')
        elif key == keyboard.Key.right:
            self.forward = True
            if self._verbose:
                print('Forward')
        elif key == keyboard.Key.left:
            self.backward = True
            if self._verbose:
                print('Backward')
        elif key == keyboard.Key.up:
            self.speed *= 2
            if self._verbose:
                print(f'Speed: {self.speed}')
        elif key == keyboard.Key.down:
            self.speed //= 2
            self.speed = max(1, self.speed)
            if self._verbose:
                print(f'Speed: {self.speed}')
        
        for callback in self.more:
            callback(key)

    def _on_release(self, key:keyboard.Key) -> None:
        pass

    def add_more(self, callback:callable) -> None:
        self.more.append(callback)
