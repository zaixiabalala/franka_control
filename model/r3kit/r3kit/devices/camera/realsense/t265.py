import os
import shutil
from typing import Tuple, List, Union, Dict, Optional
import time
import gc
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot
from threading import Lock, Thread
from multiprocessing import shared_memory, Manager
import pyrealsense2 as rs

from r3kit.devices.camera.base import CameraBase
from r3kit.devices.camera.realsense.config import *
from r3kit.utils.vis import draw_time, save_imgs, save_img


class T265(CameraBase):
    def __init__(self, id:Optional[str]=T265_ID, image:bool=True, name:str='T265') -> None:
        super().__init__(name=name)
        self._image = image

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if id is not None:
            self.config.enable_device(id)
        else:
            pass
        for stream_item in T265_STREAMS:
            if not image and stream_item[0] == rs.stream.fisheye:
                continue
            self.config.enable_stream(*stream_item)
        
        # NOTE: hard code to balance pose accuracy and smoothness
        self.pipeline.start(self.config)
        pose_sensor = self.pipeline.get_active_profile().get_device().first_pose_sensor()
        self.pipeline.stop()
        # pose_sensor.set_option(rs.option.enable_mapping, 0)
        pose_sensor.set_option(rs.option.enable_pose_jumping, 0)
        # pose_sensor.set_option(rs.option.enable_relocalization, 0)
        self.pipeline.start(self.config)
        frames = self.pipeline.wait_for_frames()
        if self._image:
            f1 = frames.get_fisheye_frame(1).as_video_frame()
            f2 = frames.get_fisheye_frame(2).as_video_frame()
            left_image = np.asanyarray(f1.get_data(), dtype=np.uint8)
            right_image = np.asanyarray(f2.get_data(), dtype=np.uint8)
            self.left_image_dtype = left_image.dtype
            self.left_image_shape = left_image.shape
            self.right_image_dtype = right_image.dtype
            self.right_image_shape = right_image.shape
        pose_frame = frames.get_pose_frame()
        pose_data_ = pose_frame.get_pose_data()
        xyz = np.array([pose_data_.translation.x, pose_data_.translation.y, pose_data_.translation.z], dtype=np.float64)
        quat = np.array([pose_data_.rotation.x, pose_data_.rotation.y, pose_data_.rotation.z, pose_data_.rotation.w], dtype=np.float64)
        self.xyz_dtype = xyz.dtype
        self.xyz_shape = xyz.shape
        self.quat_dtype = quat.dtype
        self.quat_shape = quat.shape

        self.in_streaming = False

    def get(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
        if not self.in_streaming:
            frames = self.pipeline.wait_for_frames()
            if self._image:
                f1 = frames.get_fisheye_frame(1).as_video_frame()
                f2 = frames.get_fisheye_frame(2).as_video_frame()
                left_image = np.asanyarray(f1.get_data(), dtype=np.uint8)
                right_image = np.asanyarray(f2.get_data(), dtype=np.uint8)
            else:
                left_image = None
                right_image = None
            
            pose_frame = frames.get_pose_frame()
            pose_data_ = pose_frame.get_pose_data()
            xyz = np.array([pose_data_.translation.x, pose_data_.translation.y, pose_data_.translation.z], dtype=np.float64)
            quat = np.array([pose_data_.rotation.x, pose_data_.rotation.y, pose_data_.rotation.z, pose_data_.rotation.w], dtype=np.float64)
            return (left_image, right_image, xyz, quat)
        else:
            if hasattr(self, "pose_streaming_data"):
                if self._image:
                    self.image_streaming_mutex.acquire()
                    left_image = self.image_streaming_data["left"][-1]
                    right_image = self.image_streaming_data["right"][-1]
                    self.image_streaming_mutex.release()
                else:
                    left_image = None
                    right_image = None
                self.pose_streaming_mutex.acquire()
                xyz = self.pose_streaming_data["xyz"][-1]
                quat = self.pose_streaming_data["quat"][-1]
                self.pose_streaming_mutex.release()
                return (left_image, right_image, xyz, quat)
            elif hasattr(self, "pose_streaming_array"):
                if self._image:
                    with self.image_streaming_lock:
                        left_image = np.copy(self.image_streaming_array["left"])
                        right_image = np.copy(self.image_streaming_array["right"])
                else:
                    left_image = None
                    right_image = None
                with self.pose_streaming_lock:
                    xyz = np.copy(self.pose_streaming_array["xyz"])
                    quat = np.copy(self.pose_streaming_array["quat"])
                return (left_image, right_image, xyz, quat)
            else:
                raise AttributeError
    
    def start_streaming(self, callback:Optional[callable]=None) -> None:
        self.pipeline.stop()
        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
        if not hasattr(self, "_shm"):
            self._shm = None
        
        if callback is not None:
            self.pipeline_profile = self.pipeline.start(self.config, callback)
        else:
            if self._shm is None:
                if self._image:
                    self.image_streaming_mutex = Lock()
                    self.image_streaming_data = {
                        "left": [], 
                        "right": [], 
                        "timestamp_ms": [], 
                    }
                    # TODO: ugly realtime write
                    self._write_flag = True
                    self._write_idx = 0
                    if os.path.exists(f'./.temp/{self.name}'):
                        shutil.rmtree(f'./.temp/{self.name}')
                    self._image_streaming_data_writer = Thread(target=self._write_image_streaming_data, args=(f'./.temp/{self.name}',))
                    self._image_streaming_data_writer.start()
                self.pose_streaming_mutex = Lock()
                self.pose_streaming_data = {
                    "xyz": [], 
                    "quat": [], 
                    "timestamp_ms": [], 
                }
            else:
                self.streaming_manager = Manager()
                self.pose_streaming_lock = self.streaming_manager.Lock()
                xyz_memory_size = self.xyz_dtype.itemsize * np.prod(self.xyz_shape).item()
                quat_memory_size = self.quat_dtype.itemsize * np.prod(self.quat_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                pose_streaming_memory_size = xyz_memory_size + quat_memory_size + timestamp_memory_size
                self.pose_streaming_memory = shared_memory.SharedMemory(name=self._shm+'_pose', create=True, size=pose_streaming_memory_size)
                self.pose_streaming_array = {
                    "xyz": np.ndarray(self.xyz_shape, dtype=self.xyz_dtype, buffer=self.pose_streaming_memory.buf[:xyz_memory_size]), 
                    "quat": np.ndarray(self.quat_shape, dtype=self.quat_dtype, buffer=self.pose_streaming_memory.buf[xyz_memory_size:xyz_memory_size+quat_memory_size]), 
                    "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.pose_streaming_memory.buf[xyz_memory_size+quat_memory_size:])
                }
                self.pose_streaming_array_meta = {
                    "xyz": (self.xyz_shape, self.xyz_dtype.name, (0, xyz_memory_size)), 
                    "quat": (self.quat_shape, self.quat_dtype.name, (xyz_memory_size, xyz_memory_size+quat_memory_size)), 
                    "timestamp_ms": ((1,), np.float64.__name__, (xyz_memory_size+quat_memory_size, xyz_memory_size+quat_memory_size+timestamp_memory_size))
                }
                if self._image:
                    self.image_streaming_lock = self.streaming_manager.Lock()
                    left_memory_size = self.left_image_dtype.itemsize * np.prod(self.left_image_shape).item()
                    right_memory_size = self.right_image_dtype.itemsize * np.prod(self.right_image_shape).item()
                    image_streaming_memory_size = left_memory_size + right_memory_size + timestamp_memory_size
                    self.image_streaming_memory = shared_memory.SharedMemory(name=self._shm+'_image', create=True, size=image_streaming_memory_size)
                    self.image_streaming_array = {
                        "left": np.ndarray(self.left_image_shape, dtype=self.left_image_dtype, buffer=self.image_streaming_memory.buf[:left_memory_size]), 
                        "right": np.ndarray(self.right_image_shape, dtype=self.right_image_dtype, buffer=self.image_streaming_memory.buf[left_memory_size:left_memory_size+right_memory_size]), 
                        "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.image_streaming_memory.buf[left_memory_size+right_memory_size:])
                    }
                    self.image_streaming_array_meta = {
                        "left": (self.left_image_shape, self.left_image_dtype.name, (0, left_memory_size)), 
                        "right": (self.right_image_shape, self.right_image_dtype.name, (left_memory_size, left_memory_size+right_memory_size)), 
                        "timestamp_ms": ((1,), np.float64.__name__, (left_memory_size+right_memory_size, left_memory_size+right_memory_size+timestamp_memory_size))
                    }
            self.pipeline_profile = self.pipeline.start(self.config, self.callback)
        self.in_streaming = True

    def stop_streaming(self) -> Dict[str, List[Union[np.ndarray, float]]]:
        self.pipeline.stop()
        if hasattr(self, "pose_streaming_data"):
            streaming_data = {'pose': self.pose_streaming_data}
            self.pose_streaming_data = {
                "xyz": [], 
                "quat": [], 
                "timestamp_ms": [], 
            }
            del self.pose_streaming_data
            del self.pose_streaming_mutex
            if self._image:
                # TODO: ugly realtime write
                self._write_flag = False
                self._image_streaming_data_writer.join()

                streaming_data['image'] = self.image_streaming_data
                self.image_streaming_data = {
                    "left": [], 
                    "right": [], 
                    "timestamp_ms": [], 
                }
                del self.image_streaming_data
                del self.image_streaming_mutex
        elif hasattr(self, "pose_streaming_array"):
            streaming_data = {'pose': {
                "xyz": [np.copy(self.pose_streaming_array["xyz"])], 
                "quat": [np.copy(self.pose_streaming_array["quat"])], 
                "timestamp_ms": [self.pose_streaming_array["timestamp_ms"].item()]
            }}
            self.pose_streaming_memory.close()
            self.pose_streaming_memory.unlink()
            del self.pose_streaming_memory
            del self.pose_streaming_array, self.pose_streaming_array_meta
            del self.pose_streaming_lock
            if self._image:
                streaming_data['image'] = {
                    "left": [np.copy(self.image_streaming_array["left"])], 
                    "right": [np.copy(self.image_streaming_array["right"])], 
                    "timestamp_ms": [self.image_streaming_array["timestamp_ms"].item()]
                }
                self.image_streaming_memory.close()
                self.image_streaming_memory.unlink()
                del self.image_streaming_memory
                del self.image_streaming_array, self.image_streaming_array_meta
                del self.image_streaming_lock
            del self.streaming_manager
        else:
            raise AttributeError
        self.pipeline_profile = self.pipeline.start(self.config)
        self.in_streaming = False
        return streaming_data
    
    def save_streaming(self, save_path:str, streaming_data:dict) -> None:
        if self._image:
            has_writer = False
            for root, dirs, files in os.walk(f'./.temp/{self.name}'):
                if len(files) > 0:
                    has_writer = True
                    break
            if not has_writer:
                assert len(streaming_data["image"]["left"]) == len(streaming_data["image"]["right"]) == len(streaming_data["image"]["timestamp_ms"])
            else:
                assert len(streaming_data["image"]["left"]) + self._write_idx == len(streaming_data["image"]["right"]) + self._write_idx == len(streaming_data["image"]["timestamp_ms"])
            os.makedirs(os.path.join(save_path, 'image'), exist_ok=True)
            np.save(os.path.join(save_path, 'image', "timestamps.npy"), np.array(streaming_data["image"]["timestamp_ms"], dtype=float))
            if len(streaming_data["image"]["timestamp_ms"]) > 1:
                freq = len(streaming_data["image"]["timestamp_ms"]) / (streaming_data["image"]["timestamp_ms"][-1] - streaming_data["image"]["timestamp_ms"][0])
                draw_time(streaming_data["image"]["timestamp_ms"], os.path.join(save_path, 'image', f"freq_{freq}.png"))
            else:
                freq = 0
            # os.makedirs(os.path.join(save_path, 'image', 'left'), exist_ok=True)
            # os.makedirs(os.path.join(save_path, 'image', 'right'), exist_ok=True)
            idx_bias = 0
            if has_writer:
                # os.rename(os.path.join(f'./.temp/{self.name}', 'left'), os.path.join(save_path, 'image', 'left'))
                # os.rename(os.path.join(f'./.temp/{self.name}', 'right'), os.path.join(save_path, 'image', 'right'))
                shutil.move(os.path.join(f'./.temp/{self.name}', 'left'), os.path.join(save_path, 'image', 'left'))
                shutil.move(os.path.join(f'./.temp/{self.name}', 'right'), os.path.join(save_path, 'image', 'right'))
                idx_bias = self._write_idx
                shutil.rmtree(f'./.temp/{self.name}')
            else:
                os.makedirs(os.path.join(save_path, 'image', 'left'), exist_ok=True)
                os.makedirs(os.path.join(save_path, 'image', 'right'), exist_ok=True)
            save_imgs(os.path.join(save_path, 'image', 'left'), streaming_data["image"]["left"], idx_bias=idx_bias)
            save_imgs(os.path.join(save_path, 'image', 'right'), streaming_data["image"]["right"], idx_bias=idx_bias)
        assert len(streaming_data["pose"]["xyz"]) == len(streaming_data["pose"]["quat"]) == len(streaming_data["pose"]["timestamp_ms"])
        os.makedirs(os.path.join(save_path, 'pose'), exist_ok=True)
        np.save(os.path.join(save_path, 'pose', "timestamps.npy"), np.array(streaming_data["pose"]["timestamp_ms"], dtype=float))
        if len(streaming_data["pose"]["timestamp_ms"]) > 1:
            freq = len(streaming_data["pose"]["timestamp_ms"]) / (streaming_data["pose"]["timestamp_ms"][-1] - streaming_data["pose"]["timestamp_ms"][0])
            draw_time(streaming_data["pose"]["timestamp_ms"], os.path.join(save_path, 'pose', f"freq_{freq}.png"))
        else:
            freq = 0
        np.save(os.path.join(save_path, 'pose', "xyz.npy"), np.array(streaming_data["pose"]["xyz"], dtype=float))
        np.save(os.path.join(save_path, 'pose', "quat.npy"), np.array(streaming_data["pose"]["quat"], dtype=float))
    
    def _write_image_streaming_data(self, save_path:str) -> None:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'left'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'right'), exist_ok=True)
        while True:
            if not self._write_flag:
                break

            to_write = False
            self.image_streaming_mutex.acquire()
            if len(self.image_streaming_data["left"]) > 0:
                to_write = True
                left_img = self.image_streaming_data["left"].pop(0)
                right_img = self.image_streaming_data["right"].pop(0)
            self.image_streaming_mutex.release()
            if to_write:
                save_img(self._write_idx, os.path.join(save_path, 'left'), left_img)
                save_img(self._write_idx, os.path.join(save_path, 'right'), right_img)
                self._write_idx += 1
    
    def collect_streaming(self, collect:bool=True) -> None:
        # NOTE: only valid for no-custom-callback
        self._collect_streaming_data = collect
    
    def shm_streaming(self, shm:Optional[str]=None) -> None:
        # NOTE: only valid for non-custom-callback
        assert (not self.in_streaming) or (not self._collect_streaming_data)
        self._shm = shm
    
    def get_streaming(self) -> Dict[str, List[Union[np.ndarray, float]]]:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "pose_streaming_data"):
            streaming_data = {'pose': self.pose_streaming_data}
            if self._image:
                # TODO: ugly realtime write
                self._write_flag = False
                self._image_streaming_data_writer.join()

                streaming_data['image'] = self.image_streaming_data
        elif hasattr(self, "pose_streaming_array"):
            streaming_data = {'pose': {
                "xyz": [np.copy(self.pose_streaming_array["xyz"])], 
                "quat": [np.copy(self.pose_streaming_array["quat"])], 
                "timestamp_ms": [self.pose_streaming_array["timestamp_ms"].item()]
            }}
            if self._image:
                streaming_data['image'] = {
                    "left": [np.copy(self.image_streaming_array["left"])], 
                    "right": [np.copy(self.image_streaming_array["right"])], 
                    "timestamp_ms": [self.image_streaming_array["timestamp_ms"].item()]
                }
        else:
            raise AttributeError
        return streaming_data
    
    def reset_streaming(self) -> None:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "pose_streaming_data"):
            if self._image:
                # TODO: ugly realtime write
                self._write_flag = False
                self._image_streaming_data_writer.join()

                self.image_streaming_data['left'].clear()
                self.image_streaming_data['right'].clear()
                self.image_streaming_data['timestamp_ms'].clear()
                del self.image_streaming_data
                del self.image_streaming_mutex
            self.pose_streaming_data['xyz'].clear()
            self.pose_streaming_data['quat'].clear()
            self.pose_streaming_data['timestamp_ms'].clear()
            del self.pose_streaming_data
            del self.pose_streaming_mutex
            gc.collect()
        elif hasattr(self, "pose_streaming_array"):
            if self._image:
                self.image_streaming_memory.close()
                self.image_streaming_memory.unlink()
                del self.image_streaming_memory
                del self.image_streaming_array, self.image_streaming_array_meta
                del self.image_streaming_lock
            self.pose_streaming_memory.close()
            self.pose_streaming_memory.unlink()
            del self.pose_streaming_memory
            del self.pose_streaming_array, self.pose_streaming_array_meta
            del self.pose_streaming_lock
            del self.streaming_manager
        else:
            raise AttributeError
        
        if self._shm is None:
            if self._image:
                self.image_streaming_mutex = Lock()
                self.image_streaming_data = {
                    "left": [], 
                    "right": [], 
                    "timestamp_ms": [], 
                }
                # TODO: ugly realtime write
                self._write_flag = True
                self._write_idx = 0
                if os.path.exists(f'./.temp/{self.name}'):
                    shutil.rmtree(f'./.temp/{self.name}')
                self._image_streaming_data_writer = Thread(target=self._write_image_streaming_data, args=(f'./.temp/{self.name}',))
                self._image_streaming_data_writer.start()
            self.pose_streaming_mutex = Lock()
            self.pose_streaming_data = {
                "xyz": [], 
                "quat": [], 
                "timestamp_ms": [], 
            }
        else:
            self.streaming_manager = Manager()
            self.pose_streaming_lock = self.streaming_manager.Lock()
            xyz_memory_size = self.xyz_dtype.itemsize * np.prod(self.xyz_shape).item()
            quat_memory_size = self.quat_dtype.itemsize * np.prod(self.quat_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            pose_streaming_memory_size = xyz_memory_size + quat_memory_size + timestamp_memory_size
            self.pose_streaming_memory = shared_memory.SharedMemory(name=self._shm+'_pose', create=True, size=pose_streaming_memory_size)
            self.pose_streaming_array = {
                "xyz": np.ndarray(self.xyz_shape, dtype=self.xyz_dtype, buffer=self.pose_streaming_memory.buf[:xyz_memory_size]), 
                "quat": np.ndarray(self.quat_shape, dtype=self.quat_dtype, buffer=self.pose_streaming_memory.buf[xyz_memory_size:xyz_memory_size+quat_memory_size]), 
                "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.pose_streaming_memory.buf[xyz_memory_size+quat_memory_size:])
            }
            self.pose_streaming_array_meta = {
                "xyz": (self.xyz_shape, self.xyz_dtype.name, (0, xyz_memory_size)), 
                "quat": (self.quat_shape, self.quat_dtype.name, (xyz_memory_size, xyz_memory_size+quat_memory_size)), 
                "timestamp_ms": ((1,), np.float64.__name__, (xyz_memory_size+quat_memory_size, xyz_memory_size+quat_memory_size+timestamp_memory_size))
            }
            if self._image:
                self.image_streaming_lock = self.streaming_manager.Lock()
                left_memory_size = self.left_image_dtype.itemsize * np.prod(self.left_image_shape).item()
                right_memory_size = self.right_image_dtype.itemsize * np.prod(self.right_image_shape).item()
                image_streaming_memory_size = left_memory_size + right_memory_size + timestamp_memory_size
                self.image_streaming_memory = shared_memory.SharedMemory(name=self._shm+'_image', create=True, size=image_streaming_memory_size)
                self.image_streaming_array = {
                    "left": np.ndarray(self.left_image_shape, dtype=self.left_image_dtype, buffer=self.image_streaming_memory.buf[:left_memory_size]), 
                    "right": np.ndarray(self.right_image_shape, dtype=self.right_image_dtype, buffer=self.image_streaming_memory.buf[left_memory_size:left_memory_size+right_memory_size]), 
                    "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.image_streaming_memory.buf[left_memory_size+right_memory_size:])
                }
                self.image_streaming_array_meta = {
                    "left": (self.left_image_shape, self.left_image_dtype.name, (0, left_memory_size)), 
                    "right": (self.right_image_shape, self.right_image_dtype.name, (left_memory_size, left_memory_size+right_memory_size)), 
                    "timestamp_ms": ((1,), np.float64.__name__, (left_memory_size+right_memory_size, left_memory_size+right_memory_size+timestamp_memory_size))
                }
    
    def callback(self, frame):
        ts = time.time() * 1000
        if not self._collect_streaming_data:
            return
        
        if frame.is_frameset() and self._image:
            frameset = frame.as_frameset()
            f1 = frameset.get_fisheye_frame(1).as_video_frame()
            f2 = frameset.get_fisheye_frame(2).as_video_frame()
            left_data = np.asanyarray(f1.get_data(), dtype=np.uint8)
            right_data = np.asanyarray(f2.get_data(), dtype=np.uint8)
            # ts = frameset.get_timestamp()
            if hasattr(self, "pose_streaming_data"):
                self.image_streaming_mutex.acquire()
                if len(self.image_streaming_data["timestamp_ms"]) != 0 and ts == self.image_streaming_data["timestamp_ms"][-1]:
                    pass
                else:
                    self.image_streaming_data["left"].append(left_data.copy())
                    self.image_streaming_data["right"].append(right_data.copy())
                    self.image_streaming_data["timestamp_ms"].append(ts)
                self.image_streaming_mutex.release()
            elif hasattr(self, "pose_streaming_array"):
                with self.image_streaming_lock:
                    self.image_streaming_array["left"][:] = left_data[:]
                    self.image_streaming_array["right"][:] = right_data[:]
                    self.image_streaming_array["timestamp_ms"][:] = ts
            else:
                raise AttributeError
        
        if frame.is_pose_frame():
            pose_frame = frame.as_pose_frame()
            pose_data_ = pose_frame.get_pose_data()
            xyz = np.array([pose_data_.translation.x, pose_data_.translation.y, pose_data_.translation.z], dtype=np.float64)
            quat = np.array([pose_data_.rotation.x, pose_data_.rotation.y, pose_data_.rotation.z, pose_data_.rotation.w], dtype=np.float64)
            # ts = pose_frame.timestamp
            if hasattr(self, "pose_streaming_data"):
                self.pose_streaming_mutex.acquire()
                if len(self.pose_streaming_data["timestamp_ms"]) != 0 and ts == self.pose_streaming_data["timestamp_ms"][-1]:
                    pass
                else:
                    self.pose_streaming_data["xyz"].append(xyz)
                    self.pose_streaming_data["quat"].append(quat)
                    self.pose_streaming_data["timestamp_ms"].append(ts)
                self.pose_streaming_mutex.release()
            elif hasattr(self, "pose_streaming_array"):
                with self.pose_streaming_lock:
                    self.pose_streaming_array["xyz"][:] = xyz[:]
                    self.pose_streaming_array["quat"][:] = quat[:]
                    self.pose_streaming_array["timestamp_ms"][:] = ts
            else:
                raise AttributeError
    
    @staticmethod
    def raw2pose(xyz:np.ndarray, quat:np.ndarray) -> np.ndarray:
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :3] = Rot.from_quat(quat).as_matrix()
        pose_4x4[:3, 3] = xyz
        return pose_4x4

    def __del__(self) -> None:
        self.pipeline.stop()


if __name__ == "__main__":
    camera = T265(id='230222110234', image=True, name='T265')
    streaming = True
    shm = True
    
    if not streaming:
        i = 0
        while True:
            print(f"{i}th")
            left, right, xyz, quat = camera.get()

            print(f"xyz: {xyz}")
            print(f"quat: {quat}")
            cv2.imshow('left', left)
            cv2.imshow('right', right)
            while True:
                if cv2.getWindowProperty('left', cv2.WND_PROP_VISIBLE) <= 0:
                    break
                cv2.waitKey(1)
            while True:
                if cv2.getWindowProperty('right', cv2.WND_PROP_VISIBLE) <= 0:
                    break
                cv2.waitKey(1)
            cv2.destroyAllWindows()
            
            cmd = input("whether save? (y/n): ")
            if cmd == 'y':
                cv2.imwrite(f"left_{i}.png", left)
                cv2.imwrite(f"right_{i}.png", right)
                np.savetxt(f"xyz_{i}.txt", xyz)
                np.savetxt(f"quat_{i}.txt", quat)
                i += 1
            elif cmd == 'n':
                cmd = input("whether quit? (y/n): ")
                if cmd == 'y':
                    break
                elif cmd == 'n':
                    pass
                else:
                    raise ValueError
            else:
                raise ValueError
    else:
        camera.collect_streaming(collect=True)
        camera.shm_streaming(shm='T265' if shm else None)
        camera.start_streaming()

        cmd = input("quit? (enter): ")
        streaming_data = camera.stop_streaming()
        print(len(streaming_data["image"]["timestamp_ms"]), len(streaming_data["pose"]["timestamp_ms"]))
        left, right = streaming_data["image"]["left"][-1], streaming_data["image"]["right"][-1]
        xyz, quat = streaming_data["pose"]["xyz"][-1], streaming_data["pose"]["quat"][-1]

        print(f"xyz: {xyz}")
        print(f"quat: {quat}")
        cv2.imshow('left', left)
        cv2.imshow('right', right)
        while True:
            if cv2.getWindowProperty('left', cv2.WND_PROP_VISIBLE) <= 0:
                break
            cv2.waitKey(1)
        while True:
            if cv2.getWindowProperty('right', cv2.WND_PROP_VISIBLE) <= 0:
                break
            cv2.waitKey(1)
        cv2.destroyAllWindows()

        cmd = input("whether save? (y/n): ")
        if cmd == 'y':
            camera.save_streaming('.', streaming_data)
        elif cmd == 'n':
            pass
        else:
            raise ValueError
