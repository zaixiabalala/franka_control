import os
import shutil
from typing import Tuple, List, Union, Dict, Optional
import time
import gc
import numpy as np
import cv2
from threading import Lock, Thread
from multiprocessing import shared_memory, Manager
import pyrealsense2 as rs

from r3kit.devices.camera.base import CameraBase
from r3kit.devices.camera.utils import inpaint
from r3kit.devices.camera.realsense.config import *
from r3kit.utils.vis import draw_time, save_imgs, save_img


class L515(CameraBase):
    def __init__(self, id:Optional[str]=L515_ID, name:str='L515') -> None:
        super().__init__(name=name)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if id is not None:
            self.config.enable_device(id)
        else:
            pass
        for stream_item in L515_STREAMS:
            self.config.enable_stream(*stream_item)
        # NOTE: hard code config
        self.align = rs.align(rs.stream.color)
        # self.hole_filling = rs.hole_filling_filter()
        self.hole_filling = None
        self.inpaint = False
        
        self.pipeline_profile = self.pipeline.start(self.config)
        depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame().as_video_frame()
        depth_frame = frames.get_depth_frame().as_depth_frame()
        depth2color = depth_frame.get_profile().get_extrinsics_to(color_frame.get_profile())
        self.depth2color = np.eye(4)
        self.depth2color[:3, :3] = np.array(depth2color.rotation).reshape((3, 3))
        self.depth2color[:3, 3] = depth2color.translation
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = [color_intrinsics.ppx, color_intrinsics.ppy, color_intrinsics.fx, color_intrinsics.fy]
        color_frame = color_frame.as_video_frame()
        depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
        color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
        self.color_image_dtype = color_image.dtype
        self.color_image_shape = color_image.shape
        self.depth_image_dtype = depth_image.dtype
        self.depth_image_shape = depth_image.shape
        
        self.in_streaming = False

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.in_streaming:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame().as_video_frame()
            depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
            if self.hole_filling is not None:
                depth_frame = self.hole_filling.process(depth_frame)
            color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            if self.inpaint:
                depth_image = inpaint(depth_image, missing_value=0)
            return (color_image, depth_image)
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                color_image = self.streaming_data["color"][-1]
                depth_image = self.streaming_data["depth"][-1]
                self.streaming_mutex.release()
                return (color_image, depth_image)
            elif hasattr(self, "streaming_array"):
                with self.streaming_lock:
                    color_image = np.copy(self.streaming_array["color"])
                    depth_image = np.copy(self.streaming_array["depth"])
                return (color_image, depth_image)
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
                self.streaming_mutex = Lock()
                self.streaming_data = {
                    "depth": [], 
                    "color": [], 
                    "timestamp_ms": []
                }
                # TODO: ugly realtime write
                self._write_flag = True
                self._write_idx = 0
                if os.path.exists(f'./.temp/{self.name}'):
                    shutil.rmtree(f'./.temp/{self.name}')
                self._streaming_data_writer = Thread(target=self._write_streaming_data, args=(f'./.temp/{self.name}',))
                self._streaming_data_writer.start()
            else:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()
                depth_memory_size = self.depth_image_dtype.itemsize * np.prod(self.depth_image_shape).item()
                color_memory_size = self.color_image_dtype.itemsize * np.prod(self.color_image_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                streaming_memory_size = depth_memory_size + color_memory_size + timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
                self.streaming_array = {
                    "depth": np.ndarray(self.depth_image_shape, dtype=self.depth_image_dtype, buffer=self.streaming_memory.buf[:depth_memory_size]), 
                    "color": np.ndarray(self.color_image_shape, dtype=self.color_image_dtype, buffer=self.streaming_memory.buf[depth_memory_size:depth_memory_size+color_memory_size]), 
                    "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[depth_memory_size+color_memory_size:])
                }
                self.streaming_array_meta = {
                    "depth": (self.depth_image_shape, self.depth_image_dtype.name, (0, depth_memory_size)), 
                    "color": (self.color_image_shape, self.color_image_dtype.name, (depth_memory_size, depth_memory_size+color_memory_size)), 
                    "timestamp_ms": ((1,), np.float64.__name__, (depth_memory_size+color_memory_size, depth_memory_size+color_memory_size+timestamp_memory_size))
                }
            self.pipeline_profile = self.pipeline.start(self.config, self.callback)
        self.in_streaming = True

    def stop_streaming(self) -> Dict[str, List[Union[np.ndarray, float]]]:
        self.pipeline.stop()
        if hasattr(self, "streaming_data"):
            # TODO: ugly realtime write
            self._write_flag = False
            self._streaming_data_writer.join()

            streaming_data = self.streaming_data
            self.streaming_data = {
                "depth": [], 
                "color": [], 
                "timestamp_ms": []
            }
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "depth": [np.copy(self.streaming_array["depth"])], 
                "color": [np.copy(self.streaming_array["color"])], 
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
            self.streaming_memory.close()
            self.streaming_memory.unlink()
            del self.streaming_memory
            del self.streaming_array, self.streaming_array_meta
            del self.streaming_manager
            del self.streaming_lock
        else:
            raise AttributeError
        self.pipeline_profile = self.pipeline.start(self.config)
        self.in_streaming = False
        return streaming_data
    
    def save_streaming(self, save_path:str, streaming_data:dict) -> None:
        has_writer = False
        for root, dirs, files in os.walk(f'./.temp/{self.name}'):
            if len(files) > 0:
                has_writer = True
                break
        if not has_writer:
            assert len(streaming_data["depth"]) == len(streaming_data["color"]) == len(streaming_data["timestamp_ms"])
        else:
            assert len(streaming_data["depth"]) + self._write_idx == len(streaming_data["color"]) + self._write_idx == len(streaming_data["timestamp_ms"])
        np.savetxt(os.path.join(save_path, "intrinsics.txt"), self.color_intrinsics, fmt="%.16f")
        np.savetxt(os.path.join(save_path, "depth_scale.txt"), [self.depth_scale], fmt="%.16f")
        np.savetxt(os.path.join(save_path, "depth2color.txt"), self.depth2color, fmt="%.16f")
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        if len(streaming_data["timestamp_ms"]) > 1:
            freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
            draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
        else:
            freq = 0
        # os.makedirs(os.path.join(save_path, 'depth'), exist_ok=True)
        # os.makedirs(os.path.join(save_path, 'color'), exist_ok=True)
        idx_bias = 0
        if has_writer:
            # os.rename(os.path.join(f'./.temp/{self.name}', 'depth'), os.path.join(save_path, 'depth'))
            # os.rename(os.path.join(f'./.temp/{self.name}', 'color'), os.path.join(save_path, 'color'))
            shutil.move(os.path.join(f'./.temp/{self.name}', 'depth'), os.path.join(save_path, 'depth'))
            shutil.move(os.path.join(f'./.temp/{self.name}', 'color'), os.path.join(save_path, 'color'))
            idx_bias = self._write_idx
            shutil.rmtree(f'./.temp/{self.name}')
        else:
            os.makedirs(os.path.join(save_path, 'depth'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'color'), exist_ok=True)
        save_imgs(os.path.join(save_path, 'depth'), streaming_data["depth"], idx_bias=idx_bias)
        save_imgs(os.path.join(save_path, 'color'), streaming_data["color"], idx_bias=idx_bias)
    
    def _write_streaming_data(self, save_path:str) -> None:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'color'), exist_ok=True)
        while True:
            if not self._write_flag:
                break

            to_write = False
            self.streaming_mutex.acquire()
            if len(self.streaming_data["color"]) > 0:
                to_write = True
                depth_img = self.streaming_data["depth"].pop(0)
                color_img = self.streaming_data["color"].pop(0)
            self.streaming_mutex.release()
            if to_write:
                save_img(self._write_idx, os.path.join(save_path, 'depth'), depth_img)
                save_img(self._write_idx, os.path.join(save_path, 'color'), color_img)
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
        if hasattr(self, "streaming_data"):
            # TODO: ugly realtime write
            self._write_flag = False
            self._streaming_data_writer.join()

            streaming_data = self.streaming_data
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "depth": [np.copy(self.streaming_array["depth"])], 
                "color": [np.copy(self.streaming_array["color"])], 
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
        else:
            raise AttributeError
        return streaming_data
    
    def reset_streaming(self) -> None:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            # TODO: ugly realtime write
            self._write_flag = False
            self._streaming_data_writer.join()

            self.streaming_data['depth'].clear()
            self.streaming_data['color'].clear()
            self.streaming_data['timestamp_ms'].clear()
            del self.streaming_data
            del self.streaming_mutex
            gc.collect()
        elif hasattr(self, "streaming_array"):
            self.streaming_memory.close()
            self.streaming_memory.unlink()
            del self.streaming_memory
            del self.streaming_array, self.streaming_array_meta
            del self.streaming_manager
            del self.streaming_lock
        else:
            raise AttributeError
        
        if self._shm is None:
            self.streaming_mutex = Lock()
            self.streaming_data = {
                "depth": [], 
                "color": [], 
                "timestamp_ms": []
            }
            # TODO: ugly realtime write
            self._write_flag = True
            self._write_idx = 0
            if os.path.exists(f'./.temp/{self.name}'):
                shutil.rmtree(f'./.temp/{self.name}')
            self._streaming_data_writer = Thread(target=self._write_streaming_data, args=(f'./.temp/{self.name}',))
            self._streaming_data_writer.start()
        else:
            self.streaming_manager = Manager()
            self.streaming_lock = self.streaming_manager.Lock()
            depth_memory_size = self.depth_image_dtype.itemsize * np.prod(self.depth_image_shape).item()
            color_memory_size = self.color_image_dtype.itemsize * np.prod(self.color_image_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            streaming_memory_size = depth_memory_size + color_memory_size + timestamp_memory_size
            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
            self.streaming_array = {
                "depth": np.ndarray(self.depth_image_shape, dtype=self.depth_image_dtype, buffer=self.streaming_memory.buf[:depth_memory_size]), 
                "color": np.ndarray(self.color_image_shape, dtype=self.color_image_dtype, buffer=self.streaming_memory.buf[depth_memory_size:depth_memory_size+color_memory_size]), 
                "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[depth_memory_size+color_memory_size:])
            }
            self.streaming_array_meta = {
                "depth": (self.depth_image_shape, self.depth_image_dtype.name, (0, depth_memory_size)), 
                "color": (self.color_image_shape, self.color_image_dtype.name, (depth_memory_size, depth_memory_size+color_memory_size)), 
                "timestamp_ms": ((1,), np.float64.__name__, (depth_memory_size+color_memory_size, depth_memory_size+color_memory_size+timestamp_memory_size))
            }
    
    def callback(self, frame):
        ts = time.time() * 1000
        if not self._collect_streaming_data:
            return
        
        if frame.is_frameset():
            frameset = frame.as_frameset()
            frameset = self.align.process(frameset)
            color_frame = frameset.get_color_frame().as_video_frame()
            depth_frame = frameset.get_depth_frame().as_depth_frame()
            if self.hole_filling is not None:
                depth_frame = self.hole_filling.process(depth_frame)
            color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            if self.inpaint:
                depth_image = inpaint(depth_image, missing_value=0)
            # ts = frameset.get_timestamp()
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                if len(self.streaming_data["timestamp_ms"]) != 0 and ts == self.streaming_data["timestamp_ms"][-1]:
                    pass
                else:
                    self.streaming_data["depth"].append(depth_image.copy())
                    self.streaming_data["color"].append(color_image.copy())
                    self.streaming_data["timestamp_ms"].append(ts)
                self.streaming_mutex.release()
            elif hasattr(self, "streaming_array"):
                with self.streaming_lock:
                    self.streaming_array["depth"][:] = depth_image[:]
                    self.streaming_array["color"][:] = color_image[:]
                    self.streaming_array["timestamp_ms"][:] = ts
            else:
                raise AttributeError
    
    @staticmethod
    def img2pc(depth_img:np.ndarray, intrinsics:np.ndarray, color_img:Optional[np.ndarray]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # depth_img: already scaled by depth_scale
        # intrinsics: [ppx, ppy, fx, fy]
        # color_img: already converted to rgb and scaled to [0, 1]
        height, weight = depth_img.shape
        [pixX, pixY] = np.meshgrid(np.arange(weight), np.arange(height))
        x = (pixX - intrinsics[0]) * depth_img / intrinsics[2]
        y = (pixY - intrinsics[1]) * depth_img / intrinsics[3]
        z = depth_img
        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        if color_img is None:
            rgb = None
        else:
            rgb = color_img.reshape(-1, 3)
        return xyz, rgb

    def __del__(self) -> None:
        self.pipeline.stop()


if __name__ == "__main__":
    from r3kit.utils.vis import vis_pc

    camera = L515(id='f0172289', name='L515')
    streaming = False
    shm = False

    if not streaming:
        i = 0
        while True:
            print(f"{i}th")
            color, depth = camera.get()
            z = depth * camera.depth_scale
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) / 255.0

            xyz, rgb = L515.img2pc(z, camera.color_intrinsics, rgb)
            # valid_mask = xyz[:, 2] <= 1.5
            # xyz = xyz[valid_mask, :]
            # rgb = rgb[valid_mask, :]
            print(np.mean(xyz[:, 2]))
            vis_pc(xyz, rgb)

            cv2.imshow('color', color)
            while True:
                if cv2.getWindowProperty('color', cv2.WND_PROP_VISIBLE) <= 0:
                    break
                cv2.waitKey(1)
            cv2.destroyAllWindows()
            
            cmd = input("whether save? (y/n): ")
            if cmd == 'y':
                cv2.imwrite(f"rgb_{i}.png", color)
                np.savez(f"xyzrgb_{i}.npz", xyz=xyz, rgb=rgb)
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
        camera.shm_streaming(shm='L515' if shm else None)
        camera.start_streaming()

        cmd = input("quit? (enter): ")
        streaming_data = camera.stop_streaming()
        print(len(streaming_data["timestamp_ms"]))
        color, depth = streaming_data["color"][-1], streaming_data["depth"][-1]
        z = depth * camera.depth_scale
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) / 255.0

        xyz, rgb = L515.img2pc(z, camera.color_intrinsics, rgb)
        # valid_mask = xyz[:, 2] <= 1.5
        # xyz = xyz[valid_mask, :]
        # rgb = rgb[valid_mask, :]
        print(np.mean(xyz[:, 2]))
        vis_pc(xyz, rgb)

        cv2.imshow('color', color)
        while True:
            if cv2.getWindowProperty('color', cv2.WND_PROP_VISIBLE) <= 0:
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
