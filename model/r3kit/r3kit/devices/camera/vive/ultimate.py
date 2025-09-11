"""
Modified from https://github.com/cmbruns/pyopenxr_examples/blob/main/xr_examples/vive_tracker.py
"""

import os
from typing import List, Dict, Union, Optional
import ctypes
from ctypes import cast, byref, POINTER
import time
import gc
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from threading import Thread, Lock, Event
from multiprocessing import shared_memory, Manager
from copy import deepcopy
from functools import partial
import xr

from r3kit.devices.camera.base import CameraBase
from r3kit.devices.camera.vive.context_object import ContextObject
from r3kit.devices.camera.vive.config import *
from r3kit.utils.vis import draw_time


class Ultimate(CameraBase):
    def __init__(self, id:List[str]=ULTIMATE_ROLE, fps:int=ULTIMATE_FPS, name:str='Ultimate') -> None:
        super().__init__(name=name)

        self._num = len(id)
        self._fps = fps

        # create shared context object
        self.context = ContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                enabled_extension_names=ULTIMATE_EXTENSION_NAMES,
            ),
        )
        self.roles = id
        self.instance = self.context.instance
        self.session = self.context.session
        # Save the function pointer
        self.enumerateViveTrackerPathsHTCX = cast(
            xr.get_instance_proc_addr(
                self.context.instance,
                "xrEnumerateViveTrackerPathsHTCX",
            ),
            xr.PFN_xrEnumerateViveTrackerPathsHTCX
        )
        role_path_strings = [f"/user/vive_tracker_htcx/role/{r}" for r in self.roles]
        role_paths = (xr.Path * len(role_path_strings))(
            *[xr.string_to_path(self.instance, role_path_string) for role_path_string in role_path_strings],
        )
        pose_action = xr.create_action(
            action_set=self.context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="tracker_pose",
                localized_action_name="Tracker Pose",
                count_subaction_paths=len(role_paths),
                subaction_paths=role_paths,
            ),
        )
        # Describe a suggested binding for that action and subaction path
        suggested_binding_paths = (xr.ActionSuggestedBinding * len(role_path_strings))(
            *[xr.ActionSuggestedBinding(
                pose_action,
                xr.string_to_path(self.instance, f"{role_path_string}/input/grip/pose"))
            for role_path_string in role_path_strings],
        )
        xr.suggest_interaction_profile_bindings(
            instance=self.instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(self.instance, "/interaction_profiles/htc/vive_tracker_htcx"),
                count_suggested_bindings=1,
                suggested_bindings=suggested_binding_paths,
            )
        )
        # Create action spaces for locating trackers in each role
        self.tracker_action_spaces = (xr.Space * len(role_paths))(
            *[xr.create_action_space(
                session=self.session,
                create_info=xr.ActionSpaceCreateInfo(
                    action=pose_action,
                    subaction_path=role_path,
                )
            ) for role_path in role_paths],
        )
        # Warm up
        n_paths = ctypes.c_uint32(0)
        result = self.enumerateViveTrackerPathsHTCX(self.instance, 0, byref(n_paths), None)
        if xr.check_result(result).is_exception():
            raise result
        vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
        # print(xr.Result(result), n_paths.value)
        result = self.enumerateViveTrackerPathsHTCX(self.instance, n_paths, byref(n_paths), vive_tracker_paths)
        if xr.check_result(result).is_exception():
            raise result
        # print(xr.Result(result), n_paths.value)
        # print(*vive_tracker_paths)
        self.context.attach_session_action_sets()

        # config
        initial_start_time = time.time()
        data = self._read()
        while data is None:
            data = self._read()
            if time.time() - initial_start_time > 3:
                raise RuntimeError("Ultimate cannot read data")
        self.xyz_dtype = data['xyz'].dtype
        self.xyz_shape = data['xyz'].shape
        self.quat_dtype = data['quat'].dtype
        self.quat_shape = data['quat'].shape

        self.in_streaming = Event()
    
    def _read(self) -> Optional[Dict[str, Union[np.ndarray, float]]]:
        session_was_focused = False  # Check for a common problem
        self.context.exit_render_loop = False
        self.context.poll_xr_events()
        if self.context.exit_render_loop:
            return None
        if self.context.session_is_running:
            if self.context.session_state in (
                xr.SessionState.READY,
                xr.SessionState.SYNCHRONIZED,
                xr.SessionState.VISIBLE,
                xr.SessionState.FOCUSED,
            ):
                frame_state = xr.wait_frame(self.context.session)
                receive_time = time.time() * 1000
                xr.begin_frame(self.context.session)
                self.context.render_layers = []

                if self.context.session_state == xr.SessionState.FOCUSED:
                    session_was_focused = True
                    active_action_set = xr.ActiveActionSet(
                        action_set=self.context.default_action_set,
                        subaction_path=xr.NULL_PATH,
                    )
                    xr.sync_actions(
                        session=self.session,
                        sync_info=xr.ActionsSyncInfo(
                            count_active_action_sets=1,
                            active_action_sets=ctypes.pointer(active_action_set),
                        ),
                    )

                    n_paths = ctypes.c_uint32(0)
                    result = self.enumerateViveTrackerPathsHTCX(self.instance, 0, byref(n_paths), None)
                    if xr.check_result(result).is_exception():
                        raise result
                    vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
                    # print(xr.Result(result), n_paths.value)
                    result = self.enumerateViveTrackerPathsHTCX(self.instance, n_paths, byref(n_paths), vive_tracker_paths)
                    if xr.check_result(result).is_exception():
                        raise result
                    # print(xr.Result(result), n_paths.value)
                    # print(*vive_tracker_paths)

                    xyzs, quats = [], []
                    for index, space in enumerate(self.tracker_action_spaces):
                        space_location = xr.locate_space(
                            space=space,
                            base_space=self.context.space,
                            time=frame_state.predicted_display_time,
                        )
                        if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                            xyzs.append(np.array([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z]))
                            quats.append(np.array([space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z, space_location.pose.orientation.w]))
                    if len(xyzs) != self._num:
                        raise RuntimeError(f"Expected {self._num} trackers, but found {len(xyzs)}")
                
                xr.end_frame(
                    self.context.session,
                    frame_end_info=xr.FrameEndInfo(
                        display_time=frame_state.predicted_display_time,
                        environment_blend_mode=self.context.environment_blend_mode,
                        layers=self.context.render_layers,
                    )
                )
                if not session_was_focused:
                    raise RuntimeError("This OpenXR session never entered the FOCUSED state. Did you modify the headless configuration?")
                return {'xyz': np.array(xyzs), 'quat': np.array(quats), 'timestamp_ms': receive_time}
        if not session_was_focused:
            raise RuntimeError("This OpenXR session never entered the FOCUSED state. Did you modify the headless configuration?")
        return None

    def get(self) -> Optional[Dict[str, Union[np.ndarray, float]]]:
        if not self.in_streaming.is_set():
            data = self._read()
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                data = {}
                data['xyz'] = self.streaming_data["xyz"][-1]
                data['quat'] = self.streaming_data["quat"][-1]
                data['timestamp_ms'] = self.streaming_data["timestamp_ms"][-1]
                self.streaming_mutex.release()
            elif hasattr(self, "streaming_array"):
                data = {}
                with self.streaming_lock:
                    data['xyz'] = np.copy(self.streaming_array["xyz"])
                    data['quat'] = np.copy(self.streaming_array["quat"])
                    data['timestamp_ms'] = self.streaming_array["timestamp_ms"].item()
            else:
                raise AttributeError
        return data
    
    def start_streaming(self, callback:Optional[callable]=None) -> None:
        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
        if not hasattr(self, "_shm"):
            self._shm = None
        
        self.in_streaming.set()
        if self._shm is None:
            if callback is None:
                self.streaming_mutex = Lock()
                self.streaming_data = {
                    "xyz": [], 
                    "quat": [], 
                    "timestamp_ms": []
                }
            else:
                pass
        else:
            if callback is None:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()
                xyz_memory_size = self.xyz_dtype.itemsize * np.prod(self.xyz_shape).item()
                quat_memory_size = self.quat_dtype.itemsize * np.prod(self.quat_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                streaming_memory_size = xyz_memory_size + quat_memory_size + timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
                self.streaming_array = {
                    "xyz": np.ndarray(self.xyz_shape, dtype=self.xyz_dtype, buffer=self.streaming_memory.buf[:xyz_memory_size]), 
                    "quat": np.ndarray(self.quat_shape, dtype=self.quat_dtype, buffer=self.streaming_memory.buf[xyz_memory_size:xyz_memory_size+quat_memory_size]), 
                    "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[xyz_memory_size+quat_memory_size:])
                }
                self.streaming_array_meta = {
                    "xyz": (self.xyz_shape, self.xyz_dtype.name, (0, xyz_memory_size)), 
                    "quat": (self.quat_shape, self.quat_dtype.name, (xyz_memory_size, xyz_memory_size+quat_memory_size)), 
                    "timestamp_ms": ((1,), np.float64.__name__, (xyz_memory_size+quat_memory_size, xyz_memory_size+quat_memory_size+timestamp_memory_size))
                }
            else:
                pass
        self.thread = Thread(target=partial(self._streaming_data, callback=callback), daemon=True)
        self.thread.start()

    def stop_streaming(self) -> Dict[str, Union[List[np.ndarray], List[float]]]:
        self.in_streaming.clear()
        self.thread.join()
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            self.streaming_data = {
                "xyz": [], 
                "quat": [], 
                "timestamp_ms": []
            }
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "xyz": [np.copy(self.streaming_array["xyz"])], 
                "quat": [np.copy(self.streaming_array["quat"])], 
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
        return streaming_data
    
    def save_streaming(self, save_path:str, streaming_data:dict) -> None:
        assert len(streaming_data["xyz"]) == len(streaming_data["quat"]) == len(streaming_data["timestamp_ms"])
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        if len(streaming_data["timestamp_ms"]) > 1:
            freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
            draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
        else:
            freq = 0
        np.save(os.path.join(save_path, "xyz.npy"), np.array(streaming_data["xyz"], dtype=float))
        np.save(os.path.join(save_path, "quat.npy"), np.array(streaming_data["quat"], dtype=float))
    
    def collect_streaming(self, collect:bool=True) -> None:
        self._collect_streaming_data = collect
    
    def shm_streaming(self, shm:Optional[str]=None) -> None:
        # NOTE: only valid for non-custom-callback
        assert (not self.in_streaming.is_set()) or (not self._collect_streaming_data)
        self._shm = shm
    
    def get_streaming(self) -> Dict[str, Union[List[np.ndarray], List[float]]]:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "xyz": [np.copy(self.streaming_array["xyz"])], 
                "quat": [np.copy(self.streaming_array["quat"])], 
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
        else:
            raise AttributeError
        return streaming_data
    
    def reset_streaming(self) -> None:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            self.streaming_data['xyz'].clear()
            self.streaming_data['quat'].clear()
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
                "xyz": [], 
                "quat": [], 
                "timestamp_ms": []
            }
        else:
            self.streaming_manager = Manager()
            self.streaming_lock = self.streaming_manager.Lock()
            xyz_memory_size = self.xyz_dtype.itemsize * np.prod(self.xyz_shape).item()
            quat_memory_size = self.quat_dtype.itemsize * np.prod(self.quat_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            streaming_memory_size = xyz_memory_size + quat_memory_size + timestamp_memory_size
            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
            self.streaming_array = {
                "xyz": np.ndarray(self.xyz_shape, dtype=self.xyz_dtype, buffer=self.streaming_memory.buf[:xyz_memory_size]), 
                "quat": np.ndarray(self.quat_shape, dtype=self.quat_dtype, buffer=self.streaming_memory.buf[xyz_memory_size:xyz_memory_size+quat_memory_size]), 
                "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[xyz_memory_size+quat_memory_size:])
            }
            self.streaming_array_meta = {
                "xyz": (self.xyz_shape, self.xyz_dtype.name, (0, xyz_memory_size)), 
                "quat": (self.quat_shape, self.quat_dtype.name, (xyz_memory_size, xyz_memory_size+quat_memory_size)), 
                "timestamp_ms": ((1,), np.float64.__name__, (xyz_memory_size+quat_memory_size, xyz_memory_size+quat_memory_size+timestamp_memory_size))
            }
    
    def _streaming_data(self, callback:Optional[callable]=None):
        while self.in_streaming.is_set():
            # fps
            time.sleep(1/self._fps)

            # get data
            if not self._collect_streaming_data:
                continue
            data = self._read()
            if data is not None:
                if callback is None:
                    if hasattr(self, "streaming_data"):
                        self.streaming_mutex.acquire()
                        self.streaming_data['xyz'].append(data['xyz'])
                        self.streaming_data['quat'].append(data['quat'])
                        self.streaming_data['timestamp_ms'].append(data['timestamp_ms'])
                        self.streaming_mutex.release()
                    elif hasattr(self, "streaming_array"):
                        with self.streaming_lock:
                            self.streaming_array["xyz"][:] = data['xyz'][:]
                            self.streaming_array["quat"][:] = data['quat'][:]
                            self.streaming_array["timestamp_ms"][:] = data['timestamp_ms']
                    else:
                        raise AttributeError
                else:
                    callback(deepcopy(data))
    
    @staticmethod
    def raw2pose(xyz:np.ndarray, quat:np.ndarray) -> np.ndarray:
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :3] = Rot.from_quat(quat).as_matrix()
        pose_4x4[:3, 3] = xyz
        return pose_4x4

    def __del__(self) -> None:
        del self.context


if __name__ == "__main__":
    camera = Ultimate(role=['chest', 'left_foot', 'right_foot'], fps=200, name='Ultimate')
    streaming = False
    shm = False

    if not streaming:
        while True:
            data = camera.get()
            print(data)
            time.sleep(0.1)
    else:
        camera.collect_streaming(collect=True)
        camera.shm_streaming(shm='Ultimate' if shm else None)
        camera.start_streaming()

        cmd = input("quit? (enter): ")
        streaming_data = camera.stop_streaming()
        print(len(streaming_data["timestamp_ms"]))
        xyz = streaming_data["xyz"][-1]
        quat = streaming_data["quat"][-1]
        print(xyz, quat)
