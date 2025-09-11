import os
from typing import List, Dict, Union, Optional
import struct
import time
import gc
import numpy as np
from threading import Thread, Lock, Event
from multiprocessing import shared_memory, Manager
from copy import deepcopy
from functools import partial
import pysoem

from r3kit.devices.ftsensor.base import FTSensorBase
from r3kit.devices.ftsensor.flexiv.config import *
from r3kit.utils.vis import draw_time, draw_items


class Pyft(FTSensorBase):
    def __init__(self, id:str=PYFT_ID, fps:int=PYFT_FPS, raw:bool=False, info:bool=True, name:str='Pyft') -> None:
        super().__init__(name=name)

        self._id = id
        self._fps = fps
        self._raw = raw
        self._info = info

        # set up master
        self._master = pysoem.Master()
        try:
            self._master.open(id)
        except Exception as ex:
            print(f'An error occurred: {ex}')
        if self._master.config_init() > 0:
            for device in self._master.slaves:
                if self._info:
                    print(f'FTSensor Network ID: {id}')
                    print(f'FTSensor Data Fps: {fps}')
                    print(f'FTSensor Device Name: {device.name}')
                    print(f'FTSensor Device Product Code: {device.id}')
                    print(f'FTSensor Device Vendor ID: {device.man}')
                # self._master.close()

            self._master.config_map()
            self._master.config_dc()
        # set control word
        self._master.slaves[0].sdo_write(0x7000, 0x01, b'\x01\x00\x00\x00')

        # config
        if self._raw:
            self._ft_config = np.array(PYFT_RAW_CONFIG)
            self._ft_offset = np.array(PYFT_RAW_OFFSET)
        else:
            self._ft_config = np.array(PYFT_CONFIG)
        initial_start_time = time.time()
        data = self._read()
        while data is None:
            data = self._read()
            if time.time() - initial_start_time > 3:
                raise RuntimeError("FTSensor cannot read data")
        self.status_dtype = np.dtype(np.int32)
        self.status_shape = (1,)
        self.ft_dtype = data['ft'].dtype
        self.ft_shape = data['ft'].shape
        self.acc_dtype = data['acc'].dtype
        self.acc_shape = data['acc'].shape
        self.gyro_dtype = data['gyro'].dtype
        self.gyro_shape = data['gyro'].shape
        self.temp_dtype = data['temp'].dtype
        self.temp_shape = data['temp'].shape

        # stream
        self.in_streaming = Event()
    
    def __del__(self):
        self._master.close()
    
    def _read(self) -> Optional[Dict[str, Union[int, np.ndarray, float]]]:
        self._master.send_processdata()
        self._master.receive_processdata()
        receive_time = time.time() * 1000
        inputs = self._master.slaves[0].input               # assuming we are reading from the first slave

        if len(inputs) >= 44:                               # check if we received enough data
            # parse data
            if self._raw:
                status_word, hall_sensors, accel_data, gyro_data, temps = \
                    self._parse_raw_sensor_data(inputs)     # extract each value based on the byte offsets and sizes from the C++ class
                ft = self._ft_config.reshape(6,6).T @ (np.array(hall_sensors) - self._ft_offset)
            else:
                status_word, hall_sensors, accel_data, gyro_data, temps = \
                    self._parse_sensor_data(inputs)
                ft = self._ft_config.reshape(8,6).T @ np.array(hall_sensors)
            ft = ft.tolist()

            data = {}
            data['status'] = status_word
            data['ft'] = np.array(ft, dtype=np.float64)
            data['acc'] = np.array(accel_data, dtype=np.float64)
            data['gyro'] = np.array(gyro_data, dtype=np.float64)
            data['temp'] = np.array(temps, dtype=np.float64)
            data['timestamp_ms'] = receive_time
            # print(data['ft'])
            return data
        return None
    
    def get(self) -> Optional[Dict[str, Union[int, np.ndarray, float]]]:
        if not self.in_streaming.is_set():
            data = self._read()
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                data = {}
                data['status'] = self.streaming_data['status'][-1]
                data['ft'] = self.streaming_data['ft'][-1]
                data['acc'] = self.streaming_data['acc'][-1]
                data['gyro'] = self.streaming_data['gyro'][-1]
                data['temp'] = self.streaming_data['temp'][-1]
                data['timestamp_ms'] = self.streaming_data['timestamp_ms'][-1]
                self.streaming_mutex.release()
            elif hasattr(self, "streaming_array"):
                data = {}
                with self.streaming_lock:
                    data['status'] = self.streaming_array["status"].item()
                    data['ft'] = np.copy(self.streaming_array["ft"])
                    data['acc'] = np.copy(self.streaming_array["acc"])
                    data['gyro'] = np.copy(self.streaming_array["gyro"])
                    data['temp'] = np.copy(self.streaming_array["temp"])
                    data['timestamp_ms'] = self.streaming_array["timestamp_ms"].item()
            else:
                raise AttributeError
        return data
    
    def get_mean_data(self, n=10, name='ft') -> np.ndarray:
        assert name in ['ft','acc','gyro','temp'], 'name must be one of [ft,acc,gyro,temp]'
        tare_list = []
        count = 0
        while count < n:
            data = self.get()
            if data is not None:
                tare_list.append(data[name])
                count += 1
        tare = np.mean(tare_list, axis=0)
        return tare
    
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
                    "status": [], 
                    "ft": [], 
                    "acc": [], 
                    "gyro": [], 
                    "temp": [], 
                    "timestamp_ms": []
                }
            else:
                pass
        else:
            if callback is None:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()
                status_memory_size = self.status_dtype.itemsize * np.prod(self.status_shape).item()
                ft_memory_size = self.ft_dtype.itemsize * np.prod(self.ft_shape).item()
                acc_memory_size = self.acc_dtype.itemsize * np.prod(self.acc_shape).item()
                gyro_memory_size = self.gyro_dtype.itemsize * np.prod(self.gyro_shape).item()
                temp_memory_size = self.temp_dtype.itemsize * np.prod(self.temp_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                streaming_memory_size = status_memory_size + ft_memory_size + acc_memory_size + gyro_memory_size + temp_memory_size + timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
                self.streaming_array = {
                    "status": np.ndarray(self.status_shape, dtype=self.status_dtype, buffer=self.streaming_memory.buf[:status_memory_size]), 
                    "ft": np.ndarray(self.ft_shape, dtype=self.ft_dtype, buffer=self.streaming_memory.buf[status_memory_size:status_memory_size+ft_memory_size]), 
                    "acc": np.ndarray(self.acc_shape, dtype=self.acc_dtype, buffer=self.streaming_memory.buf[status_memory_size+ft_memory_size:status_memory_size+ft_memory_size+acc_memory_size]), 
                    "gyro": np.ndarray(self.gyro_shape, dtype=self.gyro_dtype, buffer=self.streaming_memory.buf[status_memory_size+ft_memory_size+acc_memory_size:status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size]), 
                    "temp": np.ndarray(self.temp_shape, dtype=self.temp_dtype, buffer=self.streaming_memory.buf[status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size:status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size+temp_memory_size]), 
                    "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size+temp_memory_size:])
                }
                self.streaming_array_meta = {
                    "status": (self.status_shape, self.status_dtype.name, (0, status_memory_size)), 
                    "ft": (self.ft_shape, self.ft_dtype.name, (status_memory_size, status_memory_size+ft_memory_size)), 
                    "acc": (self.acc_shape, self.acc_dtype.name, (status_memory_size+ft_memory_size, status_memory_size+ft_memory_size+acc_memory_size)), 
                    "gyro": (self.gyro_shape, self.gyro_dtype.name, (status_memory_size+ft_memory_size+acc_memory_size, status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size)), 
                    "temp": (self.temp_shape, self.temp_dtype.name, (status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size, status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size+temp_memory_size)), 
                    "timestamp_ms": ((1,), np.float64.__name__, (status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size+temp_memory_size, status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size+temp_memory_size+timestamp_memory_size))
                }
            else:
                pass
        self.thread = Thread(target=partial(self._streaming_data, callback=callback), daemon=True)
        self.thread.start()
    
    def stop_streaming(self) -> Dict[str, Union[List[int], List[np.ndarray], List[float]]]:
        self.in_streaming.clear()
        self.thread.join()
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            self.streaming_data = {
                "status": [], 
                "ft": [], 
                "acc": [], 
                "gyro": [], 
                "temp": [], 
                "timestamp_ms": []
            }
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "status": [self.streaming_array["status"].item()], 
                "ft": [np.copy(self.streaming_array["ft"])], 
                "acc": [np.copy(self.streaming_array["acc"])], 
                "gyro": [np.copy(self.streaming_array["gyro"])], 
                "temp": [np.copy(self.streaming_array["temp"])], 
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
        assert len(streaming_data["status"]) == len(streaming_data["ft"]) == \
            len(streaming_data["acc"]) == len(streaming_data["gyro"]) == \
            len(streaming_data["temp"]) == len(streaming_data["timestamp_ms"])
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        if len(streaming_data["timestamp_ms"]) > 1:
            freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
            draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
        else:
            freq = 0
        np.save(os.path.join(save_path, "status.npy"), np.array(streaming_data["status"], dtype=int))
        np.save(os.path.join(save_path, "ft.npy"), np.array(streaming_data["ft"], dtype=float))
        draw_items(np.array(streaming_data["ft"], dtype=float), os.path.join(save_path, "ft.png"))
        np.save(os.path.join(save_path, "acc.npy"), np.array(streaming_data["acc"], dtype=float))
        np.save(os.path.join(save_path, "gyro.npy"), np.array(streaming_data["gyro"], dtype=float))
        np.save(os.path.join(save_path, "temp.npy"), np.array(streaming_data["temp"], dtype=float))
        draw_items(np.array(streaming_data["temp"], dtype=float), os.path.join(save_path, "temp.png"))
    
    def collect_streaming(self, collect:bool=True) -> None:
        self._collect_streaming_data = collect
    
    def shm_streaming(self, shm:Optional[str]=None) -> None:
        # NOTE: only valid for non-custom-callback
        assert (not self.in_streaming.is_set()) or (not self._collect_streaming_data)
        self._shm = shm
    
    def get_streaming(self) -> Dict[str, Union[List[int], List[np.ndarray], List[float]]]:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "status": [self.streaming_array["status"].item()], 
                "ft": [np.copy(self.streaming_array["ft"])], 
                "acc": [np.copy(self.streaming_array["acc"])], 
                "gyro": [np.copy(self.streaming_array["gyro"])], 
                "temp": [np.copy(self.streaming_array["temp"])], 
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
        else:
            raise AttributeError
        return streaming_data
    
    def reset_streaming(self) -> None:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            self.streaming_data['status'].clear()
            self.streaming_data['ft'].clear()
            self.streaming_data['acc'].clear()
            self.streaming_data['gyro'].clear()
            self.streaming_data['temp'].clear()
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
                "status": [], 
                "ft": [], 
                "acc": [], 
                "gyro": [], 
                "temp": [], 
                "timestamp_ms": []
            }
        else:
            self.streaming_manager = Manager()
            self.streaming_lock = self.streaming_manager.Lock()
            status_memory_size = self.status_dtype.itemsize * np.prod(self.status_shape).item()
            ft_memory_size = self.ft_dtype.itemsize * np.prod(self.ft_shape).item()
            acc_memory_size = self.acc_dtype.itemsize * np.prod(self.acc_shape).item()
            gyro_memory_size = self.gyro_dtype.itemsize * np.prod(self.gyro_shape).item()
            temp_memory_size = self.temp_dtype.itemsize * np.prod(self.temp_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            streaming_memory_size = status_memory_size + ft_memory_size + acc_memory_size + gyro_memory_size + temp_memory_size + timestamp_memory_size
            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
            self.streaming_array = {
                "status": np.ndarray(self.status_shape, dtype=self.status_dtype, buffer=self.streaming_memory.buf[:status_memory_size]), 
                "ft": np.ndarray(self.ft_shape, dtype=self.ft_dtype, buffer=self.streaming_memory.buf[status_memory_size:status_memory_size+ft_memory_size]), 
                "acc": np.ndarray(self.acc_shape, dtype=self.acc_dtype, buffer=self.streaming_memory.buf[status_memory_size+ft_memory_size:status_memory_size+ft_memory_size+acc_memory_size]), 
                "gyro": np.ndarray(self.gyro_shape, dtype=self.gyro_dtype, buffer=self.streaming_memory.buf[status_memory_size+ft_memory_size+acc_memory_size:status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size]), 
                "temp": np.ndarray(self.temp_shape, dtype=self.temp_dtype, buffer=self.streaming_memory.buf[status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size:status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size+temp_memory_size]), 
                "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size+temp_memory_size:])
            }
            self.streaming_array_meta = {
                "status": (self.status_shape, self.status_dtype.name, (0, status_memory_size)), 
                "ft": (self.ft_shape, self.ft_dtype.name, (status_memory_size, status_memory_size+ft_memory_size)), 
                "acc": (self.acc_shape, self.acc_dtype.name, (status_memory_size+ft_memory_size, status_memory_size+ft_memory_size+acc_memory_size)), 
                "gyro": (self.gyro_shape, self.gyro_dtype.name, (status_memory_size+ft_memory_size+acc_memory_size, status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size)), 
                "temp": (self.temp_shape, self.temp_dtype.name, (status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size, status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size+temp_memory_size)), 
                "timestamp_ms": ((1,), np.float64.__name__, (status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size+temp_memory_size, status_memory_size+ft_memory_size+acc_memory_size+gyro_memory_size+temp_memory_size+timestamp_memory_size))
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
                        self.streaming_data['status'].append(data['status'])
                        self.streaming_data['ft'].append(data['ft'])
                        self.streaming_data['acc'].append(data['acc'])
                        self.streaming_data['gyro'].append(data['gyro'])
                        self.streaming_data['temp'].append(data['temp'])
                        self.streaming_data['timestamp_ms'].append(data['timestamp_ms'])
                        self.streaming_mutex.release()
                    elif hasattr(self, "streaming_array"):
                        with self.streaming_lock:
                            self.streaming_array["status"][:] = data['status']
                            self.streaming_array["ft"][:] = data['ft'][:]
                            self.streaming_array["acc"][:] = data['acc'][:]
                            self.streaming_array["gyro"][:] = data['gyro'][:]
                            self.streaming_array["temp"][:] = data['temp'][:]
                            self.streaming_array["timestamp_ms"][:] = data['timestamp_ms']
                    else:
                        raise AttributeError
                else:
                    callback(deepcopy(data))
    
    @staticmethod
    def _parse_raw_sensor_data(inputs):
        status_word = struct.unpack_from('<H', inputs, 0)[0]
        hall_sensors = struct.unpack_from('<6H', inputs, 2)
        accel_data = struct.unpack_from('<3f', inputs, 14)
        gyro_data = struct.unpack_from('<3f', inputs, 26)
        temps = struct.unpack_from('<2f', inputs, 38)
        return status_word, hall_sensors, accel_data, gyro_data, temps
    
    @staticmethod
    def _parse_sensor_data(inputs):
        status_word = struct.unpack_from('<H', inputs, 0)[0]
        hall_sensors = struct.unpack_from('<8H', inputs, 2)
        accel_data = struct.unpack_from('<3f', inputs, 18)
        gyro_data = struct.unpack_from('<3f', inputs, 30)
        temps = struct.unpack_from('<2f', inputs, 42)
        return status_word, hall_sensors, accel_data, gyro_data, temps
    
    @staticmethod
    def raw2tare(raw_ft:np.ndarray, tare:Dict[str, Union[float, np.ndarray]], pose:np.ndarray) -> np.ndarray:
        '''
        raw_ft: raw force torque data
        pose: 3x3 rotation matrix from pyft to base
        '''
        raw_f, raw_t = raw_ft[:3], raw_ft[3:]
        f = raw_f - tare['f0']
        f -= np.linalg.inv(pose) @ np.array([0., 0., -9.8 * tare['m']])
        t = raw_t - tare['t0']
        t -= np.linalg.inv(pose) @ np.cross(np.linalg.inv(pose) @ np.array(tare['c']), np.array([0., 0., -9.8 * tare['m']]))
        return np.concatenate([f, t])


if __name__ == "__main__":
    sensor = Pyft(id='enp86s0', fps=200, raw=False, info=True, name='Pyft')
    streaming = False
    shm = False

    if not streaming:
        while True:
            data = sensor.get()
            print(data)
            time.sleep(0.1)
    else:
        sensor.collect_streaming(collect=True)
        sensor.shm_streaming(shm='Pyft' if shm else None)
        sensor.start_streaming()

        cmd = input("quit? (enter): ")
        streaming_data = sensor.stop_streaming()
        print(len(streaming_data["timestamp_ms"]))
        data = {
            "status": streaming_data["status"][-1], 
            "ft": streaming_data["ft"][-1], 
            "acc": streaming_data["acc"][-1], 
            "gyro": streaming_data["gyro"][-1], 
            "temp": streaming_data["temp"][-1]
        }
        print(data)
