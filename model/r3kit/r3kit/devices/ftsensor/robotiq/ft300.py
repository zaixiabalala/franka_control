import os
from typing import List, Dict, Union, Optional
import time
import gc
import numpy as np
from threading import Thread, Lock, Event
from multiprocessing import shared_memory, Manager
from copy import deepcopy
from functools import partial
import serial
from pymodbus.client import ModbusSerialClient
from pymodbus import FramerType

from r3kit.devices.ftsensor.base import FTSensorBase
from r3kit.devices.ftsensor.robotiq.config import *
from r3kit.utils.vis import draw_time, draw_items

'''
Modified from: https://github.com/hygradme/ft300python/tree/main/ft300python
'''

def uint_to_int(register):
    register_bytes = register.to_bytes(2, byteorder='little')
    return int.from_bytes(register_bytes, byteorder='little', signed=True)


class FT300(FTSensorBase):
    '''
    Applicable to FT300 and FT300-S
    '''
    def __init__(self, id:str=FT300_ID, fps:int=FT300_FPS, name:str='FT300') -> None:
        super().__init__(name=name)

        self._id = id
        self._fps = fps

        # set up client
        self._client = ModbusSerialClient(
            port=id,
            framer=FramerType.RTU,
            stopbits=1,
            bytesize=8,
            parity="N",
            baudrate=FT300_BAUDRATE,
            timeout=1
        )
        try:
            self._client.connect()
        except Exception as e:
            print(f'An error occurred: {e}')
        
        # config
        self.ft_dtype = np.dtype(np.float64)
        self.ft_shape = (6,)

        # stream
        self.in_streaming = Event()
    
    def __del__(self):
        if self.in_streaming.is_set():
            for i in range(50):
                self._ser.write([0xff])
            self._ser.close()
        else:
            self._client.close()
    
    def _read(self) -> Dict[str, Union[float, np.ndarray]]:
        result = self._client.read_holding_registers(
            address=FT300_REGISTER_DICT["F_x"], count=6, slave=9)
        receive_time = time.time() * 1000
        ft = [uint_to_int(ft) / coef for ft, coef in zip(result.registers, FT300_FT_COEF)]
        return {'ft': np.array(ft), 'timestamp_ms': receive_time}
    
    def get(self) -> Dict[str, Union[float, np.ndarray]]:
        if not self.in_streaming.is_set():
            data = self._read()
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                data = {}
                data['ft'] = self.streaming_data['ft'][-1]
                data['timestamp_ms'] = self.streaming_data['timestamp_ms'][-1]
                self.streaming_mutex.release()
            elif hasattr(self, "streaming_array"):
                data = {}
                with self.streaming_lock:
                    data['ft'] = np.copy(self.streaming_array['ft'])
                    data['timestamp_ms'] = self.streaming_array['timestamp_ms'].item()
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
                    "ft": [], 
                    "timestamp_ms": []
                }
            else:
                pass
        else:
            if callback is None:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()
                ft_memory_size = self.ft_dtype.itemsize * np.prod(self.ft_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                streaming_memory_size = ft_memory_size + timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
                self.streaming_array = {
                    "ft": np.ndarray(self.ft_shape, dtype=self.ft_dtype, buffer=self.streaming_memory.buf[0:ft_memory_size]), 
                    "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[ft_memory_size:])
                }
                self.streaming_array_meta = {
                    "ft": (self.ft_shape, self.ft_dtype.name, (0, ft_memory_size)), 
                    "timestamp_ms": ((1,), np.float64.__name__, (ft_memory_size, ft_memory_size+timestamp_memory_size))
                }
            else:
                pass
        # Write 0x200 in stream register to start data streaming
        try:
            self._client.write_registers(address=FT300_REGISTER_DICT['Stream'], values=FT300_STREAM_FLAG, slave=9)
        except Exception as e:
            print(f'An error occurred: {e}')
        self._ser = self._client.socket
        # Read serial buffer until founding the bytes [0x20,0x4e]
        self._ser.reset_input_buffer()
        # Ignore first several values and get value for zero ft in the end
        for i in range(10):
            self._ser.read_until(FT300_STREAM_START)
        self.thread = Thread(target=partial(self._streaming_data, callback=callback), daemon=True)
        self.thread.start()
    
    def stop_streaming(self) -> Dict[str, Union[List[np.ndarray], List[float]]]:
        self.in_streaming.clear()
        self.thread.join()
        for i in range(50):
            self._ser.write([0xff])
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            self.streaming_data = {
                "ft": [], 
                "timestamp_ms": []
            }
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "ft": [np.copy(self.streaming_array["ft"])], 
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
        assert len(streaming_data["ft"]) ==  len(streaming_data["timestamp_ms"])
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        if len(streaming_data["timestamp_ms"]) > 1:
            freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
            draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
        else:
            freq = 0
        np.save(os.path.join(save_path, "ft.npy"), np.array(streaming_data["ft"], dtype=float))
        draw_items(np.array(streaming_data["ft"], dtype=float), os.path.join(save_path, "ft.png"))
    
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
                "ft": [np.copy(self.streaming_array["ft"])], 
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
        else:
            raise AttributeError
        return streaming_data
    
    def reset_streaming(self) -> None:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            self.streaming_data['ft'].clear()
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
                "ft": [], 
                "timestamp_ms": []
            }
        else:
            self.streaming_manager = Manager()
            self.streaming_lock = self.streaming_manager.Lock()
            ft_memory_size = self.ft_dtype.itemsize * np.prod(self.ft_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            streaming_memory_size = ft_memory_size + timestamp_memory_size
            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
            self.streaming_array = {
                "ft": np.ndarray(self.ft_shape, dtype=self.ft_dtype, buffer=self.streaming_memory.buf[0:ft_memory_size]), 
                "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[ft_memory_size:])
            }
            self.streaming_array_meta = {
                "ft": (self.ft_shape, self.ft_dtype.name, (0, ft_memory_size)), 
                "timestamp_ms": ((1,), np.float64.__name__, (ft_memory_size, ft_memory_size+timestamp_memory_size))
            }
    
    def _streaming_data(self, callback:Optional[callable]=None):
        while self.in_streaming.is_set():
            # fps
            time.sleep(1/self._fps)

            # get data
            if not self._collect_streaming_data:
                continue
            raw_bytes = bytearray(self._ser.read_until(FT300_STREAM_START))
            receive_time = time.time() * 1000
            ft = [int.from_bytes(
                raw_bytes[i*2: i*2+2],
                byteorder='little',
                signed=True) / FT300_FT_COEF[i] for i in range(0, 6)
            ]
            data = {'ft': np.array(ft), 'timestamp_ms': receive_time}
            if callback is None:
                if hasattr(self, "streaming_data"):
                    self.streaming_mutex.acquire()
                    self.streaming_data['ft'].append(data['ft'])
                    self.streaming_data['timestamp_ms'].append(data['timestamp_ms'])
                    self.streaming_mutex.release()
                elif hasattr(self, "streaming_array"):
                    with self.streaming_lock:
                        self.streaming_array["ft"][:] = data['ft'][:]
                        self.streaming_array["timestamp_ms"][:] = data['timestamp_ms']
                else:
                    raise AttributeError
            else:
                callback(deepcopy(data))
    
    @staticmethod
    def raw2tare(raw_ft:np.ndarray, tare:Dict[str, Union[float, np.ndarray]], pose:np.ndarray) -> np.ndarray:
        '''
        raw_ft: raw force torque data
        pose: 3x3 rotation matrix from ft300 to base
        '''
        raw_f, raw_t = raw_ft[:3], raw_ft[3:]
        f = raw_f - tare['f0']
        f -= np.linalg.inv(pose) @ np.array([0., 0., -9.8 * tare['m']])
        t = raw_t - tare['t0']
        t -= np.linalg.inv(pose) @ np.cross(np.linalg.inv(pose) @ np.array(tare['c']), np.array([0., 0., -9.8 * tare['m']]))
        return np.concatenate([f, t])


if __name__ == '__main__':
    sensor = FT300(id='/dev/ttyUSB0', fps=100, name='FT300')
    streaming = True
    shm = False

    if not streaming:
        while True:
            data = sensor.get()
            print(data)
            time.sleep(0.1)
    else:
        sensor.collect_streaming(collect=True)
        sensor.shm_streaming(shm='FT300' if shm else None)
        sensor.start_streaming()

        cmd = input("quit? (enter): ")
        streaming_data = sensor.stop_streaming()
        print(len(streaming_data["timestamp_ms"]))
        data = {
            "ft": streaming_data["ft"][-1]
        }
        print(data)
