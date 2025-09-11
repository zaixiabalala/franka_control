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
import serial

from r3kit.devices.encoder.base import EncoderBase
from r3kit.devices.encoder.pdcd.config import *
from r3kit.utils.vis import draw_time, draw_items

'''
Modified from: https://github.com/Galaxies99/easyrobot/blob/main/easyrobot/encoder/angle.py
'''


def hex2dex(e_hex):
    return int(e_hex, 16)

def hex2bin(e_hex):
    return bin(int(e_hex, 16))

def dex2bin(e_dex):
    return bin(e_dex)

def crc16(hex_num):
    """
    CRC16 verification
    :param hex_num:
    :return:
    """
    crc = '0xffff'
    crc16 = '0xA001'
    test = hex_num.split(' ')

    crc = hex2dex(crc)  
    crc16 = hex2dex(crc16) 
    for i in test:
        temp = '0x' + i
        temp = hex2dex(temp) 
        crc ^= temp  
        for i in range(8):
            if dex2bin(crc)[-1] == '0':
                crc >>= 1
            elif dex2bin(crc)[-1] == '1':
                crc >>= 1
                crc ^= crc16

    crc = hex(crc)
    crc_H = crc[2:4]
    crc_L = crc[-2:]

    return crc, crc_H, crc_L


class Angler(EncoderBase):
    def __init__(self, id:str=ANGLER_ID, index:List[int]=ANGLER_INDEX, fps:int=ANGLER_FPS, 
                 baudrate:int=ANGLER_BAUDRATE, gap:float=ANGLER_GAP, name:str='Angler') -> None:
        super().__init__(name=name)

        self._id = id
        self._index = index
        self._num = len(index)
        self._fps = fps
        self._baudrate = baudrate
        self._gap = gap

        # serial
        self.ser = serial.Serial(id, baudrate=baudrate)
        if not self.ser.is_open:
            raise RuntimeError('Fail to open the serial port, please check your settings again.')
        self.ser.flushInput()
        self.ser.flushOutput()

        # config
        self.angle_dtype = np.dtype(np.float64)
        self.angle_shape = (self._num,)

        # stream
        self.in_streaming = Event()
    
    def _read(self) -> Dict[str, Union[np.ndarray, float]]:
        self.ser.flushInput()

        for i in range(self._num):
            sendbytes = str(self._index[i]).zfill(2) + " 03 00 41 00 01"
            crc, crc_H, crc_L = crc16(sendbytes)
            sendbytes = sendbytes + ' ' + crc_L + ' ' + crc_H
            sendbytes = bytes.fromhex(sendbytes)
            self.ser.write(sendbytes)
            time.sleep(self._gap)
        
        re = self.ser.read(7 * self._num)
        receive_time = time.time() * 1000
        
        not_received = set(self._index)
        ret = np.zeros(self._num)
        for i in range(self._num):
            rei = re[7*i:7*(i+1)]
            assert (rei[0] in not_received) and (rei[1] == 3) and (rei[2] == 2)
            not_received.remove(rei[0])
            angle = 360 * (rei[3] * 256 + rei[4]) / 4096
            ret[self._index.index(rei[0])] = angle
        return {'angle': ret, 'timestamp_ms': receive_time}
    
    def get(self) -> Dict[str, Union[np.ndarray, float]]:
        if not self.in_streaming.is_set():
            data = self._read()
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                data = {}
                data['angle'] = self.streaming_data['angle'][-1]
                data['timestamp_ms'] = self.streaming_data['timestamp_ms'][-1]
                self.streaming_mutex.release()
            elif hasattr(self, "streaming_array"):
                data = {}
                with self.streaming_lock:
                    data['angle'] = np.copy(self.streaming_array["angle"])
                    data['timestamp_ms'] = self.streaming_array["timestamp_ms"].item()
            else:
                raise AttributeError
        return data
    
    def get_mean_data(self, n=10, name='angle') -> np.ndarray:
        assert name in ['angle'], 'name must be one of [angle]'
        tare_list = []
        count = 0
        while count < n:
            data = self.get()
            tare_list.append(data[name])
            count += 1
        tare = sum(tare_list) / n
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
                    "angle": [], 
                    "timestamp_ms": []
                }
            else:
                pass
        else:
            if callback is None:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()
                angle_memory_size = self.angle_dtype.itemsize * np.prod(self.angle_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                streaming_memory_size = angle_memory_size + timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
                self.streaming_array = {
                    "angle": np.ndarray(self.angle_shape, dtype=self.angle_dtype, buffer=self.streaming_memory.buf[:angle_memory_size]), 
                    "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[angle_memory_size:])
                }
                self.streaming_array_meta = {
                    "angle": (self.angle_shape, self.angle_dtype.name, (0, angle_memory_size)), 
                    "timestamp_ms": ((1,), np.float64.__name__, (angle_memory_size, angle_memory_size+timestamp_memory_size))
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
                "angle": [], 
                "timestamp_ms": []
            }
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "angle": [np.copy(self.streaming_array["angle"])], 
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
        assert len(streaming_data["angle"]) == len(streaming_data["timestamp_ms"])
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        if len(streaming_data["timestamp_ms"]) > 1:
            freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
            draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
        else:
            freq = 0
        np.save(os.path.join(save_path, "angle.npy"), np.array(streaming_data["angle"], dtype=float))
        draw_items(np.array(streaming_data["angle"], dtype=float), os.path.join(save_path, "angle.png"))
    
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
                "angle": [np.copy(self.streaming_array["angle"])], 
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
        else:
            raise AttributeError
        return streaming_data
    
    def reset_streaming(self) -> None:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            self.streaming_data['angle'].clear()
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
                "angle": [], 
                "timestamp_ms": []
            }
        else:
            self.streaming_manager = Manager()
            self.streaming_lock = self.streaming_manager.Lock()
            angle_memory_size = self.angle_dtype.itemsize * np.prod(self.angle_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            streaming_memory_size = angle_memory_size + timestamp_memory_size
            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
            self.streaming_array = {
                "angle": np.ndarray(self.angle_shape, dtype=self.angle_dtype, buffer=self.streaming_memory.buf[:angle_memory_size]), 
                "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[angle_memory_size:])
            }
            self.streaming_array_meta = {
                "angle": (self.angle_shape, self.angle_dtype.name, (0, angle_memory_size)), 
                "timestamp_ms": ((1,), np.float64.__name__, (angle_memory_size, angle_memory_size+timestamp_memory_size))
            }
    
    def _streaming_data(self, callback:Optional[callable]=None):
        while self.in_streaming.is_set():
            # fps
            time.sleep(1/self._fps - self._gap * self._num)

            # get data
            if not self._collect_streaming_data:
                continue
            data = self._read()
            if callback is None:
                if hasattr(self, "streaming_data"):
                    self.streaming_mutex.acquire()
                    self.streaming_data['angle'].append(data['angle'])
                    self.streaming_data['timestamp_ms'].append(data['timestamp_ms'])
                    self.streaming_mutex.release()
                elif hasattr(self, "streaming_array"):
                    with self.streaming_lock:
                        self.streaming_array["angle"][:] = data['angle'][:]
                        self.streaming_array["timestamp_ms"][:] = data['timestamp_ms']
                else:
                    raise AttributeError
            else:
                callback(deepcopy(data))
    
    @staticmethod
    def raw2angle(raw:np.ndarray) -> np.ndarray:
        result = []
        assert len(raw) > 100
        if np.any(raw[:10] < 1) and np.any(raw[:10] > 359):
            initial_angle = raw[0]
        else:
            assert np.quantile(raw[:10], 0.75) - np.quantile(raw[:10], 0.25) < 1, np.quantile(raw[:10], 0.75) - np.quantile(raw[:10], 0.25)
            initial_angle = np.median(raw[:10])
        count = 0
        result.append(raw[0] - initial_angle + 360 * count)
        for i in range(1, len(raw)):
            if abs(raw[i] - raw[i-1]) > 180:
                count +=  1 if raw[i] - raw[i-1] < 0 else -1
            result.append(raw[i] - initial_angle + 360 * count)
        return np.array(result)


if __name__ == "__main__":
    encoder = Angler(id='/dev/ttyUSB0', index=[1,2,3,4,5,6,7], baudrate=1000000, fps=30, gap=0.002, name='Angler')
    streaming = False
    shm = False

    zero_point = np.array([10, 90, 0, 0, 180, 0, 120])
    middle_point = np.array([203.03, 339.08, 190.9, 251.46, 7.21, 123.4, 293.64])

    if not streaming:
        while True:
            data = encoder.get()
            angle = np.copy(data['angle']) 
            angle[angle < zero_point] += 360
            angle -= middle_point

            print({'angle': angle, 'timestamp_ms': data['timestamp_ms']})
            time.sleep(0.1)
    else:
        encoder.collect_streaming(collect=True)
        encoder.shm_streaming(shm='Angler' if shm else None)
        encoder.start_streaming()

        cmd = input("quit? (enter): ")
        streaming_data = encoder.stop_streaming()
        print(len(streaming_data["timestamp_ms"]))
        
        # 对streaming数据进行角度跳变处理
        angles = np.array(streaming_data["angle"])
        for i in range(len(angles)):
            angle = np.copy(angles[i])
            angle[angle < zero_point] += 360
            angle -= middle_point
            angles[i] = angle
        
        print(angles[-1])  # 打印最后一个处理后的角度
