from typing import List, Dict
import time
import numpy as np
from multiprocessing import shared_memory, Manager


class ObsBuffer:
    def __init__(self, num_obs:int, obs_dict:Dict[str, tuple], create:bool=False) -> None:
        self.num_obs = num_obs
        self.obs_dict = obs_dict
        self.obs_names = set(obs_dict.keys())

        item_memory_size = {}
        for name in sorted(self.obs_dict.keys()):
            shape, dtype_name = self.obs_dict[name]
            dtype = np.dtype(getattr(np, dtype_name))
            item_memory_size[name] = dtype.itemsize * np.prod(shape).item()
        step_memory_size = sum(item_memory_size.values())
        flag_shape, flag_dtype = (1,), np.dtype('bool')
        flag_memory_size = flag_dtype.itemsize * np.prod(flag_shape).item()
        length_shape, length_dtype = (1,), np.dtype('int')
        length_memory_size = length_dtype.itemsize * np.prod(length_shape).item()
        idx_shape, idx_dtype = (1,), np.dtype('int')
        idx_memory_size = idx_dtype.itemsize * np.prod(idx_shape).item()
        all_memory_size = step_memory_size * self.num_obs + flag_memory_size + length_memory_size + idx_memory_size
        
        self.shm_manager = Manager()
        self.shm_lock = self.shm_manager.Lock()
        self.create = create
        if create:
            self.shm_memory = shared_memory.SharedMemory(create=True, name='obs_buffer', size=all_memory_size)
        else:
            while True:
                try:
                    self.shm_memory = shared_memory.SharedMemory(name='obs_buffer')
                    break
                except FileNotFoundError:
                    time.sleep(0.1)
        self.shm_arrays = []
        offset = 0
        for i in range(self.num_obs):
            shm_array = {}
            for name in sorted(self.obs_dict.keys()):
                shape, dtype_name = self.obs_dict[name]
                dtype = np.dtype(getattr(np, dtype_name))
                shm_array[name] = np.ndarray(shape, dtype=dtype, buffer=self.shm_memory.buf[offset:offset+item_memory_size[name]])
                offset += item_memory_size[name]
            self.shm_arrays.append(shm_array)
        self.shm_flag = np.ndarray(flag_shape, dtype=flag_dtype, buffer=self.shm_memory.buf[offset:offset+flag_memory_size])
        offset += flag_memory_size
        self.shm_length = np.ndarray(length_shape, dtype=length_dtype, buffer=self.shm_memory.buf[offset:offset+length_memory_size])
        offset += length_memory_size
        self.shm_idx = np.ndarray(idx_shape, dtype=idx_dtype, buffer=self.shm_memory.buf[offset:offset+idx_memory_size])
        offset += idx_memory_size
        
        if self.create:
            self.setf(False)
            self.setl(0)
            self.seti(0)
    
    def __len__(self) -> int:
        return self.getl()
    
    def __del__(self) -> None:
        if hasattr(self, "shm_memory"):
            if self.create:
                self.shm_memory.close()
                self.shm_memory.unlink()
            else:
                self.shm_memory.close()
    
    def add1(self, obs:Dict[str, np.ndarray]) -> None:
        assert set(obs.keys()) == self.obs_names
        idx = self.geti()
        length = self.getl()
        with self.shm_lock:
            for name in obs.keys():
                self.shm_arrays[idx][name][:] = obs[name]
        self.seti((idx + 1) % self.num_obs)
        self.setl(min(length + 1, self.num_obs))
    
    def getn(self) -> List[Dict[str, np.ndarray]]:
        idx = self.geti()
        length = self.getl()
        with self.shm_lock:
            obs = []
            for i in range(length):
                step_obs = {}
                for name in self.obs_names:
                    step_obs[name] = np.copy(self.shm_arrays[(idx + i) % self.num_obs][name])
                obs.append(step_obs)
        return obs
    
    def setf(self, flag:bool) -> None:
        with self.shm_lock:
            self.shm_flag[:] = flag
    
    def getf(self) -> bool:
        with self.shm_lock:
            return self.shm_flag[:].item()
    
    def setl(self, length:int) -> None:
        with self.shm_lock:
            self.shm_length[:] = length
    
    def getl(self) -> int:
        with self.shm_lock:
            return self.shm_length[:].item()
    
    def seti(self, idx:int) -> None:
        with self.shm_lock:
            self.shm_idx[:] = idx
    
    def geti(self) -> int:
        with self.shm_lock:
            return self.shm_idx[:].item()


class ActBuffer:
    def __init__(self, num_act:int, act_dict:Dict[str, tuple], create:bool=False) -> None:
        self.num_act = num_act
        self.act_dict = act_dict
        self.act_names = set(act_dict.keys())

        item_memory_size = {}
        for name in sorted(self.act_dict.keys()):
            shape, dtype_name = self.act_dict[name]
            dtype = np.dtype(getattr(np, dtype_name))
            item_memory_size[name] = dtype.itemsize * np.prod(shape).item()
        step_memory_size = sum(item_memory_size.values())
        flag_shape, flag_dtype = (1,), np.dtype('bool')
        flag_memory_size = flag_dtype.itemsize * np.prod(flag_shape).item()
        length_shape, length_dtype = (1,), np.dtype('int')
        length_memory_size = length_dtype.itemsize * np.prod(length_shape).item()
        idx_shape, idx_dtype = (1,), np.dtype('int')
        idx_memory_size = idx_dtype.itemsize * np.prod(idx_shape).item()
        all_memory_size = step_memory_size * self.num_act + flag_memory_size + length_memory_size + idx_memory_size
        
        self.shm_manager = Manager()
        self.shm_lock = self.shm_manager.Lock()
        self.create = create
        if create:
            self.shm_memory = shared_memory.SharedMemory(create=True, name='act_buffer', size=all_memory_size)
        else:
            while True:
                try:
                    self.shm_memory = shared_memory.SharedMemory(name='act_buffer')
                    break
                except FileNotFoundError:
                    time.sleep(0.1)
        self.shm_arrays = []
        offset = 0
        for i in range(self.num_act):
            shm_array = {}
            for name in sorted(self.act_dict.keys()):
                shape, dtype_name = self.act_dict[name]
                dtype = np.dtype(getattr(np, dtype_name))
                shm_array[name] = np.ndarray(shape, dtype=dtype, buffer=self.shm_memory.buf[offset:offset+item_memory_size[name]])
                offset += item_memory_size[name]
            self.shm_arrays.append(shm_array)
        self.shm_flag = np.ndarray(flag_shape, dtype=flag_dtype, buffer=self.shm_memory.buf[offset:offset+flag_memory_size])
        offset += flag_memory_size
        self.shm_length = np.ndarray(length_shape, dtype=length_dtype, buffer=self.shm_memory.buf[offset:offset+length_memory_size])
        offset += length_memory_size
        self.shm_idx = np.ndarray(idx_shape, dtype=idx_dtype, buffer=self.shm_memory.buf[offset:offset+idx_memory_size])
        offset += idx_memory_size
        
        if self.create:
            self.setf(False)
            self.setl(0)
            self.seti(0)
    
    def __len__(self) -> int:
        return self.getl()
    
    def __del__(self) -> None:
        if hasattr(self, "shm_memory"):
            if self.create:
                self.shm_memory.close()
                self.shm_memory.unlink()
            else:
                self.shm_memory.close()
    
    def addn(self, act:List[Dict[str, np.ndarray]]) -> None:
        assert len(act) == self.num_act
        idx = self.geti()
        assert idx == 0
        with self.shm_lock:
            for i in range(len(act)):
                assert set(act[i].keys()) == self.act_names
                for name in act[i].keys():
                    self.shm_arrays[i][name][:] = act[i][name]
        self.setl(self.num_act)
    
    def get1(self) -> Dict[str, np.ndarray]:
        idx = self.geti()
        with self.shm_lock:
            act = {}
            for name in self.act_names:
                act[name] = np.copy(self.shm_arrays[idx][name])
        self.seti(idx + 1)
        return act
    
    def setf(self, flag:bool) -> None:
        with self.shm_lock:
            self.shm_flag[:] = flag
    
    def getf(self) -> bool:
        with self.shm_lock:
            return self.shm_flag[:].item()
    
    def setl(self, length:int) -> None:
        with self.shm_lock:
            self.shm_length[:] = length
    
    def getl(self) -> int:
        with self.shm_lock:
            return self.shm_length[:].item()
    
    def seti(self, idx:int) -> None:
        with self.shm_lock:
            self.shm_idx[:] = idx
    
    def geti(self) -> int:
        with self.shm_lock:
            return self.shm_idx[:].item()
