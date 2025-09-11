from typing import List
import os
from multiprocessing import shared_memory
import subprocess
import getpass
import psutil


def clear_shared_memory(possible_shm_names:List[str]):
    for possible_shm_name in possible_shm_names:
        try:
            shm = shared_memory.SharedMemory(name=possible_shm_name)
            shm.close()
            shm.unlink()
            print(f"Clear: {possible_shm_name}")
        except FileNotFoundError:
            continue

def clear_memory():
    memory_info = psutil.virtual_memory()
    print(memory_info.used / (1024 ** 3), memory_info.total / (1024 ** 3))
    password = getpass.getpass("Enter your sudo password: ")
    command = f"echo {password} | sudo -S sync"
    subprocess.run(command, shell=True, text=True)
    command = f"echo {password} | sudo -S sysctl -w vm.drop_caches=3"
    subprocess.run(command, shell=True, text=True)
    memory_info = psutil.virtual_memory()
    print(memory_info.used / (1024 ** 3), memory_info.total / (1024 ** 3))


if __name__ == '__main__':
    clear_shared_memory(['l515', 't265', 'd415', 'pyft', 'angler', 'ultimate', 'ft300', 'obs_buffer', 'act_buffer'])
    clear_memory()
