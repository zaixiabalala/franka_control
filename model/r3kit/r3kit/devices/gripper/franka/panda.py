try:
    from frankx import Gripper
except ImportError:
    print("Gripper Franka Panda needs `frankx")
    raise ImportError

from r3kit.devices.gripper.base import GripperBase
from r3kit.devices.gripper.franka.config import *


class Panda(GripperBase):
    MAX_WIDTH:float = 0.08

    def __init__(self, ip:str=PANDA_IP, name:str='Panda') -> None:
        super().__init__(name)

        self.gripper = Gripper(ip)
        self.set_force(PANDA_FORCE)
        self.set_speed(PANDA_SPEED)
        
    def set_force(self, force:float) -> None:
        self.gripper.gripper_force = force
    
    def set_speed(self, speed:float) -> None:
        self.gripper.gripper_speed = speed
    
    def read(self) -> float:
        '''
        width: gripper width in meter
        '''
        return self.gripper.width()
    
    def move(self, width:float) -> None:
        '''
        width: gripper width in meter
        '''
        self.gripper.move(min(max(width, 0.0), self.MAX_WIDTH), self.gripper.gripper_speed)

    def is_grasping(self) -> bool:
        return self.gripper.is_grasping()
    
    def open(self) -> None:
        # open to max width
        self.gripper.open()
    
    def release(self, width:float) -> None:
        # release after grasping
        self.gripper.release(min(max(width, 0.0), self.MAX_WIDTH))
    
    def grasp(self) -> bool:
        # grasp
        is_graspped = self.gripper.clamp()
        return is_graspped


if __name__ == "__main__":
    gripper = Panda('172.16.0.2', 'panda')

    gripper.grasp()
    gripper.open()
    is_graspped = gripper.grasp()
    is_graspped = is_graspped and gripper.is_grasping()
    print("is_graspped:", is_graspped)
    gripper.move(0.05)
    gripper_width = gripper.read()
    print("gripper width:", gripper_width)
