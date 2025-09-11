from r3kit.devices.base import DeviceBase


class RobotBase(DeviceBase):
    def __init__(self, name:str='') -> None:
        super().__init__(name)
        pass
