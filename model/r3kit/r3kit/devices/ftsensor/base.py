from r3kit.devices.base import DeviceBase


class FTSensorBase(DeviceBase):
    def __init__(self, name:str='') -> None:
        super().__init__(name)
        pass
