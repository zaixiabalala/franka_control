from abc import ABC


class DeviceBase(ABC):
    name: str
    
    def __init__(self, name:str='') -> None:
        self.name = name
        pass
