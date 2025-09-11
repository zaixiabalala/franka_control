from typing import Union
import numpy as np


def meanstd_normalize(x:Union[float, np.ndarray], 
                      mean:Union[float, np.ndarray], 
                      std:Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # actually also support torch.Tensor
    return (x - mean) / std

def meanstd_denormalize(x:Union[float, np.ndarray], 
                        mean:Union[float, np.ndarray], 
                        std:Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # actually also support torch.Tensor
    return x * std + mean


def minmax_normalize(x:Union[float, np.ndarray], 
                     min_val:Union[float, np.ndarray], 
                     max_val:Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # actually also support torch.Tensor
    return (x - min_val) / (max_val - min_val) * 2 - 1

def minmax_denormalize(x:Union[float, np.ndarray], 
                       min_val:Union[float, np.ndarray], 
                       max_val:Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # actually also support torch.Tensor
    return (x + 1) / 2 * (max_val - min_val) + min_val
