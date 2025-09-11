from typing import Tuple, List
import numpy as np


def find_nearest_idx(arr:np.ndarray, item) -> int:
    '''
    arr: must be sorted
    '''
    return np.abs(arr - item).argmin()


def seq_idx(num_samples:int, num_o:int=1, num_a:int=20, 
            pad_o:bool=True, pad_a:bool=True, pad_aa:bool=True, num_aa:int=10) -> Tuple[List[List[int]], List[List[int]], List[List[bool]], List[List[bool]]]:
    o_idxs, a_idxs = [], []
    pad_o_idts, pad_a_idts = [], []
    for current_idx in range(num_samples - 1):
        selected = True

        o_begin_idx = max(0, current_idx - num_o + 1)
        o_end_idx = min(num_samples, current_idx + 1)
        o_padding = num_o - (o_end_idx - o_begin_idx)
        if o_padding > 0:
            if pad_o:
                o_selected_idxs = [0] * o_padding + list(range(o_begin_idx, o_end_idx))
                o_pad_idts = [True] * o_padding + [False] * (o_end_idx - o_begin_idx)
            else:
                selected = False
        else:
            o_selected_idxs = list(range(o_begin_idx, o_end_idx))
            o_pad_idts = [False] * (o_end_idx - o_begin_idx)

        a_begin_idx = min(num_samples - 1, current_idx + 1)
        a_end_idx = min(num_samples, current_idx + 1 + num_a)
        a_padding = num_a - (a_end_idx - a_begin_idx)
        if a_padding > 0:
            if pad_a and pad_aa:
                if a_padding > num_a - num_aa:
                    selected = False
                else:
                    a_selected_idxs = list(range(a_begin_idx, a_end_idx)) + [num_samples - 1] * a_padding
                    a_pad_idts = [False] * (a_end_idx - a_begin_idx) + [True] * a_padding
            elif pad_a and not pad_aa:
                a_selected_idxs = list(range(a_begin_idx, a_end_idx)) + [num_samples - 1] * a_padding
                a_pad_idts = [False] * (a_end_idx - a_begin_idx) + [True] * a_padding
            else:
                selected = False
        else:
            a_selected_idxs = list(range(a_begin_idx, a_end_idx))
            a_pad_idts = [False] * (a_end_idx - a_begin_idx)
        
        if selected:
            o_idxs.append(o_selected_idxs)
            a_idxs.append(a_selected_idxs)
            pad_o_idts.append(o_pad_idts)
            pad_a_idts.append(a_pad_idts)
    return (o_idxs, a_idxs, pad_o_idts, pad_a_idts)
