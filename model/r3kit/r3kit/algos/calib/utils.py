import numpy as np
import cv2


def rodrigues_rvec2mat(rvec:np.ndarray, tvec:np.ndarray) -> np.ndarray:
    '''
    rvec: rotation vector
    tvec: translation vector
    mat: 4x4 transformation matrix
    '''
    r, _ = cv2.Rodrigues(rvec)
    tvec.shape = (3,)
    mat = np.identity(4)
    mat[0:3, 3] = tvec
    mat[0:3, 0:3] = r
    return mat
