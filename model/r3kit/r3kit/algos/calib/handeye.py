from typing import Tuple, Optional
import numpy as np
import cv2

from r3kit.algos.calib.chessboard import ChessboardExtCalibor
from r3kit.algos.calib.config import *


class HandEyeCalibor(object):
    def __init__(self, marker_type:str=HANDEYE_MARKER_TYPE, 
                 ext_calib_params:dict={'patter_size': CHESSBOARD_PATTERN_SIZE, 'square_size': CHESSBOARD_SQUARE_SIZE}) -> None:
        if marker_type == 'chessboard':
            self.ext_calibor = ChessboardExtCalibor(**ext_calib_params)
        else:
            raise NotImplementedError
        
        self.b2g = []
    
    def add_image_pose(self, img:np.ndarray, pose:np.ndarray, vis:bool=True) -> bool:
        '''
        img: the image of chessboard in [0, 255] (h, w, 3) BGR
        pose: 4x4 transformation matrix from robot base to gripper
        '''
        ret = self.ext_calibor.add_image(img, vis)
        if ret:
            self.b2g.append(pose)
        return ret
    
    def run(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        '''
        b2w: 4x4 transformation matrix from robot base to world
        g2c: 4x4 transformation matrix from gripper to camera
        '''
        w2c = self.ext_calibor.run()
        if w2c is not None:
            R_b2w, t_b2w, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(w2c[:, :3, :3], w2c[:, :3, 3], 
                                                                        np.array(self.b2g)[:, :3, :3], np.array(self.b2g)[:, :3, 3])
            b2w = np.eye(4)
            b2w[:3, :3] = R_b2w
            b2w[:3, 3:] = t_b2w
            g2c = np.eye(4)
            g2c[:3, :3] = R_g2c
            g2c[:3, 3:] = t_g2c
            return (b2w, g2c)
        else:
            return (None, None)


if __name__ == '__main__':
    import os
    import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./calib_data')
    parser.add_argument('--marker_type', type=str, default='chessboard')
    args = parser.parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.data_dir, 'image_*.png')))
    
    ext_calib_params = {'pattern_size': (11, 8), 'square_size': 15}
    calibor = HandEyeCalibor(marker_type=args.marker_type, ext_calib_params=ext_calib_params)

    b2g = np.load(os.path.join(args.data_dir, 'b2g.npy'))
    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        calibor.add_image_pose(img, b2g[idx], vis=True)
    
    b2w, g2c = calibor.run()
    print(b2w)
    print(g2c)
    np.save(os.path.join(args.data_dir, 'b2w.npy'), b2w)
    np.save(os.path.join(args.data_dir, 'g2c.npy'), g2c)
    import pdb; pdb.set_trace()
