from typing import Dict
import numpy as np

from r3kit.algos.lstsq import LinearSolver


class LinearOffsetCalibor(LinearSolver):
    def __init__(self) -> None:
        super().__init__()
    
    def add_data(self, pose:np.ndarray) -> None:
        '''
        pose: 4x4 transformation matrix from end-effector to base
        '''
        if not hasattr(self, 'R0'):
            self.R0 = pose[:3, :3]
            self.b0 = pose[:3, 3]
        else:
            A = self.R0 - pose[:3, :3]
            b = pose[:3, 3] - self.b0
            self.add_A(A)
            self.add_b(b)
    
    def run(self) -> Dict[str, np.ndarray]:
        # offset: tcp translation under end-effector frame
        X = super().run()
        return {
            'offset': X
        }


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_path', type=str, default='pose.npy')
    args = parser.parse_args()

    pose = np.load(args.pose_path)

    calibor = LinearOffsetCalibor()
    for p in pose:
        calibor.add_data(p)
    result = calibor.run()
    print(result)
