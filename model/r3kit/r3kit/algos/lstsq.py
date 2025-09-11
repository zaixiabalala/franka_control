import numpy as np


class LinearSolver(object):
    '''
    A linear solver that solves the equation A@X = b by least square method.
    '''
    def __init__(self) -> None:
        self.A_list = []
        self.b_list = []

    def add_A(self, A:np.ndarray) -> None:
        self.A_list.append(A)
    
    def add_b(self, b:np.ndarray) -> None:
        self.b_list.append(b)
    
    def run(self) -> np.ndarray:
        A = np.concatenate(self.A_list, axis=0)
        B = np.concatenate(self.b_list, axis=0)
        X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        return X
