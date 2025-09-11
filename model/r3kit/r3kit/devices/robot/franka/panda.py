import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from frankx import Robot, JointMotion, Affine, LinearMotion, ImpedanceMotion

from r3kit.devices.robot.base import RobotBase
from r3kit.devices.robot.franka.config import *


class Panda(RobotBase):
    DOF:int = 7

    def __init__(self, ip:str=PANDA_IP, name:str='Panda') -> None:
        super().__init__(name)
        
        self.robot = Robot(ip)
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()
        self.robot.set_dynamic_rel(PANDA_DYNAMIC_REL)

        self.t2f = np.array(self.robot.read_once().F_T_EE).reshape(4, 4).T

        self.in_impedance_control = False
    
    def homing(self) -> None:
        self.joint_move(PANDA_HOME_JOINTS, relative=False)
    
    def joint_read(self) -> np.ndarray:
        '''
        joints: 7 DoF joint angles in radian
        '''
        if not self.in_impedance_control:
            state = self.robot.read_once()
            joints = np.array(state.q)
        else:
            raise NotImplementedError
        return joints

    def joint_move(self, joints:np.ndarray, relative:bool=False) -> None:
        '''
        joints: 7 DoF joint angles in radian
        relative: if True, move relative to current joints; if False, move absolutely
        '''
        if not relative:
            action = JointMotion(joints.tolist())
        else:
            current_joints = self.joint_read().tolist()
            action = JointMotion([current_joints[i] + joints[i] for i in range(self.DOF)])
        
        if not self.in_impedance_control:
            self.robot.move(action)
        else:
            raise NotImplementedError

    def flange_read(self) -> np.ndarray:
        '''
        f2b: 4x4 transformation matrix from flange to robot base
        '''
        if not self.in_impedance_control:
            robot_state = self.robot.read_once()
        else:
            # NOTE: not supported in frankx, need to rebuild
            robot_state = self.impedance_motion.get_robotstate()
        t2b = np.array(robot_state.O_T_EE).reshape(4, 4).T
        f2b = t2b @ np.linalg.inv(self.t2f)
        return f2b
    
    def flange_move(self, f2b:np.ndarray, relative:bool=False) -> None:
        '''
        f2b: 4x4 transformation matrix from flange to robot base
        relative: if True, move relative to current pose; if False, move absolutely in robot base frame
        '''
        if not relative:
            t2b = f2b @ self.t2f
            tr = t2b[:3, 3]
            rot = Rot.from_matrix(t2b[:3, :3]).as_euler('ZYX')
            if not self.in_impedance_control:
                action = LinearMotion(Affine(tr[0], tr[1], tr[2], rot[0], rot[1], rot[2]))
                self.robot.move(action)
            else:
                self.impedance_motion.target = Affine(tr[0], tr[1], tr[2], rot[0], rot[1], rot[2])
        else:
            raise NotImplementedError
    
    def start_impedance_control(self, tr_stiffness:float=PANDA_TR_STIFFNESS, rot_stiffness:float=PANDA_ROT_STIFFNESS) -> None:
        self.impedance_motion = ImpedanceMotion(tr_stiffness, rot_stiffness)
        self.robot_thread = self.robot.move_async(self.impedance_motion)
        time.sleep(PANDA_IMPEDANCE_WAIT_TIME)
        self.in_impedance_control = True
    
    def stop_impedance_control(self) -> None:
        self.impedance_motion.finish()
        self.robot_thread.join()
        self.impedance_motion = None
        self.robot_thread = None
        self.in_impedance_control = False


if __name__ == "__main__":
    robot = Panda(ip='172.16.0.2', name='panda')

    robot.homing()
    print("homing")
    joint = robot.joint_read()
    print("current joint:", joint)
    pose = robot.flange_read()
    print("current pose:", pose)
    target_pose = pose.copy()
    target_pose[:3, 3] += np.array([0.05, 0.05, 0.05])
    robot.flange_move(target_pose, relative=False)
    print("move")
    joint = robot.joint_read()
    print("current joint:", joint)
    pose = robot.flange_read()
    print("current pose:", pose)
    robot.start_impedance_control(tr_stiffness=1000.0, rot_stiffness=20.0)
    print("start impedance control")
    for i in range(10):
        current_pose = robot.flange_read()
        print(i, current_pose)
        target_pose = current_pose.copy()
        target_pose[:3, 3] += np.array([0., 0.02, 0.])
        robot.flange_move(target_pose, relative=False)
        time.sleep(0.3)
    pose = robot.flange_read()
    print("current pose:", pose)
    robot.stop_impedance_control()
    robot.homing()
    print("homing")
