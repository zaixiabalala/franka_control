from typing import List, Dict, Union, Optional
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from r3kit.devices.robot.base import RobotBase
from r3kit.devices.robot.flexiv.config import *
from r3kit.utils.transformation import xyzquat2mat, mat2xyzquat, delta_xyz, delta_quat

try:
    import sys
    sys.path.insert(0, FLEXIV_RDK_PATH)
    import flexivrdk    # v1.5
except ImportError:
    print("Robot Flexiv Rizon needs `flexivrdk`")
    sys.exit(1)


class Rizon(RobotBase):
    DOF:int = 7

    def __init__(self, id:str=RIZON_ID, gripper:bool=True, name:str='Rizon') -> None:
        super().__init__(name)

        self.robot = flexivrdk.Robot(id)
        if self.robot.fault():
            if not robot.ClearFault():
                raise RuntimeError(f"Failed to clear fault for robot {name}")
        self.robot.Enable()
        seconds_waited = 0
        while not self.robot.operational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == RIZON_OPERATIONAL_WAIT_TIME:
                raise RuntimeError(f"Failed to enable robot {name}")
        self.motion_mode('primitive')
        self.block(True)
        info = self.robot.info()
        self._joint_limits = (np.array(info.q_min), np.array(info.q_max))
        if gripper:
            self.gripper = flexivrdk.Gripper(self.robot)
            gripper.Init()
            info = self.gripper.states()
            self._gripper_limits = (0, float(info.max_width))
        else:
            self.gripper = None
    
    def __del__(self) -> None:
        if hasattr(self, 'robot') and self.robot is not None:
            self.robot.Stop()
        if hasattr(self, 'gripper') and self.gripper is not None:
            self.gripper.Stop()
    
    def motion_mode(self, mode:str) -> None:
        if mode == 'primitive':
            self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        elif mode == 'joint':
            self.robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_IMPEDANCE)
        elif mode == 'tcp':
            self.robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        else:
            raise ValueError(f"Invalid motion mode: {mode}")
        self.mode = mode
    
    def block(self, blocking:bool) -> None:
        self.blocking = blocking
    
    def homing(self) -> None:
        '''
        Move robot to home position
        '''
        if self.mode == 'primitive':
            self.robot.ExecutePrimitive("Home", dict())
            if self.blocking:
                while not self.robot.primitive_states()["reachedTarget"]:
                    time.sleep(RIZON_BLOCK_WAIT_TIME)
            else:
                pass
        elif self.mode == 'joint':
            self.joint_move(RIZON_HOME_JOINTS)
        elif self.mode == 'tcp':
            self.tcp_move(RIZON_HOME_POSE)
        else:
            raise ValueError(f"Invalid motion mode: {self.mode}")
    
    def joint_read(self) -> np.ndarray:
        '''
        joints: 7 DoF joint angles in radian
        '''
        joints = np.array(self.robot.states().q)
        return joints
    
    def joint_move(self, joints:np.ndarray, velocities:Optional[np.ndarray]=None, accelerations:Optional[np.ndarray]=None) -> None:
        '''
        joints: 7 DoF joint angles in radian
        velocities: 7 DoF joint velocities in radian/s
        accelerations: 7 DoF joint accelerations in radian/s^2
        '''
        if velocities is None:
            velocities = np.array([0.0]*self.DOF)
        if accelerations is None:
            accelerations = np.array([0.0]*self.DOF)
        joints = np.clip(joints, self._joint_limits[0], self._joint_limits[1])
        max_vel = np.array([RIZON_JOINT_MAX_VEL]*self.DOF)
        max_acc = np.array([RIZON_JOINT_MAX_ACC]*self.DOF)
        
        if self.mode == 'primitive':
            self.robot.ExecutePrimitive("MoveJ", {"target": np.rad2deg(joints).tolist()})
            if self.blocking:
                while not robot.primitive_states()["reachedTarget"]:
                    time.sleep(RIZON_BLOCK_WAIT_TIME)
            else:
                pass
        elif self.mode == 'joint':
            self.robot.SendJointPosition(joints.tolist(), velocities.tolist(), accelerations.tolist(), max_vel.tolist(), max_acc.tolist())
            if self.blocking:
                error = float('inf')
                while error > RIZON_JOINT_EPSILON:
                    time.sleep(RIZON_BLOCK_WAIT_TIME)
                    error = np.abs(self.joint_read() - joints).max()
            else:
                pass
        elif self.mode == 'tcp':
            raise ValueError("Cannot move joints in tcp mode")
        else:
            raise ValueError(f"Invalid motion mode: {self.mode}")
    
    def gripper_read(self) -> float:
        '''
        width: gripper full width in meter
        '''
        if self.gripper is None:
            raise ValueError("Gripper is not initialized")
        width = float(self.gripper.states().width)
        return width
    
    def gripper_move(self, width:float, velocity:float=0.05) -> None:
        '''
        width: gripper full width in meter
        velocity: gripper velocity in m/s
        '''
        if self.gripper is None:
            raise ValueError("Gripper is not initialized")
        width = np.clip(width, self._gripper_limits[0], self._gripper_limits[1])
        self.gripper.Move(width, velocity)
        if self.blocking:
            error = float('inf')
            while error > RIZON_GRIPPER_EPSILON:
                time.sleep(RIZON_BLOCK_WAIT_TIME)
                error = np.abs(self.gripper_read() - width)
        else:
            pass
    
    def tcp_read(self) -> np.ndarray:
        '''
        pose: 4x4 transformation matrix of tcp relative to robot base
        '''
        vec = np.array(self.robot.states().tcp_pose)

        xyz = vec[:3]
        quat = vec[3:]                                          # (w, x, y, z)
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])   # (x, y, z, w)
        pose = xyzquat2mat(xyz, quat)
        return pose
    
    def tcp_move(self, pose:Optional[np.ndarray]=None, wrench:Optional[np.ndarray]=None) -> None:
        '''
        pose: 4x4 transformation matrix of tcp relative to robot base
        wrench: 6D force/torque in N/Nm in the force control reference frame
        '''
        if pose is None:
            pose = self.tcp_read()
            pure_force = True
        if wrench is None:
            wrench = np.zeros(6)
            pure_motion = True
        
        if self.mode == 'primitive':
            raise NotImplementedError("Primitive mode is not implemented for tcp control")
        elif self.mode == 'joint':
            raise ValueError("Cannot move tcp in joint mode")
        elif self.mode == 'tcp':
            vec = np.zeros(7)
            xyz, quat = mat2xyzquat(pose)
            vec[:3] = xyz
            vec[3:] = [quat[3], quat[0], quat[1], quat[2]]  # (w, x, y, z)
            self.robot.SendCartesianMotionForce(vec.tolist(), wrench.tolist(), 
                                                max_linear_vel=RIZON_TCP_MAX_VEL[0], max_linear_acc=RIZON_TCP_MAX_ACC[0], 
                                                max_angular_vel=RIZON_TCP_MAX_VEL[1], max_angular_acc=RIZON_TCP_MAX_ACC[1])
            if self.blocking:
                if pure_motion:
                    error_xyz, error_quat = float('inf'), float('inf')
                    while error_xyz > RIZON_TCP_POSE_EPSILON[0] or error_quat > RIZON_TCP_POSE_EPSILON[1]:
                        time.sleep(RIZON_BLOCK_WAIT_TIME)
                        current_pose = self.tcp_read()
                        target_pose = pose
                        current_xyz, current_quat = mat2xyzquat(current_pose)
                        target_xyz, target_quat = mat2xyzquat(target_pose)
                        error_xyz = delta_xyz(current_xyz, target_xyz)
                        error_quat = delta_quat(current_quat, target_quat)
                elif pure_force:
                    raise NotImplementedError("Blocking pure force control is not implemented")
                else:
                    raise NotImplementedError("Blocking hybrid motion and force control is not implemented")
            else:
                pass
        else:
            raise ValueError(f"Invalid motion mode: {self.mode}")
    
    def zero_ft(self) -> None:
        '''
        Zero FT sensor, must blocking
        '''
        if self.mode != 'primitive':
            original_mode = self.mode
            self.motion_mode('primitive')
        else:
            original_mode = None
        
        self.robot.ExecutePrimitive("ZeroFTSensor", dict())
        while robot.busy():
            time.sleep(RIZON_BLOCK_WAIT_TIME)
        
        if original_mode is not None:
            self.motion_mode(original_mode)
        else:
            pass
    
    def ft_read(self, tcp:bool=True, filtered:bool=True, raw:bool=False) -> np.ndarray:
        '''
        ft: 6D force/torque in N/Nm
        tcp: if True, return force/torque in tcp frame; if False, return in robot base frame
        filtered: if True, return filtered force/torque; if False, return unfiltered force/torque
        raw: if True, return raw force/torque reading; if False, return processed force/torque
        '''
        if raw:
            ft = np.array(self.robot.states().ft_sensor_raw)
        else:
            if tcp:
                if filtered:
                    ft = np.array(self.robot.states().ext_wrench_in_tcp)
                else:
                    ft = np.array(self.robot.states().ext_wrench_in_tcp_raw)
            else:
                if filtered:
                    ft = np.array(self.robot.states().ext_wrench_in_world)
                else:
                    ft = np.array(self.robot.states().ext_wrench_in_world_raw)
        return ft
    
    def set_force_control_config(self, frame_type:str, relative_transformation:np.ndarray=np.eye(4), enabled_axes:List[bool]=[False, False, False, False, False, False]) -> None:
        '''
        frame_type: 'tcp' (moving) or 'world' (fixed)
        relative_transformation: 4x4 transformation matrix of force control frame relative to selected frame
        '''
        if self.mode != 'tcp':
            original_mode = self.mode
            self.motion_mode('tcp')
        else:
            original_mode = None
        
        if frame_type == 'tcp':
            force_ctrl_frame = flexivrdk.CoordType.TCP
        elif frame_type == 'world':
            force_ctrl_frame = flexivrdk.CoordType.WORLD
        else:
            raise ValueError(f"Invalid force control frame type: {frame_type}")
        relative_vec = np.zeros(7)
        relative_vec[:3] = relative_transformation[:3, 3]
        relative_quat = Rot.from_matrix(relative_transformation[:3, :3]).as_quat()
        relative_vec[3:] = [relative_quat[3], relative_quat[0], relative_quat[1], relative_quat[2]] # (w, x, y, z)
        self.robot.SetForceControlFrame(force_ctrl_frame, relative_vec)
        self.robot.SetForceControlAxis(enabled_axes, max_linear_vel=[RIZON_TCP_MAX_VEL[0]] * 3)

        if original_mode is not None:
            self.motion_mode(original_mode)
        else:
            pass
    
    @staticmethod
    def raw2tare(raw_ft:np.ndarray, tare:Dict[str, Union[float, np.ndarray]], pose:np.ndarray) -> np.ndarray:
        '''
        raw_ft: raw force torque data
        pose: 3x3 rotation matrix from ft to base
        '''
        raw_f, raw_t = raw_ft[:3], raw_ft[3:]
        f = raw_f - tare['f0']
        f -= np.linalg.inv(pose) @ np.array([0., 0., -9.8 * tare['m']])
        t = raw_t - tare['t0']
        t -= np.linalg.inv(pose) @ np.cross(np.linalg.inv(pose) @ np.array(tare['c']), np.array([0., 0., -9.8 * tare['m']]))
        return np.concatenate([f, t])


if __name__ == "__main__":
    robot = Rizon(id='Rizon4s-12345', gripper=True, name='Rizon')

    robot.homing()
    print("homing")
    joint = robot.joint_read()
    print("current joint:", joint)
    pose = robot.tcp_read()
    print("current pose:", pose)
