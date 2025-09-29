#!/usr/bin/env python3
"""
启动Franka接口服务器
这个脚本启动一个ZeroRPC服务器，提供Franka机器人的控制接口
"""

import zerorpc
from polymetis import RobotInterface, GripperInterface
import scipy.spatial.transform as st
import numpy as np
import torch

class FrankaInterface:
    """Franka机器人接口，通过ZeroRPC与机器人通信"""
    
    def __init__(self):
        self.robot = RobotInterface(ip_address='localhost')
        self.robot.go_home()
        try:
            self.gripper = GripperInterface(ip_address='localhost')
            print("Gripper接口初始化成功")
        except Exception as e:
            print(f"Gripper接口初始化失败: {e}")
            self.gripper = None

    def get_ee_pose(self):
        """获取末端执行器姿态"""
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()
    
    def get_joint_positions(self):
        """获取关节位置"""
        return self.robot.get_joint_positions().numpy().tolist()
    
    def get_joint_velocities(self):
        """获取关节速度"""
        return self.robot.get_joint_velocities().numpy().tolist()
    
    def move_to_joint_positions(self, positions, time_to_go):
        """移动到指定关节位置"""
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(positions),
            time_to_go=time_to_go
        )
    
    def start_cartesian_impedance(self, Kx, Kxd):
        """启动笛卡尔阻抗控制"""
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx),
            Kxd=torch.Tensor(Kxd)
        )

    def start_joint_impedance(self):
        """启动关节阻抗控制"""
        self.robot.start_joint_impedance()

    def update_desired_ee_pose(self, pose):
        """更新期望末端执行器姿态"""
        pose = np.asarray(pose)
        self.robot.update_desired_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat())
        )
    
    def update_desired_joint_positions(self, positions):
        """更新期望关节位置"""
        self.robot.update_desired_joint_positions(
            positions=torch.Tensor(positions)
        )

    def terminate_current_policy(self):
        """终止当前策略"""
        self.robot.terminate_current_policy()
    
    def forward_kinematics(self, joint_positions):
        """正向运动学计算"""
        try:
            # 使用机器人模型进行正向运动学计算
            target_joints_tensor = torch.Tensor(joint_positions)
            pose, quat = self.robot.robot_model.forward_kinematics(target_joints_tensor)
            
            # 返回位置和四元数
            pose_np = pose.numpy()
            quat_np = quat.numpy()
            
            return pose_np.tolist() + quat_np.tolist()
        except Exception as e:
            print(f"正向运动学计算失败: {e}")
            # 返回默认值
            return [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]
    
    def get_gripper_width(self):
        """获取gripper当前开合宽度"""
        if self.gripper is None:
            return 0.04  # 默认宽度
        
        try:
            state = self.gripper.get_state()
            return float(state.width)
        except Exception as e:
            print(f"获取gripper宽度失败: {e}")
            return 0.04
    
    def move_gripper_to_width(self, width, speed=None, force=None):
        """移动gripper到指定宽度"""
        if self.gripper is None:
            print("Gripper未初始化，跳过控制")
            return False
        
        try:
            # 限制宽度范围
            width = float(np.clip(width, 0.0, 0.08))
            
            # 设置默认参数
            if speed is None:
                speed = 0.1
            if force is None:
                force = 10.0
            
            self.gripper.goto(width=width, speed=speed, force=force)
            return True
        except Exception as e:
            print(f"移动gripper失败: {e}")
            return False

def main():
    """启动ZeroRPC服务器"""
    print("启动Franka接口服务器...")
    s = zerorpc.Server(FrankaInterface())
    s.bind("tcp://0.0.0.0:4242")
    print("服务器已启动，监听端口4242")
    s.run()

if __name__ == '__main__':
    main()