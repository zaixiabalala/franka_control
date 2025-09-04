# import zerorpc
# from polymetis import RobotInterface
# import scipy.spatial.transform as st
# import numpy as np
# import torch

# class FrankaInterface:
#     """Franka机器人接口，通过ZeroRPC与机器人通信"""
    
#     def __init__(self):
#         self.robot = RobotInterface('localhost')

#     def get_ee_pose(self):
#         """获取末端执行器姿态"""
#         data = self.robot.get_ee_pose()
#         pos = data[0].numpy()
#         quat_xyzw = data[1].numpy()
#         rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
#         return np.concatenate([pos, rot_vec]).tolist()
    
#     def get_joint_positions(self):
#         """获取关节位置"""
#         return self.robot.get_joint_positions().numpy().tolist()
    
#     def get_joint_velocities(self):
#         """获取关节速度"""
#         return self.robot.get_joint_velocities().numpy().tolist()
    
#     def move_to_joint_positions(self, positions, time_to_go):
#         """移动到指定关节位置"""
#         self.robot.move_to_joint_positions(
#             positions=torch.Tensor(positions),
#             time_to_go=time_to_go
#         )
    
#     def start_cartesian_impedance(self, Kx, Kxd):
#         """启动笛卡尔阻抗控制"""
#         self.robot.start_cartesian_impedance(
#             Kx=torch.Tensor(Kx),
#             Kxd=torch.Tensor(Kxd)
#         )

#     def update_desired_ee_pose(self, pose):
#         """更新期望末端执行器姿态"""
#         pose = np.asarray(pose)
#         self.robot.update_desired_ee_pose(
#             position=torch.Tensor(pose[:3]),
#             orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat())
#         )
    
#     def update_desired_joint_positions(self, positions):
#         """更新期望关节位置"""
#         self.robot.update_desired_joint_positions(
#             positions=torch.Tensor(positions)
#         )

#     def terminate_current_policy(self):
#         """终止当前策略"""
#         self.robot.terminate_current_policy()
