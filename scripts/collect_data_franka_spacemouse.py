# %%
import sys
import os

# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# print(ROOT_DIR)
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
import cv2
from datetime import datetime
from spacemouse.spacemouse_shared_memory import Spacemouse
from real_world.wsg_controller import WSGController
from common.precise_sleep import precise_wait
from spacemouse.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from real_world.franka_interpolation_controller import FrankaInterpolationController
from real_world.camera_system import CameraSystem

# %%
@click.command()
@click.option('-rh', '--robot_hostname', default='192.168.1.2')
@click.option('-gh', '--gripper_hostname', default='192.168.1.2')
@click.option('-gp', '--gripper_port', type=int, default=1000)
@click.option('-f', '--frequency', type=float, default=30)
@click.option('-gs', '--gripper_speed', type=float, default=200.0)
@click.option('-c', '--cameras', type=str, default=
{
    "cam_0": "327322062498",
    "cam_1": "038522062288"
})
@click.option('-o', '--output_dir', type=str, default='data')
def main(robot_hostname, gripper_hostname, gripper_port, frequency, gripper_speed, cameras, output_dir):
    max_pos_speed = 0.25
    max_rot_speed = 0.6
    max_gripper_width = 90.
    cube_diag = np.linalg.norm([1,1,1])
    tcp_offset = 0.13
    # tcp_offset = 0
    dt = 1/frequency
    command_latency = dt / 2    

    # ==== 记录目录结构 ====
    start_wall_ts = time.time()
    record_name = datetime.fromtimestamp(start_wall_ts).strftime('record_%Y%m%d_%H%M%S')
    record_root = os.path.join(output_dir, record_name)
    os.makedirs(record_root, exist_ok=True)

    def ensure_dir(p):
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

    # 先按相机名创建目录占位
    cam_system_tmp = CameraSystem(
        fps=frequency,
        cameras=cameras
    )
    cam_names = list(cam_system_tmp.camera_names)
    cam_system_tmp.close()

    
    angles_dir = os.path.join(record_root, 'angles') # 关节角加夹爪宽度
    tcp_pose_dir = os.path.join(record_root, 'tcp_pose')
    ts_dir = os.path.join(record_root, 'timestamps')
    ensure_dir(angles_dir)
    ensure_dir(tcp_pose_dir)
    ensure_dir(ts_dir)

    cam_dirs = {}
    for cam_name in cam_names:
        cam_root = os.path.join(record_root, cam_name)
        rgb_dir = os.path.join(cam_root, 'rgb')
        depth_dir = os.path.join(cam_root, 'depth')
        ensure_dir(rgb_dir)
        ensure_dir(depth_dir)
        cam_dirs[cam_name] = {'rgb': rgb_dir, 'depth': depth_dir}

    with SharedMemoryManager() as shm_manager:
        with WSGController(
            shm_manager=shm_manager,
            hostname=gripper_hostname,
            port=gripper_port,
            frequency=frequency,
            move_max_speed=400.0,
            verbose=False
        ) as gripper,\
        KeystrokeCounter() as key_counter, \
        CameraSystem(
            fps=frequency,
            cameras=cameras
        ) as camera_system, \
        FrankaInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_hostname,
            frequency=100,
            Kx_scale=5.0,
            Kxd_scale=2.0,
            verbose=False,
            use_joint_interp=False
        ) as controller, \
        Spacemouse(
            shm_manager=shm_manager
        ) as sm:
            print('Ready!')
            # to account for recever interfance latency, use target pose
            # to init buffer.
            state = controller.get_state()
            # target_pose = state['TargetTCPPose']
            target_pose = state['ActualTCPPose']
        
            gripper_target_pos = gripper.get_state()['gripper_position']
            t_start = time.monotonic()
            gripper.restart_put(t_start)
            
            iter_idx = 0
            stop = False
            while not stop:
                state = controller.get_state()
                # print(target_pose - state['ActualTCPPose'])
                s = time.monotonic()
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    # if key_stroke != None:
                    #     print(key_stroke)
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                precise_wait(t_sample)
                sm_state = sm.get_motion_state_transformed()
                # print(sm_state)
                dpos = sm_state[:3] * (max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)

                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (drot * \
                        st.Rotation.from_rotvec(target_pose[3:])
                ).as_rotvec()

                dpos = 0
                if sm.is_button_pressed(0):
                    # close gripper
                    dpos = -gripper_speed / frequency
                if sm.is_button_pressed(1):
                    dpos = gripper_speed / frequency
                gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)

                controller.schedule_waypoint_pose(target_pose, 
                    t_command_target)
                gripper.schedule_waypoint(gripper_target_pos, 
                    t_command_target)

                # ====== 数据保存（30Hz）======
                frame_id = f"{iter_idx:06d}"

                # 时间戳：保存 monotonic 与 wall time
                ts_array = np.array([time.monotonic(), time.time()], dtype=np.float64)
                np.save(os.path.join(ts_dir, frame_id + '.npy'), ts_array)

                robot_state = controller.get_state()
                gripper_state = gripper.get_state()
                # 关节角与夹爪宽度
                joints = robot_state.get('ActualQ', None)
                if joints is None:
                    joints = np.zeros(7, dtype=np.float64)
                grip_width = np.array([gripper_state['gripper_position']], dtype=np.float64)
                angles_save = np.concatenate([np.array(joints, dtype=np.float64).reshape(-1), grip_width], axis=0)
                np.save(os.path.join(angles_dir, frame_id + '.npy'), angles_save)

                # 末端姿态（6维：xyz + rotvec）
                tcp_pose = robot_state.get('ActualTCPPose', None)
                if tcp_pose is None:
                    tcp_pose = np.zeros(6, dtype=np.float64)
                np.save(os.path.join(tcp_pose_dir, frame_id + '.npy'), np.array(tcp_pose, dtype=np.float64))

                # 各相机 RGB/DEPTH
                for cam_name in cam_names:
                    rgb_img, depth_img = camera_system.get_image_and_depth(cam_name)
                    # RGB 保存为 PNG（需转回 BGR）
                    if rgb_img is not None:
                        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(cam_dirs[cam_name]['rgb'], frame_id + '.png'), bgr)
                    # 深度保存为 PNG（优先 uint16）
                    if depth_img is not None:
                        depth_to_save = depth_img
                        if depth_to_save.dtype != np.uint16:
                            # 归一化/米→毫米到 uint16（保守处理）
                            if depth_to_save.max() <= 10.0:
                                depth_to_save = (depth_to_save * 1000.0).astype(np.uint16)
                            else:
                                depth_norm = depth_to_save.astype(np.float32)
                                depth_norm = (depth_norm / (depth_norm.max() + 1e-6)) * 65535.0
                                depth_to_save = depth_norm.astype(np.uint16)
                        cv2.imwrite(os.path.join(cam_dirs[cam_name]['depth'], frame_id + '.png'), depth_to_save)

                iter_idx += 1

                precise_wait(t_cycle_end)

    controller.terminate_current_policy()
# %%
if __name__ == '__main__':
    main()