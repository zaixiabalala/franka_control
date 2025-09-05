import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from common.joint_trajectory_interpolator import JointTrajectoryInterpolator
from common.precise_sleep import precise_wait
import torch
from common.pose_util import pose_to_mat, mat_to_pose
import zerorpc

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

# Franka末端执行器变换矩阵
tx_flangerot90_tip = np.identity(4)
tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3,:3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3,:3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)

class FrankaInterface:
    """Franka机器人接口，通过ZeroRPC与机器人通信"""
    
    def __init__(self, ip='172.16.0.3', port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        """获取末端执行器姿态（tip坐标系）"""
        flange_pose = np.array(self.server.get_ee_pose())
        tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
        return tip_pose
    
    def get_joint_positions(self):
        """获取关节位置"""
        return np.array(self.server.get_joint_positions())
    
    def get_joint_velocities(self):
        """获取关节速度"""
        return np.array(self.server.get_joint_velocities())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        """移动到指定关节位置"""
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        """启动笛卡尔阻抗控制"""
        self.server.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )

    def start_joint_impedance(self):
        """启动关节阻抗控制"""
        self.server.start_joint_impedance()
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        """更新期望末端执行器姿态"""
        self.server.update_desired_ee_pose(pose.tolist())
    
    def update_desired_joint_positions(self, joint_positions: np.ndarray):
        """更新期望关节位置"""
        self.server.update_desired_joint_positions(joint_positions.tolist())

    def terminate_current_policy(self):
        """终止当前策略"""
        self.server.terminate_current_policy()

    def forward_kinematics(self, joint_positions):
        """正向运动学计算"""
        return self.server.forward_kinematics(joint_positions.tolist()) 

    def close(self):
        """关闭连接"""
        self.server.close()


class FrankaInterpolationController(mp.Process):
    """
    为了确保以可预测的延迟向机器人发送命令，
    此控制器需要其单独的进程（由于Python GIL）
    """
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        robot_ip,
        robot_port=4242,
        frequency=1000,
        Kx_scale=1.0,
        Kxd_scale=1.0,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=3,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0
        ):
        """
        robot_ip: 中间层控制器(NUC)的IP地址
        frequency: Franka的频率，通常为1000Hz
        Kx_scale: 位置增益的缩放因子
        Kxd: 速度增益的缩放因子
        soft_real_time: 启用轮询调度和实时优先级
            需要事先运行scripts/rtprio_setup.sh
        """

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # 构建输入队列
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((7,), dtype=np.float64),  # 改为7维关节角度
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # 构建环形缓冲区
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd','get_joint_velocities'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
            
    # ========= 启动方法 ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= 上下文管理器 ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= 命令方法 ============
    def servoL(self, pose, duration=0.1):
        """
        伺服到指定姿态
        duration: 到达姿态的期望时间
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        """调度路径点"""
        pose = np.array(pose)
        assert pose.shape == (7,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        
        # 调试信息：检查消息发送到队列
        print(f"[schedule_waypoint调试] 发送消息到队列: {message}")
        print(f"[schedule_waypoint调试] 队列大小: {self.input_queue.qsize()}")
        
        self.input_queue.put(message)
    
    # ========= 接收API =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    

    # ========= 进程中的主循环 ============
    def run(self):
        self.robot = FrankaInterface(self.robot_ip, self.robot_port)

         # 首帧对齐
        print("对齐到首帧...")
        cur_rad = self.robot.get_joint_positions()
        target_joints = [68.82101, 40.603436, -128.58241, -121.37202, 131.0479, 115.84139, -60.55921]
        target_joints_rad = np.radians(target_joints)
        
        print(f"当前关节位置: {np.degrees(cur_rad)}")
        print(f"目标关节位置: {np.degrees(target_joints_rad)}")

        self.joints_init = target_joints_rad

        # 启用软实时
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        try:

            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")
            
            #self.robot.go_home()

            #初始化姿态
            if self.joints_init is not None:
                print("xxxxxxxxxxxxxx...")
                self.robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )

            # 主循环
            dt = 1. / self.frequency
            print(1)
            curr_joints = self.robot.get_joint_positions()
            print(2)

            # 使用单调时间确保控制循环永不倒退
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            # 创建关节轨迹插值器
            joint_interp = JointTrajectoryInterpolator(
                times=np.array([curr_t]),
                joints=curr_joints.reshape(1, -1)
            )

            # 启动franka关节位置控制策略
            # 注意：这里可能需要根据实际机器人接口调整
            self.robot.start_joint_impedance()

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True

            while keep_running:
                # 向机器人发送命令
                t_now = time.monotonic()
                # 使用关节插值器获取目标关节位置
                target_joints = joint_interp(t_now)
                # print("xxxxxxxx")
                # print(f"[控制器调试] 目标关节: {target_joints}")

                # 向机器人发送关节位置命令
                self.robot.update_desired_joint_positions(target_joints)

                # 更新机器人状态
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(self.robot, func_name)()

                    
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # 从队列获取命令
                try:
                    # 每个周期最多处理1个命令以保持频率
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                    
                    # # 调试信息：检查队列状态
                    # if iter_idx % 100 == 0:  # 每100次循环打印一次
                    #     print(f"[控制器调试] 队列中有 {n_cmd} 个命令")
                    #     if n_cmd > 0:
                    #         print(f"[控制器调试] 命令类型: {commands['cmd']}")
                    #         print(f"[控制器调试] 目标关节: {commands['target_pose']}")
                except Empty:
                    n_cmd = 0
                    if iter_idx % 100 == 0:  # 每100次循环打印一次
                        print(f"[控制器调试] 队列为空，没有命令")

                # 执行命令
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # 立即停止，忽略后续命令
                        break
                    elif cmd == Command.SERVOL.value:
                        # 关节位置控制 - 使用插值器驱动到目标关节位置
                        target_joints = command['target_pose']  # 现在表示关节角度
                        duration = float(command['duration'])
                        target_time = t_now + duration
                        joint_interp = joint_interp.drive_to_waypoint(
                            joints=target_joints,
                            time=target_time,
                            curr_time=t_now,
                            max_joint_speed=2.0  # 最大关节速度 2 rad/s
                        )
                        last_waypoint_time = target_time
                        if self.verbose:
                            print("[FrankaJointController] New joint target:{} duration:{}s".format(
                                target_joints, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        # 关节位置控制 - 调度目标关节位置
                        target_joints = command['target_pose']  # 现在表示关节角度
                        target_time = float(command['target_time'])
                        # 将全局时间转换为单调时间
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        joint_interp = joint_interp.schedule_waypoint(
                            joints=target_joints,
                            time=target_time,
                            max_joint_speed=2.0,  # 最大关节速度 2 rad/s
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # 调节频率
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # 第一次循环成功，准备接收命令
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[FrankaPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            # 强制清理
            # 终止
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            self.robot.terminate_current_policy()
            del self.robot
            self.ready_event.set()

            if self.verbose:
                print(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
