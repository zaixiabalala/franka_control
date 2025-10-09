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
from common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from common.precise_sleep import precise_wait
import torch
from common.pose_util import pose_to_mat, mat_to_pose
import zerorpc

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    SCHEDULE_WAYPOINT_POSE = 3

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
        use_joint_interp=True,
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
        self.use_joint_interp = use_joint_interp

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

        example['robot_receive_timestamp'] = time.monotonic()
        example['robot_timestamp'] = time.monotonic()
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
        
        if self.use_joint_interp:
            # 关节插值模式：期望7维关节角度
            assert pose.shape == (7,), f"关节插值模式下期望7维关节角度，实际为{pose.shape}"
            target_pose = pose
        else:
            # 姿态插值模式：期望6维姿态
            assert pose.shape == (6,), f"姿态插值模式下期望6维姿态，实际为{pose.shape}"
            # 将6维姿态数据扩展为7维数组（第7维设为0）
            target_pose = np.zeros(7, dtype=np.float64)
            target_pose[:6] = pose

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': target_pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        """调度路径点"""
        pose = np.array(pose)
        
        if self.use_joint_interp:
            # 关节插值模式：期望7维关节角度
            assert pose.shape == (7,), f"关节插值模式下期望7维关节角度，实际为{pose.shape}"
            target_pose = pose
        else:
            # 姿态插值模式：期望6维姿态
            assert pose.shape == (6,), f"姿态插值模式下期望6维姿态，实际为{pose.shape}"
            # 将6维姿态数据扩展为7维数组（第7维设为0）
            target_pose = np.zeros(7, dtype=np.float64)
            target_pose[:6] = pose

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': target_pose,
            'target_time': target_time
        }
        
        # 调试信息：检查消息发送到队列
        # print(f"[schedule_waypoint调试] 发送消息到队列: {message}")
        # print(f"[schedule_waypoint调试] 队列大小: {self.input_queue.qsize()}")
        
        self.input_queue.put(message)

    def schedule_waypoint_pose(self, pose, target_time):
        """调度路径点姿态"""
        pose = np.array(pose)
        assert pose.shape == (6,), f"姿态数据应为6维，实际为{pose.shape}"
        
        # 将6维姿态数据扩展为7维数组（第7维设为0）
        target_pose = np.zeros(7, dtype=np.float64)
        target_pose[:6] = pose
        
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT_POSE.value,
            'target_pose': target_pose,
            'target_time': target_time
        }
        
        # 详细的命令发送debug信息
        print(f"[Controller调试] 接收姿态命令:")
        print(f"  - 原始6维姿态: {pose}")
        print(f"  - 扩展7维姿态: {target_pose}")
        print(f"  - 目标时间: {target_time}")
        print(f"  - 当前队列大小: {self.input_queue.qsize()}")
        
        try:
            self.input_queue.put(message)
            print(f"  - 命令发送成功")
        except Exception as e:
            print(f"  - 命令发送失败: {e}")
            raise

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
        print("[Controller启动] 正在连接 ZeroRPC 服务...")
        self.robot = FrankaInterface(self.robot_ip, self.robot_port)
        print("[Controller启动] ZeroRPC 客户端创建完成")

         # 首帧对齐
        print("对齐到首帧...")
        print("[Controller启动] 获取关节位置...")
        cur_rad = self.robot.get_joint_positions()
        print("[Controller启动] 关节位置获取完成")
        target_joints_rad = [-2.1321869348527853, -0.5936860169121746, 1.1673866503505355, -3.027992172362134, 1.4189840432563865, 2.127658623566947, -1.6444724013831855]
        #target_joints_rad = np.radians(target_joints)
        
        print(f"当前关节位置: {np.degrees(cur_rad)}")
        print(f"目标关节位置: {np.degrees(target_joints_rad)}")

        self.joints_init = np.array(target_joints_rad)

        # 启用软实时
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        try:

            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")
            
            #self.robot.go_home()

            #初始化姿态 - 只在关节插值模式下执行
            if self.joints_init is not None and self.use_joint_interp:
                print(f"[Controller初始化] 移动机器人到初始关节位置...")
                try:
                    self.robot.move_to_joint_positions(
                        positions=np.asarray(self.joints_init),
                        time_to_go=self.joints_init_duration
                    )
                    print(f"[Controller初始化] 初始关节位置设置完成")
                except Exception as e:
                    print(f"[Controller初始化] 初始关节位置设置失败: {e}")
                    # 不要因为初始化失败就崩溃，继续运行
                    print(f"[Controller初始化] 警告：初始关节位置设置失败，但继续运行")
            elif self.joints_init is not None and not self.use_joint_interp:
                print(f"[Controller初始化] 姿态插值模式，跳过关节位置初始化")

            # 主循环
            dt = 1. / self.frequency
            print("[Controller初始化] 首次获取机器人状态...")
            curr_joints = self.robot.get_joint_positions()
            curr_pose = self.robot.get_ee_pose()
            print("[Controller初始化] 首次状态获取完成")

            # 使用单调时间确保控制循环永不倒退
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            # 创建关节轨迹插值器
            print(f"[Controller初始化] 检查use_joint_interp参数: {self.use_joint_interp}")
            if self.use_joint_interp:
                print(f"[Controller初始化] 进入关节插值模式")
                joint_interp = JointTrajectoryInterpolator(
                    times=np.array([curr_t]),
                    joints=curr_joints.reshape(1, -1)
                )
                # 启动franka关节位置控制策略
                # 注意：这里可能需要根据实际机器人接口调整
                self.robot.start_joint_impedance()
            else:
                print(f"[Controller初始化] 姿态插值模式:")
                print(f"  - 当前姿态: {curr_pose}")
                print(f"  - 姿态形状: {curr_pose.shape}")
                print(f"  - 当前时间: {curr_t}")
                
                pose_interp = PoseTrajectoryInterpolator(
                    times=np.array([curr_t]),
                    poses=curr_pose.reshape(1, -1)
                )
                print(f"  - 姿态插值器创建完成")
                print(f"  - 插值器时间: {pose_interp.times}")
                print(f"  - 插值器姿态: {pose_interp.poses}")
                
                # 先启动笛卡尔阻抗控制，再设置初始期望姿态
                print(f"  - 启动笛卡尔阻抗控制...")
                print(f"  - Kx: {self.Kx}")
                print(f"  - Kxd: {self.Kxd}")
                try:
                    print("  - 准备启动笛卡尔阻抗控制...")
                    self.robot.start_cartesian_impedance(
                        Kx=self.Kx,
                        Kxd=self.Kxd
                    )
                    print(f"  - 笛卡尔阻抗控制启动成功")
                except Exception as e:
                    print(f"  - 笛卡尔阻抗控制启动失败: {e}")
                    raise
                
                # 等待一小段时间让控制器完全启动
                time.sleep(0.1)
                
                print(f"  - 设置初始期望姿态...")
                try:
                    print("  - 设置初始期望姿态中...")
                    self.robot.update_desired_ee_pose(curr_pose)
                    print(f"  - 初始姿态设置成功")
                except Exception as e:
                    print(f"  - 初始姿态设置失败: {e}")
                    # 不要因为设置初始姿态失败就崩溃，继续运行
                    print(f"  - 警告：初始姿态设置失败，但继续运行")

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True

            while keep_running:
                # 向机器人发送命令
                t_now = time.monotonic()
                # 使用关节插值器获取目标关节位置
                if iter_idx % 100 == 0:
                    print(f"[Controller主循环] 循环 {iter_idx}, use_joint_interp: {self.use_joint_interp}")
                if self.use_joint_interp:
                    target_joints = joint_interp(t_now)
                    # 向机器人发送关节位置命令
                    if iter_idx % 10 == 0:
                        print(f"[Controller机器人控制] 发送关节位置: {target_joints}")
                    self.robot.update_desired_joint_positions(target_joints)
                else:
                    target_pose = pose_interp(t_now)
                    # 向机器人发送末端执行器姿态命令
                    if iter_idx % 10 == 0:
                        print(f"[Controller机器人控制] 发送姿态命令: {target_pose}")
                        print(f"  - 当前时间: {t_now}")
                        print(f"  - 插值器时间范围: {pose_interp.times[0]:.3f} - {pose_interp.times[-1]:.3f}")
                    try:
                        self.robot.update_desired_ee_pose(target_pose)
                        if iter_idx % 10 == 0:
                            print(f"  - 姿态命令发送成功")
                    except Exception as e:
                        print(f"  - 姿态命令发送失败: {e}")
                        raise

                # 更新机器人状态
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(self.robot, func_name)()

                    
                t_recv = time.monotonic()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # 从队列获取命令
                try:
                    # 每个周期最多处理1个命令以保持频率
                    queue_size_before = self.input_queue.qsize()
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                    queue_size_after = self.input_queue.qsize()
                    
                    # 详细的队列处理debug信息
                    print(f"[Controller主循环] 循环 {iter_idx}:")
                    print(f"  - 处理前队列大小: {queue_size_before}")
                    print(f"  - 处理后队列大小: {queue_size_after}")
                    print(f"  - 处理命令数量: {n_cmd}")
                    
                    if n_cmd > 0:
                        print(f"  - 命令类型: {commands['cmd']}")
                        print(f"  - 目标姿态: {commands['target_pose']}")
                        print(f"  - 目标时间: {commands['target_time']}")
                except Empty:
                    n_cmd = 0
                    if iter_idx % 10 == 0:  # 每10次循环打印一次
                        print(f"[Controller主循环] 循环 {iter_idx}: 队列为空，没有命令")

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
                        # 根据use_joint_interp参数选择控制方式
                        target_data = command['target_pose']
                        duration = float(command['duration'])
                        target_time = t_now + duration
                        
                        if self.use_joint_interp:
                            # 关节位置控制 - 使用插值器驱动到目标关节位置
                            target_joints = target_data  # 7维关节角度
                            joint_interp = joint_interp.drive_to_waypoint(
                                joints=target_joints,
                                time=target_time,
                                curr_time=t_now,
                                max_joint_speed=2.0  # 最大关节速度 2 rad/s
                            )
                            if self.verbose:
                                print("[FrankaJointController] New joint target:{} duration:{}s".format(
                                    target_joints, duration))
                        else:
                            # 姿态控制 - 使用插值器驱动到目标姿态
                            target_pose = target_data[:6]  # 提取前6维作为姿态数据
                            pose_interp = pose_interp.drive_to_waypoint(
                                pose=target_pose,
                                time=target_time,
                                curr_time=t_now,
                                max_linear_vel=0.25,  # 最大线速度 0.25 m/s
                                max_angular_vel=0.6   # 最大角速度 0.6 rad/s
                            )
                            if self.verbose:
                                print("[FrankaPoseController] New pose target:{} duration:{}s".format(
                                    target_pose, duration))
                        
                        last_waypoint_time = target_time
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        # 根据use_joint_interp参数选择控制方式
                        target_data = command['target_pose']
                        target_time = float(command['target_time'])
                        curr_time = t_now + dt
                        
                        # 调试信息
                        if iter_idx % 50 == 0:  # 每50次循环打印一次
                            print(f"[时间调试] 目标时间: {target_time:.3f}")
                            print(f"[时间调试] 当前时间: {curr_time:.3f}")
                            print(f"[时间调试] 时间差: {target_time - curr_time:.3f}")
                        
                        if self.use_joint_interp:
                            # 关节位置控制 - 调度目标关节位置
                            target_joints = target_data  # 7维关节角度
                            joint_interp = joint_interp.schedule_waypoint(
                                joints=target_joints,
                                time=target_time,
                                max_joint_speed=2.0,  # 最大关节速度 2 rad/s
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                        else:
                            # 姿态控制 - 调度目标姿态
                            target_pose = target_data[:6]  # 提取前6维作为姿态数据
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=target_pose,
                                time=target_time,
                                max_pos_speed=0.25,  # 最大线速度 0.25 m/s
                                max_rot_speed=0.6,   # 最大角速度 0.6 rad/s
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                        
                        last_waypoint_time = target_time
                    elif cmd == Command.SCHEDULE_WAYPOINT_POSE.value:
                        # 这个命令只在姿态插值模式下有效
                        print(f"[Controller命令处理] 处理SCHEDULE_WAYPOINT_POSE命令:")
                        print(f"  - use_joint_interp: {self.use_joint_interp}")
                        
                        if not self.use_joint_interp:
                            target_pose_7d = command['target_pose']
                            # 提取前6维作为姿态数据
                            target_pose = target_pose_7d[:6]
                            target_time = float(command['target_time'])
                            curr_time = t_now + dt
                            
                            print(f"  - 7维输入: {target_pose_7d}")
                            print(f"  - 6维姿态: {target_pose}")
                            print(f"  - 目标时间: {target_time}")
                            print(f"  - 当前时间: {curr_time}")
                            print(f"  - 时间差: {target_time - curr_time}")
                            
                            # 获取当前插值器状态
                            current_pose = pose_interp(t_now)
                            print(f"  - 当前插值姿态: {current_pose}")
                            
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=target_pose,
                                time=target_time,
                                max_pos_speed=0.25,  # 最大线速度 0.25 m/s
                                max_rot_speed=0.6,   # 最大角速度 0.6 rad/s
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                            last_waypoint_time = target_time
                            print(f"  - 姿态插值器更新完成")
                        else:
                            print("[警告] SCHEDULE_WAYPOINT_POSE命令在关节插值模式下被忽略")
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