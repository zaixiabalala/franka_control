#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSG Gripper Controller - 基于UMI的WSGController实现
用于控制WSG夹爪
"""

import os
import time
import multiprocessing as mp
import enum
import numpy as np
from queue import Empty

from multiprocessing.managers import SharedMemoryManager
from shared_memory import SharedMemoryQueue, SharedMemoryRingBuffer
from common.precise_sleep import precise_wait


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


class WSGController(mp.Process):
    """WSG夹爪控制器 - 基于UMI实现"""
    
    def __init__(self,
            shm_manager: SharedMemoryManager,
            hostname,
            port=4242,
            frequency=30,
            home_to_open=True,
            move_max_speed=200.0,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.0,
            use_meters=False,
            verbose=False,
            # Grasp 相关参数
            use_grasp_for_closing=True,
            grasp_threshold=0.005,  # 5mm，闭合超过此阈值时使用grasp
            grasp_speed=0.05,       # grasp速度 (m/s)
            grasp_force=20.0,       # grasp力 (N)
            grasp_epsilon_inner=0.005,  # grasp内侧容差 (m)
            grasp_epsilon_outer=0.005,  # grasp外侧容差 (m)
            grasp_keep_threshold=0.035,  # 35mm，抓取保持的宽度上限
            release_width_threshold=0.040  # 40mm，超过此宽度时松开
            ):
        
        super().__init__()
        self.hostname = hostname
        self.port = port
        self.frequency = frequency
        self.home_to_open = home_to_open
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.scale = 1000.0 if use_meters else 1.0
        self.verbose = verbose
        
        # Grasp 参数
        self.use_grasp_for_closing = use_grasp_for_closing
        self.grasp_threshold = grasp_threshold
        self.grasp_speed = grasp_speed
        self.grasp_force = grasp_force
        self.grasp_epsilon_inner = grasp_epsilon_inner
        self.grasp_epsilon_outer = grasp_epsilon_outer
        self.grasp_keep_threshold = grasp_keep_threshold
        self.release_width_threshold = release_width_threshold

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # 构建输入队列
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # 构建环形缓冲区
        example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
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

    # ========= 启动方法 ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[WSGController] 控制器进程已启动，PID: {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
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
    def schedule_waypoint(self, pos: float, target_time: float):
        """调度夹爪位置"""
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        """重启数据记录"""
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= 接收API =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= 主循环进程 ============
    def run(self):
        # 启动连接
        try:
            if self.verbose:
                print(f"[WSGController] 连接到夹爪: {self.hostname}:{self.port}")
            
            # 使用ZeroRPC连接到Franka服务器
            wsg = None
            try:
                import zerorpc
                wsg = zerorpc.Client(timeout=5)  # 添加5秒超时
                wsg.connect(f"tcp://{self.hostname}:{self.port}")
                if self.verbose:
                    print(f"[WSGController] 使用ZeroRPC连接到Franka服务器成功: {self.hostname}:{self.port}")
            except Exception as e:
                if self.verbose:
                    print(f"[WSGController] ZeroRPC连接失败: {e}")
                    print(f"[WSGController] 尝试连接: {self.hostname}:{self.port}")
                wsg = None

            # 初始化位置插值器
            from common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
            
            curr_t = time.monotonic()
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[0.0, 0, 0, 0, 0, 0]]  # 只有x位置有效
            )
            
            t_start = time.monotonic()
            iter_idx = 0
            last_waypoint_time = curr_t
            keep_running = True
            last_gripper_width = 0.0  # 用于判断闭合/张开方向
            is_grasping = False  # 标记是否处于抓取状态
            grasp_engaged_width = 0.0  # 记录开始抓取时的宽度
            
            while keep_running:
                t_now = time.monotonic()
                
                # 命令夹爪
                target_pos = pose_interp(t_now)[0]
                target_vel = (target_pos - pose_interp(t_now - 1/self.frequency)[0]) * self.frequency

                # 控制夹爪
                if wsg is not None:
                    try:
                        # 通过ZeroRPC调用服务器端的gripper方法
                        current_width = wsg.get_gripper_width()
                        
                        info = {
                            'state': 0,  # 简化状态
                            'position': current_width,
                            'velocity': target_vel / self.scale,
                            'force_motor': 0.0,
                            'measure_timestamp': time.time()
                        }
                        
                        # 【核心修改】智能判断使用 move 还是 grasp
                        target_width_m = target_pos / self.scale
                        current_width_m = current_width
                        width_change = target_width_m - current_width_m
                        
                        # 【优先级1】判断是否应该松开（最高优先级）
                        # 明确的松开条件：目标宽度超过松开阈值（默认 40mm）
                        should_release = (
                            is_grasping and 
                            target_width_m > self.release_width_threshold
                        )
                        
                        if should_release:
                            # 明确的松开动作，退出抓取状态
                            is_grasping = False
                            if self.verbose:
                                print(f"[WSGController] ✓ 松开物体: 目标={target_width_m*1000:.1f}mm > {self.release_width_threshold*1000:.1f}mm")
                        
                        # 【优先级2】判断是否应该使用 grasp
                        # 1. 新的闭合动作：大幅度闭合触发
                        # 2. 保持抓取：已经在抓取且目标宽度仍然较小（< grasp_keep_threshold）
                        should_use_grasp = (
                            self.use_grasp_for_closing and
                            not should_release and  # 如果正在松开，则不 grasp
                            (width_change < -self.grasp_threshold or  # 大幅闭合
                             (is_grasping and target_width_m < self.grasp_keep_threshold))  # 或保持抓取
                        )
                        
                        if should_use_grasp:
                            # 使用 grasp：接触即停止，适合抓取物体
                            
                            # 【关键修改】只在第一次进入抓取状态时调用 grasp
                            # 之后不再调用，让 grasp 持续保持力
                            if not is_grasping:
                                # 第一次进入抓取状态
                                is_grasping = True
                                grasp_engaged_width = target_width_m
                                
                                if self.verbose:
                                    print(f"[WSGController] ✓ 初次抓取: 目标={target_width_m*1000:.1f}mm, "
                                          f"当前={current_width_m*1000:.1f}mm, "
                                          f"速度={self.grasp_speed*1000:.1f}mm/s, "
                                          f"力={self.grasp_force:.1f}N")
                                
                                # 只调用一次 grasp
                                try:
                                    result = wsg.grasp_gripper(
                                        width=target_width_m,
                                        speed=self.grasp_speed,
                                        force=self.grasp_force,
                                        epsilon_inner=self.grasp_epsilon_inner,
                                        epsilon_outer=self.grasp_epsilon_outer
                                    )
                                    if self.verbose:
                                        print(f"[WSGController] Grasp 命令已发送: {result}")
                                except (AttributeError, TypeError) as e:
                                    if self.verbose:
                                        print(f"[WSGController] 警告: Grasp调用失败 ({type(e).__name__}: {e})，降级使用 move")
                                    wsg.move_gripper_to_width(target_width_m)
                            else:
                                # 已经在抓取状态，不再调用 grasp，让底层保持持续施力
                                if self.verbose and iter_idx % 90 == 0:  # 每 3 秒打印一次
                                    print(f"[WSGController] 保持抓取: 当前宽度={current_width_m*1000:.1f}mm, "
                                          f"持续施加 {self.grasp_force:.1f}N 的力")
                        else:
                            # 使用 move：精确到达目标宽度，适合张开或小幅调整
                            # 【重要保护】如果正在抓取但不满足 should_use_grasp，不执行任何命令
                            # 避免误调用 move 中断 grasp
                            if is_grasping:
                                # 理论上不应该到这里（should_use_grasp 应该为 True）
                                # 如果到了这里，说明逻辑有问题，保持抓取，不调用 move
                                if self.verbose and iter_idx % 90 == 0:
                                    print(f"[WSGController] 警告: 抓取状态但不满足 grasp 条件，保持不动 "
                                          f"(目标={target_width_m*1000:.1f}mm)")
                            else:
                                # 正常的 move 操作（张开、微调等）
                                if self.verbose and iter_idx % 30 == 0 and abs(width_change) > 0.001:
                                    print(f"[WSGController] 使用 MOVE: 目标={target_width_m*1000:.1f}mm, "
                                          f"当前={current_width_m*1000:.1f}mm, "
                                          f"变化={width_change*1000:.1f}mm")
                                wsg.move_gripper_to_width(target_width_m)
                        
                        last_gripper_width = current_width_m
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"[WSGController] Gripper控制失败: {e}")
                        # 使用默认值
                        info = {
                            'state': 0,
                            'position': target_pos / self.scale,
                            'velocity': target_vel / self.scale,
                            'force_motor': 0.0,
                            'measure_timestamp': time.time()
                        }
                else:
                    # 模拟模式
                    info = {
                        'state': 0,
                        'position': target_pos / self.scale,
                        'velocity': target_vel / self.scale,
                        'force_motor': 0.0,
                        'measure_timestamp': time.time()
                    }
                
                # 获取机器人状态
                state = {
                    'gripper_state': info['state'],
                    'gripper_position': info['position'] / self.scale,
                    'gripper_velocity': info['velocity'] / self.scale,
                    'gripper_force': info['force_motor'],
                    'gripper_measure_timestamp': info['measure_timestamp'],
                    'gripper_receive_timestamp': time.time(),
                    'gripper_timestamp': time.time() - self.receive_latency
                }
                self.ring_buffer.put(state)

                # 从队列获取命令
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                
                # 执行命令
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        # 立即停止，忽略后续命令
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos'] * self.scale
                        target_time = command['target_time']
                        curr_time = t_now
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break
                    
                # 第一次循环成功，准备接收命令
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                
                # 调节频率
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
                
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[WSGController] 断开夹爪连接: {self.hostname}")
