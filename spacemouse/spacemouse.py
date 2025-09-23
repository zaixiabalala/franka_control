"""
空间鼠标控制核心模块
基于 3Dconnexion SpaceMouse 的 6DOF 控制接口
"""
import multiprocessing as mp
import numpy as np
import time
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared_memory.shared_memory_queue import SharedMemoryQueue

from common.precise_sleep import precise_wait
try:
    from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
    SPNAV_AVAILABLE = True
except ImportError:
    print("警告: spnav 库未安装，将使用模拟模式")
    SPNAV_AVAILABLE = False

class Spacemouse(mp.Process):
    def __init__(self, 
            shm_manager=None, 
            get_max_k=30, 
            frequency=200,
            max_value=500, 
            deadzone=(0,0,0,0,0,0), 
            dtype=np.float32,
            n_buttons=2,
            ):
        """
        空间鼠标控制器
        
        Args:
            shm_manager: 共享内存管理器（可选）
            get_max_k: 最大缓存数量
            frequency: 采样频率
            max_value: 最大值范围 {300, 500} 300为有线版本，500为无线版本
            deadzone: 死区设置 [0,1]，低于此值的轴将保持为0
            dtype: 数据类型
            n_buttons: 按钮数量
        
        坐标系说明:
        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        super().__init__()
        
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        # 参数设置
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons
        
        # 坐标变换矩阵：从空间鼠标坐标系到右手坐标系
        self.tx_zup_spnav = np.array([
            [0,0,-1],
            [1,0,0],
            [0,1,0]
        ], dtype=dtype)

        # 创建示例数据结构
        example = {
            # 3个平移 + 3个旋转 + 1个周期
            'motion_event': np.zeros((7,), dtype=np.int64),
            # 左右按钮状态
            'button_state': np.zeros((n_buttons,), dtype=bool),
            'receive_timestamp': time.time()
        }
        
        # 创建环形缓冲区
        if shm_manager is not None:
            self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=shm_manager, 
                examples=example,
                get_max_k=get_max_k,
                get_time_budget=0.2,
                put_desired_frequency=frequency
            )
        else:
            self.ring_buffer = SharedMemoryRingBuffer(
                examples=example,
                max_k=get_max_k,
                time_budget=0.2
            )

        # 进程控制事件
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

    # ======= 状态获取 API ==========

    def get_motion_state(self):
        """获取原始运动状态"""
        state = self.ring_buffer.get()
        if state is None:
            return np.zeros(6, dtype=self.dtype)
            
        state = np.array(state['motion_event'][:6], dtype=self.dtype) / self.max_value
        # 应用死区
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state
    
    def get_motion_state_transformed(self):
        """
        获取变换后的运动状态（右手坐标系）
        
        坐标系:
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back
        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]  # 平移
        tf_state[3:] = self.tx_zup_spnav @ state[3:]  # 旋转
        return tf_state

    def get_button_state(self):
        """获取按钮状态"""
        state = self.ring_buffer.get()
        if state is None:
            return np.zeros(self.n_buttons, dtype=bool)
        return state['button_state']
    
    def is_button_pressed(self, button_id):
        """检查指定按钮是否被按下"""
        return self.get_button_state()[button_id]
    
    #========== 启动停止 API ===========

    def start(self, wait=True):
        """启动空间鼠标进程"""
        super().start()
        if wait:
            self.ready_event.wait()
    
    def stop(self, wait=True):
        """停止空间鼠标进程"""
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= 主循环 ==========
    def run(self):
        """主循环：监听空间鼠标事件"""
        if not SPNAV_AVAILABLE:
            print("警告: 使用模拟模式运行空间鼠标")
            self._run_simulation()
            return
            
        try:
            spnav_open()
            self._run_real()
        except Exception as e:
            print(f"空间鼠标初始化失败: {e}")
            print("切换到模拟模式")
            self._run_simulation()
        finally:
            if SPNAV_AVAILABLE:
                spnav_close()
    
    def _run_real(self):
        """真实的空间鼠标事件处理"""
        motion_event = np.zeros((7,), dtype=np.int64)
        button_state = np.zeros((self.n_buttons,), dtype=bool)
        
        # 发送初始消息
        self.ring_buffer.put({
            'motion_event': motion_event,
            'button_state': button_state,
            'receive_timestamp': time.monotonic()
        })
        self.ready_event.set()

        while not self.stop_event.is_set():
            event = spnav_poll_event()
            receive_timestamp = time.monotonic()
            
            if isinstance(event, SpnavMotionEvent):
                motion_event[:3] = event.translation
                motion_event[3:6] = event.rotation
                motion_event[6] = event.period
            elif isinstance(event, SpnavButtonEvent):
                button_state[event.bnum] = event.press
            else:
                # 完成这轮事件处理后发送
                self.ring_buffer.put({
                    'motion_event': motion_event,
                    'button_state': button_state,
                    'receive_timestamp': receive_timestamp
                })
                precise_wait(1/self.frequency)
    
    def _run_simulation(self):
        """模拟模式：用于测试和开发"""
        motion_event = np.zeros((7,), dtype=np.int64)
        button_state = np.zeros((self.n_buttons,), dtype=bool)
        
        # 发送初始消息
        self.ring_buffer.put({
            'motion_event': motion_event,
            'button_state': button_state,
            'receive_timestamp': time.time()
        })
        self.ready_event.set()
        
        print("空间鼠标模拟模式已启动")
        print("提示: 安装 spnav 库以使用真实空间鼠标")
        
        while not self.stop_event.is_set():
            # 模拟模式：发送零状态
            self.ring_buffer.put({
                'motion_event': motion_event,
                'button_state': button_state,
                'receive_timestamp': time.time()
            })
            time.sleep(1/self.frequency)
