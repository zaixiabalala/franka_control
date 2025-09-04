# Franka Control UMI

基于UMI (Universal Manipulation Interface) 提取的Franka机器人控制系统，专门为单臂Franka机器人设计。

## 功能特性

- **共享内存系统**: 高效的进程间数据共享，支持实时控制
- **Franka控制接口**: 通过ZeroRPC与机器人通信，支持笛卡尔阻抗控制
- **插值控制器**: 平滑的轨迹控制，支持路径点调度
- **策略接口**: 统一的策略控制接口，易于集成不同的控制策略
- **通用工具**: 姿态处理、插值、时间控制等实用工具

## 目录结构

```
franka_control_umi/
├── shared_memory/          # 共享内存系统
│   ├── __init__.py
│   ├── shared_ndarray.py
│   ├── shared_memory_ring_buffer.py
│   ├── shared_memory_queue.py
│   └── shared_memory_util.py
├── common/                 # 通用工具
│   ├── __init__.py
│   ├── pose_util.py
│   ├── pose_trajectory_interpolator.py
│   ├── interpolation_util.py
│   └── precise_sleep.py
├── real_world/            # 真实世界控制
│   ├── __init__.py
│   ├── franka_interface.py
│   └── franka_interpolation_controller.py
├── scripts/               # 脚本文件
│   ├── launch_franka_interface_server.py
│   └── control_franka.py
├── config/                # 配置文件
│   └── robot_config.yaml
├── policy_interface.py    # 策略接口
├── __init__.py
└── README.md
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 启动Franka接口服务器

首先需要启动ZeroRPC服务器来与Franka机器人通信：

```bash
python scripts/launch_franka_interface_server.py
```

### 2. 基本控制示例

```python
import numpy as np
from franka_control_umi import PolicyInterface, SimplePolicy

# 创建简单策略
target_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])  # 目标姿态
policy = SimplePolicy(target_pose)

# 创建策略接口
interface = PolicyInterface('config/robot_config.yaml', policy)

# 运行策略
with interface:
    interface.run_policy(max_steps=100)
```

### 3. 高级控制示例

```python
import time
import numpy as np
from franka_control_umi import FrankaInterpolationController
from multiprocessing.managers import SharedMemoryManager

def custom_policy(obs):
    """自定义策略函数"""
    current_pose = np.concatenate([
        obs['robot0_eef_pos'],
        obs['robot0_eef_rot_axis_angle']
    ])
    
    # 实现你的控制逻辑
    target_pose = current_pose + np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    return target_pose

# 使用策略接口
interface = PolicyInterface('config/robot_config.yaml', custom_policy)

with interface:
    # 获取当前观测
    obs = interface.get_observation()
    print(f"当前位置: {obs['robot0_eef_pos']}")
    
    # 执行动作
    action = custom_policy(obs)
    interface.execute_action(action)
```

## 配置说明

配置文件 `config/robot_config.yaml` 包含以下主要参数：

```yaml
robot:
  ip: "172.16.0.3"        # 机器人IP地址
  port: 4242              # 端口
  frequency: 1000         # 控制频率 (Hz)
  Kx_scale: 1.0          # 位置增益缩放
  Kxd_scale: 1.0         # 速度增益缩放

policy:
  action_dim: 6          # 动作维度 (位置3 + 旋转3)
  obs_dim: 13            # 观测维度 (关节位置7 + 关节速度6)
  action_latency: 0.1    # 动作延迟补偿 (秒)
  obs_latency: 0.0001    # 观测延迟补偿 (秒)
```

## API 参考

### PolicyInterface

主要的策略控制接口类。

#### 方法

- `start()`: 启动策略接口
- `stop()`: 停止策略接口
- `get_observation()`: 获取当前观测
- `execute_action(action, target_time)`: 执行动作
- `run_policy(max_steps, step_callback)`: 运行策略

### FrankaInterpolationController

Franka机器人插值控制器。

#### 方法

- `servoL(pose, duration)`: 伺服到指定姿态
- `schedule_waypoint(pose, target_time)`: 调度路径点
- `get_state(k, out)`: 获取机器人状态
- `get_all_state()`: 获取所有状态数据

## 注意事项

1. **安全第一**: 确保在机器人附近有紧急停止按钮
2. **网络连接**: 确保机器人IP地址配置正确
3. **权限设置**: 如果使用软实时控制，需要运行相应的权限设置脚本
4. **策略验证**: 在真实机器人上运行前，建议先在仿真环境中测试

## 故障排除

### 常见问题

1. **连接失败**: 检查机器人IP地址和网络连接
2. **控制不稳定**: 调整Kx_scale和Kxd_scale参数
3. **延迟问题**: 检查action_latency和obs_latency配置

### 调试模式

在配置文件中设置 `verbose: true` 可以启用详细日志输出。

## 许可证

本项目基于UMI项目提取，遵循相应的开源许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。
