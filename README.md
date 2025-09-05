# Franka Control Final

基于UMI (Universal Manipulation Interface) 重构的Franka机器人控制系统，支持关节空间控制和gripper控制。

## 功能特性

- **关节空间控制**: 直接控制7个关节角度，提供更稳定的控制性能
- **Gripper控制**: 支持WSG gripper的编码器控制和宽度控制
- **共享内存系统**: 高效的进程间数据共享，支持实时控制
- **轨迹复现**: 支持录制轨迹的精确复现
- **插值控制器**: 平滑的轨迹控制，支持路径点调度
- **策略接口**: 统一的策略控制接口，易于集成不同的控制策略

## 目录结构

```
franka_control_final/
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
│   ├── joint_trajectory_interpolator.py
│   ├── gripper_util.py
│   ├── interpolation_util.py
│   └── precise_sleep.py
├── real_world/            # 真实世界控制
│   ├── __init__.py
│   ├── franka_interface.py
│   ├── franka_interpolation_controller.py
│   └── wsg_controller.py
├── scripts/               # 脚本文件
│   ├── launch_franka_interface_server.py
│   ├── control_franka.py
│   ├── example_policy.py
│   ├── replay_trajectory.py
│   ├── start_server.sh
│   └── start_replay.sh
├── config/                # 配置文件
│   └── robot_config.yaml
├── policy_interface.py    # 策略接口
├── requirements.txt
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

### 2. 运行示例策略

```bash
python scripts/example_policy.py
```

### 3. 轨迹复现

```bash
python scripts/replay_trajectory.py --data_dir /path/to/trajectory/data
```

## 配置说明

配置文件 `config/robot_config.yaml` 包含以下主要参数：

```yaml
robot:
  ip: "192.168.1.2"        # 机器人IP地址
  port: 4242              # 端口
  frequency: 1000         # 控制频率 (Hz)
  Kx_scale: 1.0          # 位置增益缩放
  Kxd_scale: 1.0         # 速度增益缩放

gripper:
  hostname: "192.168.1.2" # Gripper IP地址
  port: 4242             # 端口（与机器人相同）
  frequency: 30          # 控制频率 (Hz)
  move_max_speed: 200.0  # 最大移动速度

policy:
  action_dim: 7          # 动作维度 (7个关节角度)
  obs_dim: 14            # 观测维度 (关节位置7 + 关节速度7)
  action_latency: 0.1    # 动作延迟补偿 (秒)
  obs_latency: 0.0001    # 观测延迟补偿 (秒)
```

## 主要特性

### 关节空间控制

- 直接控制7个关节角度，避免复杂的位姿插值
- 提供更稳定的控制性能和更精确的轨迹跟踪
- 支持关节限制和碰撞检测

### Gripper控制

- 支持WSG gripper的编码器控制和宽度控制
- 自动转换编码器值到gripper宽度
- 支持实时gripper状态监控

### 轨迹复现

- 支持录制轨迹的精确复现
- 自动处理关节角度和gripper编码器数据
- 支持多种数据格式和时间同步

## API 参考

### PolicyInterface

主要的策略控制接口类。

#### 方法

- `start()`: 启动策略接口
- `stop()`: 停止策略接口
- `get_observation()`: 获取当前观测
- `execute_action(action, target_time)`: 执行动作
- `execute_gripper_action(gripper_encoder, target_time)`: 执行gripper动作
- `run_policy(max_steps, step_callback)`: 运行策略

### FrankaInterpolationController

Franka机器人插值控制器。

#### 方法

- `servoL(joints, duration)`: 伺服到指定关节位置
- `schedule_waypoint(joints, target_time)`: 调度关节路径点
- `get_state(k, out)`: 获取机器人状态
- `get_all_state()`: 获取所有状态数据

### WSGController

Gripper控制器。

#### 方法

- `schedule_waypoint(width, target_time)`: 调度gripper宽度
- `get_state()`: 获取gripper状态

## 使用示例

### 基本控制

```python
import numpy as np
from policy_interface import create_policy_interface

def simple_policy(obs):
    """简单策略：移动到目标关节位置"""
    target_joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
    return target_joints

def gripper_policy(obs):
    """Gripper策略：根据时间控制gripper"""
    import time
    current_time = time.time()
    # 简单的开合控制
    if int(current_time) % 2 == 0:
        return 0.0  # 关闭
    else:
        return 0.05  # 打开到5cm

# 创建策略接口
interface = create_policy_interface('config/robot_config.yaml', simple_policy)

# 添加gripper策略
interface.policy_fn.get_gripper_action = gripper_policy

# 运行策略
with interface:
    interface.run_policy(max_steps=100)
```

### 轨迹复现

```python
from scripts.replay_trajectory import CorrectTrajectoryReplayer

# 创建轨迹复现器
replayer = CorrectTrajectoryReplayer(
    data_dir="/path/to/trajectory/data",
    config_path="config/robot_config.yaml"
)

# 运行轨迹复现
replayer.run()
```

## 注意事项

1. **安全第一**: 确保在机器人附近有紧急停止按钮
2. **网络连接**: 确保机器人IP地址配置正确
3. **关节限制**: 使用前检查关节角度是否在安全范围内
4. **Gripper安全**: 确保gripper不会夹到物体或手指

## 故障排除

### 常见问题

1. **连接失败**: 检查机器人IP地址和网络连接
2. **控制不稳定**: 调整Kx_scale和Kxd_scale参数
3. **Gripper不工作**: 检查gripper配置和服务器连接
4. **轨迹复现失败**: 检查数据格式和文件路径

### 调试模式

在配置文件中设置 `verbose: true` 可以启用详细日志输出。

## 许可证

本项目基于UMI项目提取，遵循相应的开源许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。