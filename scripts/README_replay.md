# 轨迹复现功能说明

这个目录包含了轨迹复现功能，可以将采集的轨迹数据通过`franka_control_umi`系统进行复现。

## 文件说明

- `replay_trajectory.py`: 轨迹复现脚本，正确实现模型推理模拟
- `start_replay.sh`: 启动脚本，方便使用
- `test_replay.py`: 测试脚本
- `README_replay.md`: 本说明文件

## 工作原理

1. **数据加载**: 从指定目录加载`.npy`格式的轨迹文件
2. **预处理转换**: 一次性将所有关节数据通过正向运动学转换为6维动作数据
3. **策略模拟**: 使用`TrajectoryPolicy`按时间索引获取预计算的动作，模拟模型推理过程
4. **动作执行**: 通过`PolicyInterface`将动作发送到UMI系统，由UMI系统负责插值控制

## 数据格式要求

轨迹文件应为`.npy`格式，包含8维数据：
- 前7维: 关节角度(度数)
- 第8维: 夹爪数据(度数)

文件命名支持：
- `angle_cam0_*.npy` (优先)
- `*.npy` (回退)

## 使用方法

### 方法1: 使用启动脚本 (推荐)

```bash
# 使用默认参数
./start_replay.sh

# 指定参数
./start_replay.sh --angles_dir /path/to/your/angles --policy_frequency 30.0
```

### 方法2: 直接运行Python脚本

```bash
python3 replay_trajectory.py \
    --config config/robot_config.yaml \
    --angles_dir /path/to/your/angles \
    --policy_frequency 20.0 \
    --data_frequency 30.0
```

## 参数说明

- `--config`: 机器人配置文件路径 (默认: `config/robot_config.yaml`)
- `--angles_dir`: 轨迹数据目录 (默认: `/home/franka/code/polymetis/examples/angles1/angles`)
- `--policy_frequency`: 策略推理频率 (默认: 20.0Hz)
- `--data_frequency`: 数据采集频率 (默认: 30.0Hz)

## 与replay3_fixed.py的对比

| 特性 | replay3_fixed.py | replay_trajectory.py |
|------|------------------|---------------------|
| 控制方式 | 直接Polymetis接口 | franka_control_umi系统 |
| 插值方式 | 30Hz→1000Hz | 预处理+策略推理 |
| 动作格式 | 关节角度 | 6维末端姿态 |
| 系统集成 | 独立脚本 | 集成到UMI系统 |
| 策略模拟 | 无 | 支持策略接口 |
| 数据转换 | 实时插值 | 预处理转换 |

## 关键代码片段

### 正向运动学计算
```python
# 与replay3_fixed.py保持一致
target_joints_tensor = torch.from_numpy(target_joints)
pose, quat = self.robot_model.forward_kinematics(target_joints_tensor)
pose = torch.from_numpy(pose)
quat = torch.from_numpy(quat)
pose = pose.to(torch.float32)
quat = quat.to(torch.float32)
```

### 动作构建
```python
# 将四元数转换为旋转向量
quat_np = quat.numpy()
rot_matrix = st.Rotation.from_quat(quat_np)
rotvec = rot_matrix.as_rotvec()

# 构建6维动作 [位置3 + 旋转3]
action = np.concatenate([pose.numpy(), rotvec])
```

## 注意事项

1. **依赖检查**: 确保已安装所有必要的依赖包
2. **配置文件**: 确保机器人配置文件正确
3. **数据格式**: 确保轨迹数据格式符合要求
4. **网络连接**: 确保机器人网络连接正常
5. **安全第一**: 确保在机器人附近有紧急停止按钮

## 故障排除

### 常见问题

1. **配置文件不存在**
   ```
   错误: 配置文件不存在: config/robot_config.yaml
   ```
   解决: 检查配置文件路径是否正确

2. **数据目录不存在**
   ```
   错误: 数据目录不存在: /path/to/angles
   ```
   解决: 检查数据目录路径是否正确

3. **机器人连接失败**
   ```
   机器人运动学模型初始化失败: Connection refused
   ```
   解决: 检查机器人IP地址和网络连接

4. **数据格式错误**
   ```
   维度异常，期望 8，得到 7
   ```
   解决: 检查轨迹数据文件格式

### 调试模式

在配置文件中设置 `verbose: true` 可以启用详细日志输出。

## 扩展功能

可以根据需要扩展以下功能：
- 支持更多数据格式
- 添加轨迹可视化
- 支持轨迹编辑
- 添加性能监控
- 支持多机器人控制
