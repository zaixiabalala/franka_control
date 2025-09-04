# Franka Control - Joint Space Control

这是从 `franka_control_umi` 重构而来的版本，主要变化是将控制方式从笛卡尔位姿控制改为关节空间控制。

## 主要修改

### 1. 配置文件修改
- `config/robot_config.yaml`:
  - `action_dim`: 从 6 改为 7 (7个关节角度)
  - `obs_dim`: 从 13 改为 14 (7个关节位置 + 7个关节速度)

### 2. 控制器修改
- `real_world/franka_interpolation_controller.py`:
  - 共享内存结构: `target_pose` 字段从6维改为7维，现在表示关节角度
  - 控制逻辑: 从 `update_desired_ee_pose` 改为 `update_desired_joint_positions`
  - 插值器: 简化为直接使用关节位置，不再需要复杂的位姿插值

### 3. 策略接口修改
- `policy_interface.py`:
  - `execute_action`: 动作形状从 (6,) 改为 (7,)
  - `SimplePolicy`: 从位姿控制改为关节位置控制
  - 观测构建: 主要使用关节信息，保留位姿信息用于兼容性

### 4. 示例策略修改
- `scripts/example_policy.py`:
  - 目标从位姿改为关节角度
  - 默认目标关节: `[0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0]`

### 5. 轨迹复现修改
- `scripts/replay_trajectory.py`:
  - 直接使用关节数据作为轨迹数据，不再需要正向运动学转换
  - `TrajectoryPolicy`: 输出7维关节角度而不是6维位姿
  - 观测使用关节位置而不是末端执行器位姿

## 使用方法

### 运行示例策略
```bash
cd franka_control_final
python scripts/example_policy.py
```

### 运行轨迹复现
```bash
cd franka_control_final
python scripts/replay_trajectory.py
```

## 优势

1. **简化控制**: 直接控制关节角度，避免了复杂的位姿插值和正向运动学计算
2. **提高稳定性**: 关节空间控制通常比笛卡尔空间控制更稳定
3. **减少延迟**: 不需要实时进行正向运动学计算
4. **更好的轨迹跟踪**: 直接控制关节位置，可以更精确地复现录制的轨迹

## 注意事项

1. 确保机器人服务器支持 `update_desired_joint_positions` 方法
2. 关节角度单位是弧度 (rad)
3. 关节限制需要在使用前检查
4. 某些复杂的轨迹可能需要额外的关节限制处理

## 兼容性

- 保持了原有的接口结构，只是改变了数据含义
- 观测字典仍然包含位姿信息，用于向后兼容
- 共享内存结构保持兼容，只是字段含义改变
