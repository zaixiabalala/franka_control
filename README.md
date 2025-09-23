# Franka Control

提供 SpaceMouse 控制 Franka、30Hz 数据采集、多相机同步与策略推理示例。

## FrankaInterface 与 Polymetis 部署

- 组件位置参考：
  - FrankaInterface (Server)：`scripts/launch_franka_interface_server.py`（L7）
  - FrankaInterface (Client)：`real_world/franka_interpolation_controller.py`（L36）
  - FrankaInterpolationController：`real_world/franka_interpolation_controller.py`（L71）

- 步骤：
  1) 按官方文档在带实时内核的电脑（例如 NUC）安装 Polymetis。
  2) 在 NUC 上启动 FrankaInterface Server：
     ```bash
     python scripts/launch_franka_interface_server.py
     ```
  3) （可选）在另一台桌面端（用于运行策略）通过 SpaceMouse 控制 Franka：
     ```bash
     python spacemouse/control_franka_spacemouse.py
     ```
  4) 修改 `config\robot_config.yaml`：适配你的机器人


## 安装
- client端创建环境
```bash
conda create -n franka python=3.10 -y
conda activate franka
```
- 安装基础依赖
```bash
pip install -r franka_control/requirements.txt
```
- 可选组件
  - r3kit/RealSense: 访问 https://github.com/ForceMimic/r3kit 安装
  - LeRobot: 访问 https://github.com/huggingface/lerobot 安装
  - RISE: 访问 https://github.com/rise-policy/rise 安装
  - SpaceMouse(真实设备): Linux/WSL 安装 `spacenavd` 与 Python 绑定


## 快速开始

- 部署lerobot封装的act
```bash
python franka_control/scripts/inference_with_policy_interface_act.py
```

- spacemouse 30Hz 数据采集
```bash
python franka_control/scripts/collect_data_franka_spacemouse.py
```

- 采集数据轨迹复现
```bash
python franka_control/scripts/replay_trajectory.py
```
  
## 目录结构
```
franka_control/
  common/            # 插值、姿态、定时工具
  real_world/        # 机器人/夹爪/相机系统
  shared_memory/     # 共享内存环形缓冲/队列
  spacemouse/        # 空间鼠标接口与控制脚本
  scripts/           # 数据采集与policy部署脚本
  requirements.txt   # 依赖清单
```

## 致谢

本项目基于 [UMI (Universal Manipulation Interface)](https://github.com/UMI-Robot/universal_manipulation_interface) 项目提取和重构。

### 主要改进
- 从笛卡尔位姿控制重构为关节空间控制
- 添加了WSG gripper控制支持
- 简化了轨迹复现流程
- 优化了代码结构和文档

### 原始项目
- **UMI**: Universal Manipulation Interface
- **GitHub**: https://github.com/UMI-Robot/universal_manipulation_interface
- **论文**: Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Data

## 许可证

本项目基于UMI项目提取，遵循相应的开源许可证。请参考原始UMI项目的许可证条款。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。


