#!/bin/bash

# 轨迹复现启动脚本

echo "启动轨迹复现..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import numpy, scipy, torch, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要的Python依赖"
    echo "请运行: pip install -r requirements.txt"
    exit 1
fi

# 设置默认参数
CONFIG_FILE="../config/robot_config.yaml"
ANGLES_DIR="/media/robotflow/Elements/train_data/records_easy/record_20250823_133050/angles"
POLICY_FREQ=20.0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --angles_dir)
            ANGLES_DIR="$2"
            shift 2
            ;;
        --policy_frequency)
            POLICY_FREQ="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --config CONFIG_FILE        配置文件路径 (默认: ../config/robot_config.yaml)"
            echo "  --angles_dir ANGLES_DIR     轨迹数据目录 (默认: /home/franka/code/polymetis/examples/angles1/angles)"
            echo "  --policy_frequency FREQ     策略推理频率 (默认: 20.0)"
            echo "  --help                      显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 启动轨迹复现
cd "$(dirname "$0")"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查数据目录
if [ ! -d "$ANGLES_DIR" ]; then
    echo "错误: 数据目录不存在: $ANGLES_DIR"
    exit 1
fi

echo "配置信息:"
echo "  配置文件: $CONFIG_FILE"
echo "  数据目录: $ANGLES_DIR"
echo "  策略频率: ${POLICY_FREQ}Hz"
python3 replay_trajectory.py \
    --config "$CONFIG_FILE" \
    --angles_dir "$ANGLES_DIR" \
    --policy_frequency "$POLICY_FREQ"
