#!/bin/bash

# Franka接口服务器启动脚本

echo "启动Franka接口服务器..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import polymetis, zerorpc, numpy, scipy, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要的Python依赖"
    echo "请运行: pip install -r requirements.txt"
    exit 1
fi

# 启动服务器
echo "启动ZeroRPC服务器..."
cd "$(dirname "$0")"
python3 launch_franka_interface_server.py
