# RISE环境配置完整指南

## 📋 概述

本指南记录了在Ubuntu系统上配置RISE（基于点云的机器人策略）环境的完整步骤，包括MinkowskiEngine、PyTorch3D、r3kit等依赖的安装。

## 🎯 最终配置状态

| 组件 | 版本 | 状态 |
|------|------|------|
| Python | 3.9.23 | ✅ 成功 |
| PyTorch | 2.4.0+cu121 | ✅ 成功 |
| TorchVision | 0.19.0+cu121 | ✅ 成功 |
| NumPy | 1.26.4 | ✅ 成功 |
| MinkowskiEngine | 0.5.4 | ✅ 成功 |
| PyTorch3D | 0.7.8 | ✅ 成功 |
| CUDA | 12.1 | ✅ 可用 |
| r3kit | 最新版本 | ✅ 成功 |

## 🚀 安装步骤

### 步骤1：创建Conda环境

```bash
# 创建新的conda环境
conda create -n minkowski_cuda12 python=3.9 -y

# 激活环境
conda activate minkowski_cuda12
```

### 步骤2：安装PyTorch和CUDA工具链

```bash
# 安装PyTorch 2.4.0和CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装OpenBLAS
conda install -y openblas-devel -c anaconda

# 安装CUDA工具链
conda install -y nvidia/label/cuda-12.1.0::cuda-toolkit
```

### 步骤3：修复GCC兼容性问题

**重要：** 这是解决MinkowskiEngine编译问题的关键步骤！

```bash
# 备份原文件
sudo cp /usr/include/c++/9/bits/shared_ptr_base.h /usr/include/c++/9/bits/shared_ptr_base.h.backup

# 修复std::__to_address冲突
sudo sed -i 's/auto __raw = __to_address(__r\\.get());/auto __raw = std::__to_address(__r.get());/g' /usr/include/c++/9/bits/shared_ptr_base.h
```

### 步骤4：安装MinkowskiEngine

```bash
# 克隆MinkowskiEngine（使用CUDA 12兼容分支）
git clone https://github.com/CiSong10/MinkowskiEngine.git
cd MinkowskiEngine
git checkout cuda12-installation

# 设置编译器
export CC=gcc-9
export CXX=g++-9

# 安装兼容的依赖版本
pip install numpy==1.21.6
pip install setuptools==69.5.1

# 编译安装MinkowskiEngine
python setup.py install --blas=openblas
```

### 步骤5：安装RISE基础依赖

```bash
# 回到项目根目录
cd /path/to/your/project/RISE

# 安装基础依赖
pip install tqdm==4.66.2 open3d==0.18.0 easydict==1.13 einops==0.7.0 matplotlib==3.7.5 opencv-python==4.9.0.80 diffusers==0.27.2 huggingface-hub==0.25.2
```

### 步骤6：安装PyTorch3D

```bash
# 创建依赖目录
mkdir -p dependencies
cd dependencies

# 克隆并安装PyTorch3D
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```

### 步骤7：安装r3kit

```bash
# 回到r3kit目录
cd /path/to/your/project/model/r3kit

# 安装缺失的依赖
pip install psutil yourdfpy pynput

# 安装r3kit
pip install -e .
```

## 🧪 验证安装

### 验证MinkowskiEngine

```python
import torch
import MinkowskiEngine as ME
import numpy as np

# 测试MinkowskiEngine基本功能
coords = torch.randint(0, 100, (100, 4), dtype=torch.int32).cuda()
feats = torch.randn(100, 3).cuda()

sparse_tensor = ME.SparseTensor(features=feats, coordinates=coords)
print('✅ 稀疏张量创建成功')

conv = ME.MinkowskiConvolution(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    dimension=3
).cuda()

output = conv(sparse_tensor)
print('✅ 卷积层测试成功')
print(f'输出特征形状: {output.features.shape}')
```

### 验证RISE环境

```python
import torch
import MinkowskiEngine as ME
import pytorch3d
import numpy as np

print('✅ PyTorch:', torch.__version__)
print('✅ MinkowskiEngine:', ME.__version__)
print('✅ PyTorch3D:', pytorch3d.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
```

## 🐛 常见问题解决

### 问题1：CUDA错误 - invalid configuration argument

**症状：**
```
RuntimeError: CUDA error: invalid configuration argument
```

**解决方案：**
1. 设置CUDA环境变量：
```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'
```

2. 确保模型在评估模式：
```python
model.eval()
```

3. 添加数据验证和错误处理

### 问题2：GCC编译错误

**症状：**
```
error: 'std::__to_address' is ambiguous
```

**解决方案：**
按照步骤3修复GCC兼容性问题

### 问题3：依赖版本冲突

**症状：**
```
ERROR: pip's dependency resolver does not currently have a strategy
```

**解决方案：**
1. 使用指定的版本号安装
2. 必要时使用 `--force-reinstall` 参数
3. 考虑使用conda环境隔离依赖

## 📝 推理脚本配置

### 环境变量设置

在推理脚本中添加：

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用CUDA同步调试
os.environ['OMP_NUM_THREADS'] = '8'       # 限制OpenMP线程数
```

### 数据验证

```python
def create_input(self, color_image, depth_image, cam_intrinsics):
    cloud = self.create_point_cloud(color_image, depth_image, cam_intrinsics)
    
    # 数据验证和清理
    if len(cloud) == 0:
        print("警告: 点云为空，使用默认点云")
        cloud = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    
    # 检查点云数据是否有效
    if np.any(np.isnan(cloud)) or np.any(np.isinf(cloud)):
        print("警告: 点云包含无效值，进行清理")
        valid_mask = ~(np.isnan(cloud).any(axis=1) | np.isinf(cloud).any(axis=1))
        cloud = cloud[valid_mask]
        
        if len(cloud) == 0:
            print("警告: 清理后点云为空，使用默认点云")
            cloud = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    
    coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype=np.int32)
    
    # 检查坐标范围
    if np.any(np.abs(coords) > 100000):
        print("警告: 坐标值过大，进行裁剪")
        coords = np.clip(coords, -100000, 100000)
    
    coords_batch, feats_batch = self.create_batch(coords, cloud)
    return coords_batch, feats_batch, cloud
```

## 🎉 完成

按照以上步骤，你应该能够成功配置RISE环境并运行推理脚本。如果遇到问题，请参考常见问题解决部分或检查具体错误信息。

## 📚 参考文档

- [MinkowskiEngine官方文档](https://github.com/NVIDIA/MinkowskiEngine)
- [PyTorch3D官方文档](https://github.com/facebookresearch/pytorch3d)
- [RISE项目文档](https://github.com/your-repo/RISE)

---

**注意：** 本指南基于Ubuntu系统，其他操作系统可能需要调整部分步骤。建议在虚拟环境中进行安装以避免系统依赖冲突。
