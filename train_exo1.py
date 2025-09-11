from pathlib import Path
from exodata_joint1 import Dataset60  # 修改为新的数据集类

import torch
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# 创建输出目录
output_directory = Path("./outputs/train/exo_act3")
output_directory.mkdir(parents=True, exist_ok=True)

# 创建检查点目录
checkpoint_directory = output_directory / "checkpoints"
checkpoint_directory.mkdir(parents=True, exist_ok=True)

# 命令行参数
parser = argparse.ArgumentParser(description="外骨骼机械臂 ACT 训练脚本")
parser.add_argument("--data_root", type=str, default="/media/robotflow/Elements/train_data1",
                   help="数据根目录，包含 records_easy 和 scene_* 文件夹")
parser.add_argument("--chunk_size", type=int, default=32,
                   help="动作序列长度")
parser.add_argument("--joint_dim", type=int, default=8,
                   help="关节角度维度")
parser.add_argument("--batch_size", type=int, default=512,
                   help="批处理大小")
parser.add_argument("--training_steps", type=int, default=10000,
                   help="训练步数")
parser.add_argument("--learning_rate", type=float, default=1e-4,
                   help="学习率")
parser.add_argument("--log_freq", type=int, default=250,
                   help="日志记录频率")
parser.add_argument("--checkpoint_freq", type=int, default=2500,
                   help="检查点保存频率")
parser.add_argument("--provide_eih", action="store_true", default=False,
                   help="是否提供第二路图像 observation.image.eih 作为输入")
args = parser.parse_args()

# 训练参数
training_steps = args.training_steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_freq = args.log_freq
checkpoint_freq = args.checkpoint_freq

# 数据集参数
data_root = Path(args.data_root)
if not data_root.exists():
    raise FileNotFoundError(f"数据根目录不存在: {data_root.resolve()}")
chunk_size = args.chunk_size
joint_dim = args.joint_dim
batch_size = args.batch_size

print(f"外骨骼机械臂 ACT 训练")
print(f"数据根目录: {data_root.resolve()}")
print(f"设备: {device}")
print(f"训练参数: chunk_size={chunk_size}, joint_dim={joint_dim}, batch_size={batch_size}")
print(f"检查点保存频率: {checkpoint_freq} 步")

# 加载数据集
print(f"\n加载多视角外骨骼数据集...")
dataset = Dataset60(
    data_root=str(data_root),        # 数据根目录
    chunk_size=chunk_size,           # 动作序列长度
    joint_dim=joint_dim,             # 关节角度维度
    image_size=(224, 224),           # 图像尺寸
    provide_eih=args.provide_eih     # 是否提供第二路图像
)
print(f"数据集加载成功! 大小: {len(dataset)} 个样本")
print(args.provide_eih)

# 显示数据集统计信息
print(f"数据集统计:")
print(f"  - 关节角度范围: {dataset.stats['observation.state']['min']} 到 {dataset.stats['observation.state']['max']}")
print(f"  - 图像归一化: {dataset.stats['observation.image.color']['min']} 到 {dataset.stats['observation.image.color']['max']}")

# 创建 DataLoader
print(f"\n创建数据加载器...")
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=16,  # Windows 上设为 0
    batch_size=batch_size,
    shuffle=True,
    pin_memory=device != torch.device("cpu"),
    drop_last=True,
)
print(f" DataLoader 创建成功!")

# 创建 ACT 配置 - 适配单视角图像输入
print(f"\n创建 ACT 配置...")
input_shapes = {
    "observation.image.color": [3, 224, 224],
    "observation.state": [joint_dim],
}
input_norm_modes = {
    "observation.image.color": "min_max",
    "observation.state": "min_max",
}
if args.provide_eih:
    input_shapes["observation.image.eih"] = [3, 224, 224]
    input_norm_modes["observation.image.eih"] = "min_max"

config = ACTConfig(
    input_shapes=input_shapes,
    output_shapes={
        "action": [joint_dim],
    },
    chunk_size=chunk_size,
    n_action_steps=chunk_size,
    input_normalization_modes=input_norm_modes,
    output_normalization_modes={
        "action": "min_max",
    }
)
print(f"ACT 配置创建成功!")

# 创建策略
print(f"\n创建 ACT 策略...")
policy = ACTPolicy(config, dataset_stats=dataset.stats)
policy.to(device)
policy.train()
print(f"策略创建成功!")

# 显示模型信息
total_params = sum(p.numel() for p in policy.parameters())
trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

# 优化器
print(f"\n创建优化器...")
optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
print(f"优化器创建成功! 学习率: {args.learning_rate}")

# 训练循环
print(f"\n开始训练...")
print(f"目标步数: {training_steps}")
print(f"日志频率: {log_freq} 步")
print("-" * 60)

step = 0
done = False
total_loss = 0.0
loss_history = []  # 记录loss历史
steps_per_epoch = len(dataloader)  # 每个epoch的步数

try:
    while not done:
        for batch in dataloader:
            # 将数据移到设备
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            output_dict, _ = policy.forward(batch)
            loss = output_dict["loss"]
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # 累计损失
            total_loss += loss.item()
            loss_history.append(loss.item())  # 记录当前loss
            
            # 计算当前epoch
            current_epoch = step // steps_per_epoch + 1
            epoch_progress = (step % steps_per_epoch) / steps_per_epoch * 100
            
            # 日志记录
            if step % log_freq == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {current_epoch:3d} ({epoch_progress:5.1f}%) | Step {step:6d} | Loss: {loss.item():.6f} | Avg Loss: {avg_loss:.6f}")
                
                # 显示当前批次信息
                if step == 0:  # 只在第一步显示详细信息
                    print(f"批次信息:")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  - {key}: {value.shape} ({value.dtype})")
                        else:
                            print(f"  - {key}: {value}")
                    print(f"每个epoch步数: {steps_per_epoch}")
                    print("-" * 60)
            
            # 检查点保存
            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_path = checkpoint_directory / f"checkpoint_step_{step}.safetensors"
                try:
                    policy.save_pretrained(checkpoint_path)
                    print(f"检查点已保存: {checkpoint_path}")
                except Exception as e:
                    print(f"保存检查点失败: {e}")
            
            step += 1
            if step >= training_steps:
                done = True
                break
    
    print(f"\n训练完成!")
    print(f"总步数: {step}")
    print(f"总epoch数: {step // steps_per_epoch + 1}")
    print(f"最终损失: {loss.item():.6f}")
    print(f"平均损失: {total_loss / step:.6f}")
    
    # 绘制loss变化图
    print(f"\n绘制loss变化图...")
    try:
        plt.figure(figsize=(12, 6))
        
        # 绘制原始loss曲线
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, alpha=0.7, linewidth=0.8)
        plt.title('Training Loss (Raw)')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # 绘制平滑后的loss曲线（移动平均）
        plt.subplot(1, 2, 2)
        window_size = min(100, len(loss_history) // 10)  # 动态窗口大小
        if window_size > 1:
            smoothed_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(loss_history)), smoothed_loss, linewidth=1.5, color='red')
        else:
            plt.plot(loss_history, linewidth=1.5, color='red')
        plt.title(f'Training Loss (Smoothed, window={window_size})')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存loss图
        loss_plot_path = output_directory / "training_loss.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"Loss变化图已保存到: {loss_plot_path}")
        
        # 保存loss数据
        loss_data_path = output_directory / "loss_history.npy"
        np.save(loss_data_path, np.array(loss_history))
        print(f"Loss数据已保存到: {loss_data_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"绘制loss图失败: {e}")
        import traceback
        traceback.print_exc()
    
except KeyboardInterrupt:
    print(f"\n训练被用户中断")
    print(f"已训练步数: {step}")
    print(f"已训练epoch数: {step // steps_per_epoch + 1}")
    if step > 0:
        print(f"平均损失: {total_loss / step:.6f}")
        
        # 即使中断也绘制loss图
        if len(loss_history) > 0:
            print(f"\n绘制当前loss变化图...")
            try:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.plot(loss_history, alpha=0.7, linewidth=0.8)
                plt.title('Training Loss (Raw) - Interrupted')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                window_size = min(100, len(loss_history) // 10)
                if window_size > 1:
                    smoothed_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(range(window_size-1, len(loss_history)), smoothed_loss, linewidth=1.5, color='red')
                else:
                    plt.plot(loss_history, linewidth=1.5, color='red')
                plt.title(f'Training Loss (Smoothed) - Interrupted')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                loss_plot_path = output_directory / "training_loss_interrupted.png"
                plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
                print(f"Loss变化图已保存到: {loss_plot_path}")
                plt.show()
            except Exception as plot_e:
                print(f"绘制loss图失败: {plot_e}")

except Exception as e:
    print(f"\n训练过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
    if step > 0:
        print(f"已训练步数: {step}")
        print(f"已训练epoch数: {step // steps_per_epoch + 1}")
        print(f"平均损失: {total_loss / step:.6f}")
        
        # 即使出错也尝试绘制loss图
        if len(loss_history) > 0:
            print(f"\n绘制当前loss变化图...")
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(loss_history, alpha=0.7, linewidth=0.8)
                plt.title('Training Loss - Error Occurred')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                loss_plot_path = output_directory / "training_loss_error.png"
                plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
                print(f"Loss变化图已保存到: {loss_plot_path}")
                plt.show()
            except Exception as plot_e:
                print(f"绘制loss图失败: {plot_e}")

# 保存模型
print(f"\n保存模型...")
try:
    policy.save_pretrained(output_directory)
    print(f"模型已保存到: {output_directory}")
    
    # 保存训练配置
    config_path = output_directory / "training_config.txt"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"外骨骼机械臂 ACT 训练配置\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"数据根目录: {data_root}\n")
        f.write(f"数据集大小: {len(dataset)}\n")
        f.write(f"chunk_size: {chunk_size}\n")
        f.write(f"joint_dim: {joint_dim}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"训练步数: {step}\n")
        f.write(f"训练epoch数: {step // steps_per_epoch + 1 if step > 0 else 0}\n")
        f.write(f"每个epoch步数: {steps_per_epoch}\n")
        f.write(f"学习率: {args.learning_rate}\n")
        f.write(f"检查点保存频率: {checkpoint_freq}\n")
        f.write(f"最终损失: {loss.item() if 'loss' in locals() else 'N/A'}\n")
        f.write(f"平均损失: {total_loss / step if step > 0 else 'N/A'}\n")
        f.write(f"设备: {device}\n")
    
    print(f"训练配置已保存到: {config_path}")
    
except Exception as e:
    print(f"保存模型失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n训练脚本执行完成!") 