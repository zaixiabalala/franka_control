#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gripper工具函数
"""

import numpy as np


def convert_gripper_encoder_to_width(encoder_value):
    """
    将gripper编码器值转换为夹爪宽度(米)
    
    Args:
        encoder_value: 编码器值 (通常是度数)
        
    Returns:
        float: 夹爪宽度(米)，范围[0, 0.08]
    """
    # 假设第8维是度数，转换为夹爪宽度(米)
    rad = float(encoder_value) / 180.0 * np.pi
    w = abs(rad) * 18.0 / 1000.0 * 2.0
    return float(np.clip(w, 0.0, 0.08))


def convert_gripper_width_to_encoder(width):
    """
    将夹爪宽度(米)转换为编码器值
    
    Args:
        width: 夹爪宽度(米)
        
    Returns:
        float: 编码器值
    """
    # 反向转换
    w = np.clip(width, 0.0, 0.08)
    rad = w / (18.0 / 1000.0 * 2.0)
    encoder_value = rad * 180.0 / np.pi
    return float(encoder_value)


def limit_gripper_step(current_width, target_width, max_step=0.010):
    """
    限制夹爪每步变化量
    
    Args:
        current_width: 当前夹爪宽度(米)
        target_width: 目标夹爪宽度(米)
        max_step: 最大步长(米)
        
    Returns:
        tuple: (next_width, need_more)
            - next_width: 下一步夹爪宽度
            - need_more: 是否需要继续移动
    """
    current_width = float(current_width)
    target_width = float(target_width)
    delta = np.clip(target_width - current_width, -max_step, max_step)
    next_width = float(np.clip(current_width + delta, 0.0, 0.08))
    need_more = abs(target_width - next_width) > 1e-5
    return next_width, need_more
