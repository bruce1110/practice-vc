"""
颜色特征提取模块
支持RGB和HSV颜色空间的特征提取
"""

import numpy as np
import cv2


def extract_rgb_features(image):
    """
    从RGB颜色空间提取像素级颜色特征
    
    Args:
        image: 输入图像 (H, W, 3) BGR格式
        
    Returns:
        features: 特征矩阵 (H*W, 3)，每行是一个像素的RGB特征
    """
    # 将BGR转换为RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = rgb_image.shape
    
    # 将图像重塑为 (H*W, 3)
    features = rgb_image.reshape(-1, 3).astype(np.float32)
    
    return features


def extract_hsv_features(image):
    """
    从HSV颜色空间提取像素级颜色特征
    
    Args:
        image: 输入图像 (H, W, 3) BGR格式
        
    Returns:
        features: 特征矩阵 (H*W, 3)，每行是一个像素的HSV特征
    """
    # 将BGR转换为HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w, c = hsv_image.shape
    
    # 将图像重塑为 (H*W, 3)
    features = hsv_image.reshape(-1, 3).astype(np.float32)
    
    return features


def extract_color_features(image, color_space='rgb'):
    """
    提取颜色特征的统一接口
    
    Args:
        image: 输入图像 (H, W, 3) BGR格式
        color_space: 颜色空间，'rgb' 或 'hsv'
        
    Returns:
        features: 特征矩阵 (H*W, 3)
    """
    if color_space.lower() == 'rgb':
        return extract_rgb_features(image)
    elif color_space.lower() == 'hsv':
        return extract_hsv_features(image)
    else:
        raise ValueError(f"不支持的颜色空间: {color_space}，请选择 'rgb' 或 'hsv'")

