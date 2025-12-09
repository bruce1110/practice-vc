"""
纹理特征提取模块
基于RFS (Rotation and Scale Invariant) 滤波器组提取纹理特征
"""

import numpy as np
import cv2
import scipy.io as sio


def make_rfs_filters():
    """
    创建RFS滤波器组
    参考MATLAB的makeRFSfilters.m实现
    
    Returns:
        F: 滤波器组 (49, 49, 38)
    """
    SUP = 49  # 最大滤波器的支撑尺寸（必须是奇数）
    SCALEX = [1, 2, 4]  # 方向滤波器的Sigma_x
    NORIENT = 6  # 方向数量
    
    NROTINV = 2
    NBAR = len(SCALEX) * NORIENT
    NEDGE = len(SCALEX) * NORIENT
    NF = NBAR + NEDGE + NROTINV
    F = np.zeros((SUP, SUP, NF))
    
    hsup = (SUP - 1) // 2
    x, y = np.meshgrid(np.arange(-hsup, hsup + 1), np.arange(hsup, -hsup - 1, -1))
    orgpts = np.array([x.flatten(), y.flatten()])
    
    count = 0
    for scale_idx, scale in enumerate(SCALEX):
        for orient in range(NORIENT):
            angle = np.pi * orient / NORIENT
            c = np.cos(angle)
            s = np.sin(angle)
            rot_matrix = np.array([[c, -s], [s, c]])
            rotpts = rot_matrix @ orgpts
            
            # 第一个导数（bar）
            F[:, :, count] = make_filter(scale, 0, 1, rotpts, SUP)
            # 第二个导数（edge）
            F[:, :, count + NEDGE] = make_filter(scale, 0, 2, rotpts, SUP)
            count += 1
    
    # 各向同性高斯滤波器
    F[:, :, NBAR + NEDGE] = normalise(gaussian_filter2d(SUP, 10))
    # LoG滤波器
    F[:, :, NBAR + NEDGE + 1] = normalise(log_filter(SUP, 10))
    
    return F


def make_filter(scale, phasex, phasey, pts, sup):
    """
    创建单个滤波器
    
    Args:
        scale: 尺度参数
        phasex: x方向的相位
        phasey: y方向的相位
        pts: 点坐标
        sup: 支撑尺寸
        
    Returns:
        f: 滤波器 (sup, sup)
    """
    gx = gauss1d(3 * scale, 0, pts[0, :], phasex)
    gy = gauss1d(scale, 0, pts[1, :], phasey)
    f = normalise((gx * gy).reshape(sup, sup))
    return f


def gauss1d(sigma, mean, x, ord):
    """
    计算高斯导数
    
    Args:
        sigma: 标准差
        mean: 均值
        x: 输入点
        ord: 导数阶数 (0, 1, 2)
        
    Returns:
        g: 高斯导数
    """
    # 避免除零警告
    if sigma < 1e-10:
        return np.zeros_like(x)
    
    x = x - mean
    num = x * x
    variance = sigma ** 2
    denom = 2 * variance
    # 使用 np.maximum 避免除零
    g = np.exp(-np.maximum(num, 0) / np.maximum(denom, 1e-10)) / np.sqrt(np.maximum(np.pi * denom, 1e-10))
    
    if ord == 1:
        g = -g * (x / np.maximum(variance, 1e-10))
    elif ord == 2:
        g = g * ((num - variance) / np.maximum(variance ** 2, 1e-10))
    
    return g


def normalise(f):
    """
    归一化滤波器
    
    Args:
        f: 输入滤波器
        
    Returns:
        f: 归一化后的滤波器
    """
    f = f - np.mean(f)
    sum_abs = np.sum(np.abs(f))
    # 避免除零警告
    if sum_abs > 1e-10:
        f = f / sum_abs
    else:
        # 如果所有值都接近0，返回零数组
        f = np.zeros_like(f)
    return f


def gaussian_filter2d(size, sigma):
    """
    创建2D高斯滤波器
    
    Args:
        size: 滤波器尺寸
        sigma: 标准差
        
    Returns:
        g: 高斯滤波器
    """
    h = (size - 1) // 2
    y, x = np.ogrid[-h:h+1, -h:h+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g


def log_filter(size, sigma):
    """
    创建LoG (Laplacian of Gaussian) 滤波器
    
    Args:
        size: 滤波器尺寸
        sigma: 标准差
        
    Returns:
        log_f: LoG滤波器
    """
    # 避免除零警告
    if sigma < 1e-10:
        return np.zeros((size, size))
    
    h = (size - 1) // 2
    y, x = np.ogrid[-h:h+1, -h:h+1]
    
    # 高斯
    sigma_sq = sigma ** 2
    g = np.exp(-(x**2 + y**2) / (2 * sigma_sq))
    g = g / (2 * np.pi * sigma_sq)
    
    # Laplacian
    log_f = ((x**2 + y**2 - 2 * sigma_sq) / np.maximum(sigma**4, 1e-10)) * g
    
    return log_f


def load_filter_bank(filter_bank_path=None):
    """
    加载滤波器组
    
    Args:
        filter_bank_path: 滤波器组.mat文件路径，如果为None则生成
        
    Returns:
        F: 滤波器组 (49, 49, 38)
    """
    if filter_bank_path:
        try:
            mat_data = sio.loadmat(filter_bank_path)
            F = mat_data['F']
            return F
        except:
            print(f"无法加载滤波器组文件 {filter_bank_path}，将生成新的滤波器组")
    
    return make_rfs_filters()


def extract_texture_features(image, filter_bank=None, filter_bank_path=None):
    """
    提取纹理特征
    
    Args:
        image: 输入图像 (H, W, 3) BGR格式或 (H, W) 灰度图
        filter_bank: 预计算的滤波器组，如果为None则加载或生成
        filter_bank_path: 滤波器组.mat文件路径
        
    Returns:
        features: 特征矩阵 (H*W, 38)，每行是一个像素的纹理特征
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float32)
    h, w = gray.shape
    
    # 加载或生成滤波器组
    if filter_bank is None:
        F = load_filter_bank(filter_bank_path)
    else:
        F = filter_bank
    
    num_filters = F.shape[2]
    responses = np.zeros((h, w, num_filters))
    
    # 对每个滤波器进行卷积
    for i in range(num_filters):
        # 使用valid模式卷积，保持输出尺寸
        kernel = F[:, :, i]
        # 使用scipy的convolve2d或opencv的filter2D
        response = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        # 取绝对值作为纹理响应
        responses[:, :, i] = np.abs(response)
    
    # 将响应重塑为 (H*W, num_filters)
    features = responses.reshape(-1, num_filters).astype(np.float32)
    
    return features

