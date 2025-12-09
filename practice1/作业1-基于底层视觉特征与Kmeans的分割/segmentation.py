"""
图像分割模块
使用K-Means聚类算法进行图像分割
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def kmeans_segmentation(features, n_clusters=5, random_state=42, normalize=True):
    """
    使用K-Means进行图像分割
    
    Args:
        features: 特征矩阵 (N, D)，N是像素数，D是特征维度
        n_clusters: 聚类数量
        random_state: 随机种子
        normalize: 是否对特征进行标准化
        
    Returns:
        labels: 聚类标签 (N,)，每个像素的类别
        centers: 聚类中心 (n_clusters, D)
    """
    # 检查特征是否有效
    if features.size == 0:
        raise ValueError("特征矩阵为空")
    
    # 处理NaN和Inf值
    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # 特征标准化
    if normalize:
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        # 再次处理标准化后可能出现的NaN
        features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=1e10, neginf=-1e10)
    else:
        features_normalized = features
        scaler = None
    
    # K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_normalized)
    centers = kmeans.cluster_centers_
    
    return labels, centers, scaler


def reshape_labels(labels, image_shape):
    """
    将标签重塑为图像形状
    
    Args:
        labels: 标签向量 (H*W,)
        image_shape: 图像形状 (H, W)
        
    Returns:
        label_image: 标签图像 (H, W)
    """
    h, w = image_shape[:2]
    label_image = labels.reshape(h, w)
    return label_image


def visualize_segmentation(image, label_image, alpha=0.6):
    """
    可视化分割结果
    
    Args:
        image: 原始图像 (H, W, 3)
        label_image: 标签图像 (H, W)
        alpha: 透明度
        
    Returns:
        vis_image: 可视化图像
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # 创建颜色映射
    n_clusters = len(np.unique(label_image))
    try:
        # 新版本 matplotlib API
        colors = plt.colormaps['tab10'].resampled(n_clusters)
    except (AttributeError, KeyError):
        # 兼容旧版本
        colors = plt.cm.get_cmap('tab10', n_clusters)
    
    # 将标签映射到颜色，避免除零警告
    max_label = np.max(label_image)
    if max_label > 0:
        colored_labels = colors(label_image / max_label)
    else:
        colored_labels = colors(np.zeros_like(label_image))
    
    # 混合原图和分割结果
    vis_image = (1 - alpha) * image + alpha * (colored_labels[:, :, :3] * 255)
    vis_image = vis_image.astype(np.uint8)
    
    return vis_image

