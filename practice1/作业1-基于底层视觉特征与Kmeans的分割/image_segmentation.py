"""
图像分割主模块
整合颜色特征、纹理特征和K-Means分割功能
"""

from color_features import extract_color_features
from texture_features import extract_texture_features, load_filter_bank
from segmentation import kmeans_segmentation, reshape_labels
from feature_fusion import fuse_features


class ImageSegmentation:
    """图像分割类"""

    def __init__(self, filter_bank_path=None):
        """
        初始化
        
        Args:
            filter_bank_path: 滤波器组文件路径
        """
        self.filter_bank = None
        if filter_bank_path:
            self.filter_bank = load_filter_bank(filter_bank_path)
        else:
            # 尝试加载默认路径
            import os
            default_path = os.path.join(
                os.path.dirname(__file__),
                'Image Segmentation provided-files',
                'filterBank.mat'
            )
            if os.path.exists(default_path):
                self.filter_bank = load_filter_bank(default_path)
            else:
                from texture_features import make_rfs_filters
                self.filter_bank = make_rfs_filters()

    def segment_by_color(self, image, color_space='rgb', n_clusters=5, random_state=42):
        """
        基于颜色特征进行分割
        
        Args:
            image: 输入图像 (H, W, 3) BGR格式
            color_space: 颜色空间 'rgb' 或 'hsv'
            n_clusters: 聚类数量
            random_state: 随机种子
            
        Returns:
            label_image: 分割结果 (H, W)
            features: 提取的特征
        """
        # 提取颜色特征
        features = extract_color_features(image, color_space)

        # K-Means分割
        labels, centers, scaler = kmeans_segmentation(
            features, n_clusters=n_clusters, random_state=random_state
        )

        # 重塑为图像形状
        label_image = reshape_labels(labels, image.shape)

        return label_image, features

    def segment_by_texture(self, image, n_clusters=5, random_state=42):
        """
        基于纹理特征进行分割
        
        Args:
            image: 输入图像 (H, W, 3) BGR格式
            n_clusters: 聚类数量
            random_state: 随机种子
            
        Returns:
            label_image: 分割结果 (H, W)
            features: 提取的特征
        """
        # 提取纹理特征
        features = extract_texture_features(image, filter_bank=self.filter_bank)

        # K-Means分割
        labels, centers, scaler = kmeans_segmentation(
            features, n_clusters=n_clusters, random_state=random_state
        )

        # 重塑为图像形状
        label_image = reshape_labels(labels, image.shape)

        return label_image, features

    def segment_by_fused_features(self, image, color_space='rgb', n_clusters=5,
                                  fusion_method='normalized_concat',
                                  color_weight=0.5, texture_weight=0.5,
                                  random_state=42):
        """
        基于融合特征进行分割
        
        Args:
            image: 输入图像 (H, W, 3) BGR格式
            color_space: 颜色空间 'rgb' 或 'hsv'
            n_clusters: 聚类数量
            fusion_method: 融合方法
            color_weight: 颜色特征权重
            texture_weight: 纹理特征权重
            random_state: 随机种子
            
        Returns:
            label_image: 分割结果 (H, W)
            color_features: 颜色特征
            texture_features: 纹理特征
            fused_features: 融合特征
        """
        # 提取颜色特征
        color_features = extract_color_features(image, color_space)

        # 提取纹理特征
        texture_features = extract_texture_features(image, filter_bank=self.filter_bank)

        # 融合特征
        fused_features = fuse_features(
            color_features, texture_features,
            fusion_method=fusion_method,
            color_weight=color_weight,
            texture_weight=texture_weight
        )

        # K-Means分割
        labels, centers, scaler = kmeans_segmentation(
            fused_features, n_clusters=n_clusters, random_state=random_state
        )

        # 重塑为图像形状
        label_image = reshape_labels(labels, image.shape)

        return label_image, color_features, texture_features, fused_features
