"""
特征融合模块
将颜色特征和纹理特征进行融合
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def fuse_features(color_features, texture_features, fusion_method='concat', 
                  color_weight=0.5, texture_weight=0.5):
    """
    融合颜色特征和纹理特征
    
    Args:
        color_features: 颜色特征 (N, D1)
        texture_features: 纹理特征 (N, D2)
        fusion_method: 融合方法
            - 'concat': 简单拼接
            - 'weighted': 加权拼接（需要先标准化）
            - 'normalized_concat': 标准化后拼接
        color_weight: 颜色特征权重（仅用于weighted方法）
        texture_weight: 纹理特征权重（仅用于weighted方法）
        
    Returns:
        fused_features: 融合后的特征 (N, D1+D2) 或 (N, D1+D2)
    """
    assert color_features.shape[0] == texture_features.shape[0], \
        "颜色特征和纹理特征的样本数必须相同"
    
    if fusion_method == 'concat':
        # 简单拼接
        fused_features = np.hstack([color_features, texture_features])
        
    elif fusion_method == 'normalized_concat':
        # 标准化后拼接
        scaler_color = StandardScaler()
        scaler_texture = StandardScaler()
        
        color_norm = scaler_color.fit_transform(color_features)
        texture_norm = scaler_texture.fit_transform(texture_features)
        
        fused_features = np.hstack([color_norm, texture_norm])
        
    elif fusion_method == 'weighted':
        # 加权拼接（先标准化）
        scaler_color = StandardScaler()
        scaler_texture = StandardScaler()
        
        color_norm = scaler_color.fit_transform(color_features)
        texture_norm = scaler_texture.fit_transform(texture_features)
        
        # 加权组合
        color_weighted = color_weight * color_norm
        texture_weighted = texture_weight * texture_norm
        
        fused_features = np.hstack([color_weighted, texture_weighted])
        
    else:
        raise ValueError(f"不支持的融合方法: {fusion_method}")
    
    return fused_features.astype(np.float32)

