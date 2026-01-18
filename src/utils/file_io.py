"""
文件读写工具
"""
import json
import numpy as np
from pathlib import Path


def save_weights(weights: np.ndarray, filepath: Path, metadata: dict = None):
    """
    保存权重数据
    
    Args:
        weights: 权重矩阵 (N × M)
        filepath: 保存路径
        metadata: 元数据（可选）
    """
    data = {
        "shape": weights.shape,
        "weights": weights.tolist(),
        "metadata": metadata or {}
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ 权重已保存到: {filepath}")


def load_weights(filepath: Path) -> tuple:
    """
    加载权重数据
    
    Args:
        filepath: 文件路径
    
    Returns:
        (weights, metadata)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    weights = np.array(data["weights"], dtype=np.float32)
    metadata = data.get("metadata", {})
    
    print(f"✓ 权重已加载: {filepath}")
    print(f"  形状: {weights.shape}")
    
    return weights, metadata


def save_weights_npz(weights: np.ndarray, filepath: Path, **kwargs):
    """
    以NPZ格式保存权重（更高效）
    
    Args:
        weights: 权重矩阵
        filepath: 保存路径
        **kwargs: 其他要保存的数组
    """
    np.savez_compressed(filepath, weights=weights, **kwargs)
    print(f"✓ 权重已保存到: {filepath}")


def load_weights_npz(filepath: Path) -> np.ndarray:
    """
    加载NPZ格式权重
    
    Args:
        filepath: 文件路径
    
    Returns:
        权重矩阵
    """
    data = np.load(filepath)
    weights = data['weights']
    print(f"✓ 权重已加载: {filepath}")
    print(f"  形状: {weights.shape}")
    return weights


def save_animation(clip, filepath: Path):
    """
    保存动画数据
    
    Args:
        clip: AnimationClip对象
        filepath: 保存路径
    """
    data = {
        "name": clip.name,
        "duration": clip.duration,
        "keyframes": {}
    }
    
    for joint_name, keyframes in clip.keyframes.items():
        data["keyframes"][joint_name] = [
            {
                "time": kf.time,
                "rotation": kf.rotation,
                "translation": kf.translation,
                "scale": kf.scale
            }
            for kf in keyframes
        ]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ 动画已保存到: {filepath}")


def load_animation(filepath: Path):
    """
    加载动画数据
    
    Args:
        filepath: 文件路径
    
    Returns:
        AnimationClip对象
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 延迟导入，避免循环依赖
    import importlib
    keyframe_module = importlib.import_module('src.animation.keyframe')
    AnimationClip = keyframe_module.AnimationClip
    JointKeyframe = keyframe_module.JointKeyframe
        
    clip = AnimationClip(data["name"], data["duration"])
    
    for joint_name, keyframes_data in data["keyframes"].items():
        for kf_data in keyframes_data:
            keyframe = JointKeyframe(
                kf_data["time"],
                tuple(kf_data["rotation"]),
                tuple(kf_data["translation"]),
                tuple(kf_data["scale"])
            )
            clip.add_keyframe(joint_name, keyframe)
    
    print(f"✓ 动画已加载: {filepath}")
    return clip