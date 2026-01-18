"""
插值算法
"""
from typing import List
from .keyframe import JointKeyframe
from src.utils.math_utils import lerp  


def find_keyframe_interval(keyframes: List[JointKeyframe], time: float) -> tuple:
    """
    找到时间点所在的关键帧区间
    
    Args:
        keyframes: 关键帧列表（已按时间排序）
        time: 当前时间
    
    Returns:
        (prev_keyframe, next_keyframe, blend_factor)
    """
    if not keyframes:
        return None, None, 0.0
    
    # 如果只有一个关键帧
    if len(keyframes) == 1:
        return keyframes[0], keyframes[0], 0.0
    
    # 在范围之前
    if time <= keyframes[0].time:
        return keyframes[0], keyframes[0], 0.0
    
    # 在范围之后
    if time >= keyframes[-1].time:
        return keyframes[-1], keyframes[-1], 0.0
    
    # 查找区间
    for i in range(len(keyframes) - 1):
        if keyframes[i].time <= time <= keyframes[i + 1].time:
            t0 = keyframes[i].time
            t1 = keyframes[i + 1].time
            
            # 计算混合因子
            blend = (time - t0) / (t1 - t0) if t1 > t0 else 0.0
            
            return keyframes[i], keyframes[i + 1], blend
    
    return keyframes[-1], keyframes[-1], 0.0


def interpolate_keyframe(kf0: JointKeyframe, kf1: JointKeyframe, t: float) -> JointKeyframe:
    """
    在两个关键帧之间进行线性插值
    
    Args:
        kf0: 起始关键帧
        kf1: 结束关键帧
        t: 混合因子 [0, 1]
    
    Returns:
        插值后的关键帧
    """
    # 插值旋转
    rotation = (
        lerp(kf0.rotation[0], kf1.rotation[0], t),
        lerp(kf0.rotation[1], kf1.rotation[1], t),
        lerp(kf0.rotation[2], kf1.rotation[2], t)
    )
    
    # 插值平移
    translation = (
        lerp(kf0.translation[0], kf1.translation[0], t),
        lerp(kf0.translation[1], kf1.translation[1], t),
        lerp(kf0.translation[2], kf1.translation[2], t)
    )
    
    # 插值缩放
    scale = (
        lerp(kf0.scale[0], kf1.scale[0], t),
        lerp(kf0.scale[1], kf1.scale[1], t),
        lerp(kf0.scale[2], kf1.scale[2], t)
    )
    
    # 创建插值关键帧
    time = lerp(kf0.time, kf1.time, t)
    return JointKeyframe(time, rotation, translation, scale)