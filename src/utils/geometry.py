"""
几何计算工具：点到线段距离等
"""
import numpy as np
from .math_utils import Vector3


def point_to_segment_distance(point: Vector3, seg_start: Vector3, seg_end: Vector3) -> float:
    """
    计算点到线段的最短距离
    
    Args:
        point: 待计算的点
        seg_start: 线段起点
        seg_end: 线段终点
    
    Returns:
        最短距离
    """
    # 线段向量
    ab = seg_end - seg_start
    ap = point - seg_start
    
    # 线段长度平方
    ab_squared = Vector3.dot(ab, ab)
    
    # 处理退化情况（起点终点重合）
    if ab_squared < 1e-10:
        return Vector3.distance(point, seg_start)
    
    # 投影参数 t
    t = Vector3.dot(ap, ab) / ab_squared
    
    # 限制在[0, 1]范围内（确保在线段上）
    t = max(0.0, min(1.0, t))
    
    # 线段上最近点
    closest_point = seg_start + ab * t
    
    # 返回距离
    return Vector3.distance(point, closest_point)


def point_to_segment_closest_point(point: Vector3, seg_start: Vector3, seg_end: Vector3) -> Vector3:
    """
    计算点在线段上的最近点
    
    Args:
        point: 待计算的点
        seg_start: 线段起点
        seg_end: 线段终点
    
    Returns:
        线段上的最近点
    """
    ab = seg_end - seg_start
    ap = point - seg_start
    
    ab_squared = Vector3.dot(ab, ab)
    
    if ab_squared < 1e-10:
        return seg_start
    
    t = Vector3.dot(ap, ab) / ab_squared
    t = max(0.0, min(1.0, t))
    
    return seg_start + ab * t