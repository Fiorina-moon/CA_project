"""
动画模块
包含关键帧、插值、动画控制等功能
"""

from .keyframe import JointKeyframe, AnimationClip
from .interpolation import find_keyframe_interval, interpolate_keyframe
from .animator import Animator

__all__ = [
    'JointKeyframe',
    'AnimationClip',
    'find_keyframe_interval',
    'interpolate_keyframe',
    'Animator',
]
