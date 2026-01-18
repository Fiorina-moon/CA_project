"""
动画控制器
负责动画播放、时间控制和骨架姿态更新
"""
import numpy as np
from typing import Optional

from src.animation.keyframe import AnimationClip
from src.animation.interpolation import find_keyframe_interval, interpolate_keyframe
from src.core.skeleton import Skeleton


class Animator:
    """
    动画控制器
    
    管理动画片段的播放、暂停、循环等状态
    负责根据时间插值关键帧并更新骨架姿态
    """
    
    def __init__(self, skeleton: Skeleton):
        """
        初始化动画控制器
        
        Args:
            skeleton: 要控制的骨架
        """
        self.skeleton = skeleton
        self.current_clip: Optional[AnimationClip] = None
        self.current_time: float = 0.0
        self.is_playing: bool = False
        self.loop: bool = True
    
    # ===== 动画片段管理 =====
    
    def load_clip(self, clip: AnimationClip):
        """
        加载动画片段
        
        Args:
            clip: 要加载的动画片段
        """
        self.current_clip = clip
        self.current_time = 0.0
        print(f"✓ 加载动画: {clip.name} ({clip.duration:.2f}s)")
    
    # ===== 播放控制 =====
    
    def play(self):
        """开始播放动画"""
        self.is_playing = True
    
    def pause(self):
        """暂停动画"""
        self.is_playing = False
    
    def stop(self):
        """停止动画并重置到起点"""
        self.is_playing = False
        self.current_time = 0.0
    
    # ===== 时间控制 =====
    
    def get_current_time(self) -> float:
        """
        获取当前播放时间
        
        Returns:
            当前时间（秒）
        """
        return self.current_time
    
    def set_time(self, time: float):
        """
        设置当前播放时间
        
        Args:
            time: 目标时间（秒）
        """
        if self.current_clip:
            self.current_time = max(0.0, min(time, self.current_clip.duration))
        else:
            self.current_time = max(0.0, time)
        
        self._update_skeleton_pose()
    
    # ===== 更新逻辑 =====
    
    def update(self, delta_time: float):
        """
        更新动画状态（每帧调用）
        
        Args:
            delta_time: 时间增量（秒）
        """
        if not self.is_playing or not self.current_clip:
            return
        
        # 更新时间
        self.current_time += delta_time
        
        # 循环处理
        if self.current_time > self.current_clip.duration:
            if self.loop:
                self.current_time = self.current_time % self.current_clip.duration
            else:
                self.current_time = self.current_clip.duration
                self.is_playing = False
        
        # 更新骨架姿态
        self._update_skeleton_pose()
    
    def _update_skeleton_pose(self):
        """
        根据当前时间更新骨架姿态
        
        流程：
            1. 对每个有动画的关节查找关键帧区间
            2. 插值计算当前时刻的变换
            3. 设置关节局部变换
            4. 更新全局变换
        """
        if not self.current_clip:
            return
        
        # 对每个有动画的关节进行插值
        for joint_name in self.current_clip.get_joint_names():
            joint = self.skeleton.joint_map.get(joint_name)
            if not joint:
                continue
            
            # 获取关键帧序列
            keyframes = self.current_clip.get_keyframes(joint_name)
            
            # 查找当前时间所在的关键帧区间
            kf0, kf1, blend = find_keyframe_interval(keyframes, self.current_time)
            
            if kf0 is None:
                continue
            
            # 插值计算当前变换
            if kf0 == kf1:
                # 精确命中关键帧或超出范围
                current_kf = kf0
            else:
                # 在两个关键帧之间，需要插值
                current_kf = interpolate_keyframe(kf0, kf1, blend)
            
            # 应用变换到关节
            joint.local_transform = current_kf.get_transform_matrix()
        
        # 更新全局变换
        self.skeleton.update_global_transforms()
