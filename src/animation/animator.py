"""
动画控制器
"""
import numpy as np
from typing import Dict
from animation.keyframe import AnimationClip, JointKeyframe  # 改为绝对导入
from animation.interpolation import find_keyframe_interval, interpolate_keyframe  # 改为绝对导入
from core.skeleton import Skeleton  # 改为绝对导入
from utils.math_utils import Matrix4,Vector3  # 改为绝对导入


class Animator:
    """动画控制器"""
    
    def __init__(self, skeleton: Skeleton):
        """
        Args:
            skeleton: 骨架
        """
        self.skeleton = skeleton
        self.current_clip: AnimationClip = None
        self.current_time: float = 0.0
        self.is_playing: bool = False
        self.loop: bool = True
    
    def load_clip(self, clip: AnimationClip):
        """加载动画片段"""
        self.current_clip = clip
        self.current_time = 0.0
        print(f"✓ 加载动画: {clip.name} ({clip.duration}s)")
    
    def play(self):
        """播放动画"""
        self.is_playing = True
    
    def pause(self):
        """暂停动画"""
        self.is_playing = False
    
    def stop(self):
        """停止动画"""
        self.is_playing = False
        self.current_time = 0.0
    
    def update(self, delta_time: float):
        """
        更新动画状态
        
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
        """更新骨架姿态"""
        if not self.current_clip:
            return
        
        # 对每个有动画的关节进行插值
        for joint_name in self.current_clip.get_joint_names():
            joint = self.skeleton.joint_map.get(joint_name)
            if not joint:
                continue
            
            # 获取关键帧
            keyframes = self.current_clip.get_keyframes(joint_name)
            kf0, kf1, blend = find_keyframe_interval(keyframes, self.current_time)
            
            if kf0 is None:
                continue
            
            # 插值
            current_kf = kf0 if kf0 == kf1 else interpolate_keyframe(kf0, kf1, blend)
            
            # 设置局部变换
            joint.local_transform = current_kf.get_transform_matrix()
        
        # 更新全局变换
        self.skeleton.update_global_transforms()
        
        # 调试
        if int(self.current_time * 10) % 10 == 0:
            test_joint = self.skeleton.joint_map.get("RigLBLeg2_04")
            if test_joint:
                print(f"  [调试] t={self.current_time:.2f}s, 位置=({test_joint.current_position.x:.3f}, {test_joint.current_position.y:.3f}, {test_joint.current_position.z:.3f})")
    
    def get_current_time(self) -> float:
        """获取当前时间"""
        return self.current_time
    
    def set_time(self, time: float):
        """设置当前时间"""
        self.current_time = max(0.0, min(time, self.current_clip.duration if self.current_clip else 0.0))
        self._update_skeleton_pose()