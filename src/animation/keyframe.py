"""
关键帧数据结构
"""
from typing import Dict, List
from utils.math_utils import Vector3, Matrix4  # 改为绝对导入


class JointKeyframe:
    """单个关节的关键帧"""
    
    def __init__(self, time: float, rotation: tuple = (0, 0, 0), 
                 translation: tuple = (0, 0, 0), scale: tuple = (1, 1, 1)):
        """
        Args:
            time: 时间点（秒）
            rotation: 旋转角度（弧度）(rx, ry, rz)
            translation: 平移 (tx, ty, tz)
            scale: 缩放 (sx, sy, sz)
        """
        self.time = time
        self.rotation = rotation
        self.translation = translation
        self.scale = scale
    
    def get_transform_matrix(self) -> Matrix4:
        """获取变换矩阵"""
        # 平移矩阵
        T = Matrix4.translation(*self.translation)
        
        # 旋转矩阵（按Z-Y-X顺序）
        Rx = Matrix4.rotation_x(self.rotation[0])
        Ry = Matrix4.rotation_y(self.rotation[1])
        Rz = Matrix4.rotation_z(self.rotation[2])
        R = Rz * Ry * Rx
        
        # 组合：T * R（先旋转后平移）
        return T * R
    
    def __repr__(self) -> str:
        return f"Keyframe(t={self.time:.2f}, rot={self.rotation}, trans={self.translation})"


class AnimationClip:
    """动画片段"""
    
    def __init__(self, name: str, duration: float = 1.0):
        """
        Args:
            name: 动画名称
            duration: 持续时间（秒）
        """
        self.name = name
        self.duration = duration
        
        # 关键帧数据：{joint_name: [keyframes]}
        self.keyframes: Dict[str, List[JointKeyframe]] = {}
    
    def add_keyframe(self, joint_name: str, keyframe: JointKeyframe):
        """添加关键帧"""
        if joint_name not in self.keyframes:
            self.keyframes[joint_name] = []
        self.keyframes[joint_name].append(keyframe)
        
        # 按时间排序
        self.keyframes[joint_name].sort(key=lambda k: k.time)
    
    def get_keyframes(self, joint_name: str) -> List[JointKeyframe]:
        """获取指定关节的关键帧"""
        return self.keyframes.get(joint_name, [])
    
    def get_joint_names(self) -> List[str]:
        """获取所有有动画的关节名称"""
        return list(self.keyframes.keys())
    
    def __repr__(self) -> str:
        return f"AnimationClip(name={self.name}, duration={self.duration}s, joints={len(self.keyframes)})"