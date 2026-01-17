"""
关键帧数据结构
"""
from typing import Dict, List
from utils.math_utils import Vector3, Matrix4


class JointKeyframe:
    """单个关节的关键帧"""
    
    # 🔧 类级别的旋转放大系数（所有关键帧共享）
    ROTATION_SCALE = 1.0  # 🎯 关键参数！调整这个值：2.0-3.5
    
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
        """
        获取变换矩阵（TRS顺序）
        
        Returns:
            4x4变换矩阵 = T * R * S
        """
        import math
        
        # 🔧 放大旋转（保持符号）
        scaled_rotation = tuple(r * self.ROTATION_SCALE for r in self.rotation)
        
        # 🔧 调试输出（可选，运行时会打印很多信息）
        # print(f"旋转值: {self.rotation}")
        # print(f"放大后: {scaled_rotation}")
        # print(f"转成角度: {[math.degrees(r) for r in scaled_rotation]}")
        
        # 1. 缩放矩阵
        S = Matrix4.scale(self.scale[0], self.scale[1], self.scale[2])
        
        # 2. 旋转矩阵（使用放大后的旋转）
        Rx = Matrix4.rotation_x(scaled_rotation[0])  # 🔧 使用 scaled_rotation
        Ry = Matrix4.rotation_y(scaled_rotation[1])  # 🔧 使用 scaled_rotation
        Rz = Matrix4.rotation_z(scaled_rotation[2])  # 🔧 使用 scaled_rotation
        R = Rz * Ry * Rx  # Blender的XYZ顺序
        
        # 3. 平移矩阵
        T = Matrix4.translation(self.translation[0], self.translation[1], self.translation[2])
        
        # 4. 组合：先缩放，再旋转，最后平移
        return T * R * S
    
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
