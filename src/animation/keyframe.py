"""
关键帧数据结构
定义动画的时间采样点和变换数据
"""
from typing import Dict, List
from src.utils.math_utils import Matrix4


class JointKeyframe:
    """
    单个关节的关键帧
    
    存储某一时刻关节的局部变换（TRS分量）
    """
    
    # 旋转放大系数（全局调整动画幅度）
    ROTATION_SCALE = 1.0  # 调整范围：1.0-3.5
    
    def __init__(self, time: float, rotation: tuple = (0, 0, 0), 
                 translation: tuple = (0, 0, 0), scale: tuple = (1, 1, 1)):
        """
        初始化关键帧
        
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
        计算变换矩阵（TRS顺序）
        
        Returns:
            4x4变换矩阵 = T * R * S
        
        Note:
            - 旋转顺序为 Z*Y*X（Blender XYZ Euler）
            - 应用了 ROTATION_SCALE 放大系数
        """
        # 放大旋转（用于调整动画幅度）
        scaled_rotation = tuple(r * self.ROTATION_SCALE for r in self.rotation)
        
        # 1. 缩放矩阵
        S = Matrix4.scale(self.scale[0], self.scale[1], self.scale[2])
        
        # 2. 旋转矩阵（XYZ Euler顺序）
        Rx = Matrix4.rotation_x(scaled_rotation[0])
        Ry = Matrix4.rotation_y(scaled_rotation[1])
        Rz = Matrix4.rotation_z(scaled_rotation[2])
        R = Rz * Ry * Rx  # Blender的XYZ顺序
        
        # 3. 平移矩阵
        T = Matrix4.translation(self.translation[0], self.translation[1], self.translation[2])
        
        # 4. 组合变换：先缩放，再旋转，最后平移
        return T * R * S
    
    def __repr__(self) -> str:
        return (f"Keyframe(t={self.time:.2f}, "
                f"rot={self.rotation}, "
                f"trans={self.translation})")


class AnimationClip:
    """
    动画片段
    
    存储一段动画的所有关节的关键帧序列
    """
    
    def __init__(self, name: str, duration: float = 1.0):
        """
        初始化动画片段
        
        Args:
            name: 动画名称
            duration: 持续时间（秒）
        """
        self.name = name
        self.duration = duration
        
        # 关键帧数据：{joint_name: [keyframes]}
        self.keyframes: Dict[str, List[JointKeyframe]] = {}
    
    def add_keyframe(self, joint_name: str, keyframe: JointKeyframe):
        """
        添加关键帧
        
        Args:
            joint_name: 关节名称
            keyframe: 关键帧数据
        
        Note:
            - 会自动按时间排序
            - 同一时间点的关键帧会被后者覆盖
        """
        if joint_name not in self.keyframes:
            self.keyframes[joint_name] = []
        
        self.keyframes[joint_name].append(keyframe)
        
        # 按时间排序（保证插值时能正确查找区间）
        self.keyframes[joint_name].sort(key=lambda kf: kf.time)
    
    def get_keyframes(self, joint_name: str) -> List[JointKeyframe]:
        """
        获取指定关节的关键帧序列
        
        Args:
            joint_name: 关节名称
        
        Returns:
            关键帧列表（按时间排序），如果关节不存在则返回空列表
        """
        return self.keyframes.get(joint_name, [])
    
    def get_joint_names(self) -> List[str]:
        """
        获取所有有动画的关节名称
        
        Returns:
            关节名称列表
        """
        return list(self.keyframes.keys())
    
    def __repr__(self) -> str:
        return (f"AnimationClip(name={self.name}, "
                f"duration={self.duration}s, "
                f"joints={len(self.keyframes)})")
