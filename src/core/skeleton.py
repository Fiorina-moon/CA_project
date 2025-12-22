"""
骨架数据结构 - 修复版
"""
from typing import List, Dict, Optional
import numpy as np
from utils.math_utils import Vector3, Matrix4


class Joint:
    """关节点类"""
    
    def __init__(self, name: str, index: int, head: Vector3, tail: Vector3, parent: Optional[str] = None):
        self.name = name
        self.index = index
        self.head = head  # 绑定姿态位置（世界空间）
        self.tail = tail
        self.parent_name = parent
        
        # 层级关系
        self.parent: Optional['Joint'] = None
        self.children: List['Joint'] = []
        
        # 变换
        self.local_transform = Matrix4.identity()  # 局部动画变换
        self.global_transform = Matrix4.identity()  # 当前全局变换
        
        # 位置
        self.current_position = Vector3(head.x, head.y, head.z)
    
    def __repr__(self) -> str:
        return f"Joint({self.name})"


class Bone:
    """骨骼类"""
    
    def __init__(self, start_joint: Joint, end_joint: Joint, index: int):
        self.start_joint = start_joint
        self.end_joint = end_joint
        self.index = index
        self.name = f"{start_joint.name}_to_{end_joint.name}"


class Skeleton:
    """骨架类"""
    
    def __init__(self):
        self.joints: List[Joint] = []
        self.bones: List[Bone] = []
        self.root_joint: Optional[Joint] = None
        self.joint_map: Dict[str, Joint] = {}
        self.joint_index_map: Dict[int, Joint] = {}
    
    def add_joint(self, joint: Joint):
        self.joints.append(joint)
        self.joint_map[joint.name] = joint
        self.joint_index_map[joint.index] = joint
    
    def build_hierarchy(self):
        for joint in self.joints:
            if joint.parent_name:
                parent = self.joint_map.get(joint.parent_name)
                if parent:
                    joint.parent = parent
                    parent.children.append(joint)
            else:
                if self.root_joint is None:
                    self.root_joint = joint
        
        # 初始化全局变换为绑定姿态
        self._init_transforms()
    
    def _init_transforms(self):
        """初始化变换 - 设置为绑定姿态"""
        for joint in self.joints:
            # 全局变换 = 平移到绑定位置
            joint.global_transform = Matrix4.translation(joint.head.x, joint.head.y, joint.head.z)
            joint.current_position = Vector3(joint.head.x, joint.head.y, joint.head.z)
    
    def build_bones(self):
        self.bones.clear()
        for joint in self.joints:
            if joint.parent:
                self.bones.append(Bone(joint.parent, joint, len(self.bones)))
    
    def update_global_transforms(self, joint: Joint = None):
        """
        更新全局变换
        
        关键：
        - 绑定姿态时：global = T(head)
        - 动画时：global = T(head) × local
        - 父子关系：global = parent.global × T(offset) × local
        """
        if joint is None:
            joint = self.root_joint
            if joint is None:
                return
        
        if joint.parent:
            # 子关节位置相对于父关节的偏移
            offset = joint.head - joint.parent.head
            offset_matrix = Matrix4.translation(offset.x, offset.y, offset.z)
            
            # 全局变换 = 父全局 × 偏移 × 局部动画
            joint.global_transform = joint.parent.global_transform * offset_matrix * joint.local_transform
        else:
            # 根节点：全局变换 = 绑定位置 × 局部动画
            bind_matrix = Matrix4.translation(joint.head.x, joint.head.y, joint.head.z)
            joint.global_transform = bind_matrix * joint.local_transform
        
        # 从全局变换提取位置
        joint.current_position = Vector3(
            joint.global_transform.data[0, 3],
            joint.global_transform.data[1, 3],
            joint.global_transform.data[2, 3]
        )
        
        # 递归更新子节点
        for child in joint.children:
            self.update_global_transforms(child)
    
    def get_joint_count(self) -> int:
        return len(self.joints)
    
    def get_bone_count(self) -> int:
        return len(self.bones)
    
    def __repr__(self) -> str:
        return f"Skeleton(joints={len(self.joints)}, bones={len(self.bones)})"