"""
骨架数据结构 - 修复版（支持完整LBS）
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
        
        # LBS关键：绑定姿态矩阵
        self.bind_matrix = Matrix4.identity()  # 绑定姿态的全局变换
        self.inverse_bind_matrix = Matrix4.identity()  # 绑定姿态逆矩阵
        
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
    
    def get_start_position(self) -> Vector3:
        """获取起点位置（绑定姿态）"""
        return self.start_joint.head
    
    def get_end_position(self) -> Vector3:
        """获取终点位置（绑定姿态）"""
        return self.end_joint.head


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
        """构建层级关系并计算绑定姿态矩阵"""
        for joint in self.joints:
            if joint.parent_name:
                parent = self.joint_map.get(joint.parent_name)
                if parent:
                    joint.parent = parent
                    parent.children.append(joint)
            else:
                if self.root_joint is None:
                    self.root_joint = joint
        
        # 计算绑定姿态矩阵（递归从根节点开始）
        self._compute_bind_matrices()
        
        # 初始化当前变换为绑定姿态
        self._init_transforms()
    
    def _compute_bind_matrices(self, joint: Joint = None):
        """
        计算绑定姿态矩阵
        bind_matrix = 关节在绑定姿态下的全局变换
        """
        if joint is None:
            joint = self.root_joint
            if joint is None:
                return
        
        if joint.parent:
            # 子关节相对父关节的偏移
            offset = joint.head - joint.parent.head
            offset_matrix = Matrix4.translation(offset.x, offset.y, offset.z)
            # 绑定姿态：bind = parent.bind × offset
            joint.bind_matrix = joint.parent.bind_matrix * offset_matrix
        else:
            # 根节点：bind = 平移到head位置
            joint.bind_matrix = Matrix4.translation(joint.head.x, joint.head.y, joint.head.z)
        
        # 计算逆矩阵（LBS公式需要）
        joint.inverse_bind_matrix = joint.bind_matrix.inverse()
        
        # 递归处理子节点
        for child in joint.children:
            self._compute_bind_matrices(child)
    
    def _init_transforms(self):
        """初始化变换 - 设置为绑定姿态"""
        for joint in self.joints:
            # 初始全局变换 = 绑定姿态
            joint.global_transform = Matrix4(joint.bind_matrix.data.copy())
            joint.current_position = Vector3(joint.head.x, joint.head.y, joint.head.z)
    
    def build_bones(self):
        self.bones.clear()
        for joint in self.joints:
            if joint.parent:
                self.bones.append(Bone(joint.parent, joint, len(self.bones)))
    
    def update_global_transforms(self, joint: Joint = None):
        """
        更新全局变换（动画）
        
        关键公式：
        global = bind × local  （根节点）
        global = parent.global × offset × local  （子节点）
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
        
        # 从全局变换提取位置（用于可视化）
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